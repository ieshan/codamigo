// Package localembed embeds text in-process with GoMLX, with no network calls
// and no API key.
//
// It is a second implementation of embedder.Embedder alongside
// go-embedder/openai, selected by embedding_provider: local. Weights come from
// HuggingFace via [Download] and live under ~/.codamigo/models; the model runs
// on a GoMLX compute backend chosen by [selectBackend].
//
// Construct with [New], which loads the weights and builds the inference graph
// once for the process lifetime. [Embedder.Close] releases the native compute
// resources and must be called; unlike an HTTP client, the buffers here are not
// garbage-collected.
//
// This package deliberately does not import codamigo's config package — every
// setting arrives through [Options] at construction time.
package localembed

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"golang.org/x/sync/semaphore"
)

// defaultCloseTimeout bounds how long [Embedder.Close] waits for in-flight
// calls. Long enough for a large batch on the slow pure-Go backend, short enough
// that a wedged call cannot hang `codamigo serve` on shutdown.
const defaultCloseTimeout = 30 * time.Second

// defaultConcurrency is the number of batches in flight at once.
//
// This is deliberately far below the HTTP embedder's concurrency and below
// cfg.IndexConcurrency (default 20). Inference here is CPU- or GPU-bound and
// GoMLX already parallelizes internally, so fanning out wide oversubscribes the
// machine instead of helping. Two lets tokenization of one batch overlap
// inference of another.
const defaultConcurrency = 2

// Options configures [New]. Zero-value numeric fields take built-in defaults.
type Options struct {
	// Model is a registry short name or a HuggingFace repository id. Empty uses
	// [DefaultModel].
	Model string
	// ModelsRoot is the directory holding downloaded models, normally
	// config.ModelsDir(). Required.
	ModelsRoot string
	// Backend selects the compute backend: "auto" (default), "go", "xla",
	// "xla:cpu", or "xla:cuda".
	Backend string
	// MaxSeqLen caps the token sequence length; longer inputs are truncated.
	// Zero uses the model's own maximum. Values above it are clamped down.
	MaxSeqLen int
	// BatchSize is the largest batch submitted to the backend in one call.
	// Zero defaults to 32.
	BatchSize int
	// Dimensions is required for a repository id that is not in the registry,
	// where nothing else declares the width. For a registry model it is
	// optional and only cross-checked.
	Dimensions int
	// ApplyQueryPrefix makes the returned Embedder the query side, prepending
	// the model's query instruction prefix to every text. Off by default:
	// documents must not be prefixed. Use this rather than
	// [Embedder.WithPrefix] when the query side is the only one needed, so the
	// caller holds the Embedder that owns Close.
	ApplyQueryPrefix bool
}

// Embedder embeds text with a locally-run model. It implements
// embedder.Embedder, and additionally io.Closer.
//
// An Embedder returned by [New] owns the underlying model. [Embedder.WithPrefix]
// returns a lightweight view that shares it; see that method for the lifecycle
// rules.
type Embedder struct {
	shared *shared
	prefix string
	// owner is true only for the Embedder returned by New, so Close on a
	// WithPrefix view is a no-op rather than a use-after-free for the original.
	owner bool
}

// shared holds everything the model owns. It is referenced by the Embedder from
// New and by every WithPrefix view, so exactly one set of weights, one compute
// backend, and one concurrency limit exist per process.
type shared struct {
	tokenizer api.Tokenizer
	backend   compute.Backend
	store     *model.Store
	exec      *model.Exec

	descriptor Model
	// queryPrefix is the model's own query instruction prefix, kept here so
	// [Embedder.WithPrefix] can derive the query side from a document embedder
	// that shares these weights.
	queryPrefix string
	backendName string
	padID       int32
	dim         int
	maxSeqLen   int
	seqB        []int
	batchB      []int

	// sem caps concurrent batch execution. It is only a throughput limit — Close
	// deliberately does not drain through it, because holding every slot forever
	// would strand any caller that had already passed the closed check and was
	// about to acquire one.
	sem *semaphore.Weighted

	// tokMu guards the tokenizer, whose concurrency-safety go-huggingface does
	// not document. Tokenization happens in a phase of its own, before any
	// semaphore slot is taken, so this lock is never held while acquiring sem and
	// cannot participate in a cycle.
	tokMu sync.Mutex

	// compileMu serializes graph compilation, and compiled records the shapes
	// already built.
	//
	// GoMLX singleflights compilation per shape, but building the graph for two
	// different shapes concurrently is a data race: the graph function reaches
	// gomlx/ml/zoo/transformer.populateOrderedScopes, which lazily fills in a
	// *kvcache.KVCache shared by every graph built from this model. The race
	// detector reports it on any concurrent first-compile of distinct shapes.
	// Serializing compilation costs nothing steady-state — each of the 21 shapes
	// is compiled once — and execution stays fully concurrent afterwards.
	compileMu sync.RWMutex
	compiled  map[[2]int]bool

	// mu makes "not closed" and "registered as in-flight" a single atomic step,
	// which is what lets Close know it has seen every call that will ever run.
	mu       sync.RWMutex
	closed   bool
	inFlight sync.WaitGroup

	closeOnce sync.Once
	closeErr  error
}

// New loads the model and builds its inference graph.
//
// The model must already be on disk — run `codamigo download-model` first;
// [ErrModelNotDownloaded] says so explicitly rather than starting a multi-
// hundred-megabyte download from inside an index run.
//
// The caller must call [Embedder.Close].
func New(opts Options) (*Embedder, error) {
	if opts.ModelsRoot == "" {
		return nil, errors.New("ModelsRoot must be set")
	}
	descriptor, err := Lookup(opts.Model)
	if err != nil {
		return nil, err
	}
	modelDir, err := ModelDir(opts.ModelsRoot, descriptor)
	if err != nil {
		return nil, err
	}
	// Resolve the revision before anything looks for files: the snapshot
	// directory is named after the commit hash, so MissingFiles cannot find
	// anything until the hash is known.
	descriptor, pin, err := ResolvePin(modelDir, descriptor)
	if err != nil {
		return nil, err
	}
	missing, err := MissingFiles(modelDir, descriptor)
	if err != nil {
		return nil, err
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("%w: %s is missing %d file(s) under %s (first: %s). "+
			"Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, descriptor.DisplayName(), len(missing), modelDir, missing[0], descriptor.DisplayName())
	}

	// go-huggingface re-resolves the revision over the network on every load and
	// ignores its own on-disk cache, so point it at a local shim that answers
	// from the pin instead. Every *hub.Repo access happens below, during
	// loading, so the shim can be shut down as soon as New returns.
	shim, err := startInfoShim(descriptor, pin)
	if err != nil {
		return nil, err
	}
	defer func() { _ = shim.Close() }()

	// loadErr wraps a failure from any call below that can reach *hub.Repo. If
	// the shim refused a request, that is almost always the real cause: a file
	// outside standardManifest that the local snapshot never had. Naming it beats
	// surfacing the underlying HTTP error, which just says the URL 404ed.
	loadErr := func(err error, fallback string) error {
		if missed := shim.missedPaths(); len(missed) > 0 {
			return fmt.Errorf("%w: %s is not in the local snapshot for %s. "+
				"Run: codamigo download-model --model %s",
				ErrModelNotDownloaded, missed[0], descriptor.DisplayName(), descriptor.DisplayName())
		}
		return fmt.Errorf("%s: %w", fallback, err)
	}

	repo := hub.New(descriptor.RepoID).
		WithCacheDir(modelDir).
		WithRevision(descriptor.Revision).
		WithEndpoint(shim.URL)
	repo.Verbosity = 0
	// The files are already local, but LoadModel needs the repo's file index.
	if err := repo.DownloadInfo(false); err != nil {
		return nil, fmt.Errorf("%w: reading local model info for %s: %v. Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, descriptor.DisplayName(), err, descriptor.DisplayName())
	}

	hfModel, err := transformer.LoadModel(repo)
	if err != nil {
		return nil, loadErr(err, fmt.Sprintf("loading %s", descriptor.DisplayName()))
	}
	dim := hfModel.Config.HiddenSize
	if err := checkDimensions(descriptor, opts.Dimensions, dim); err != nil {
		return nil, err
	}

	tokenizer, err := hfModel.GetTokenizer()
	if err != nil {
		return nil, loadErr(err, fmt.Sprintf("loading tokenizer for %s", descriptor.DisplayName()))
	}
	padID, err := tokenizer.SpecialTokenID(api.TokPad)
	if err != nil {
		return nil, fmt.Errorf("resolving pad token for %s: %w", descriptor.DisplayName(), err)
	}

	maxSeqLen := resolveMaxSeqLen(opts.MaxSeqLen, hfModel.Config.MaxPositionEmbeddings)
	seqB, err := seqBuckets(maxSeqLen)
	if err != nil {
		return nil, err
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	batchB, err := batchBuckets(batchSize)
	if err != nil {
		return nil, err
	}

	backend, backendName, err := selectBackend(opts.Backend)
	if err != nil {
		return nil, err
	}
	// From here on every failure must release what has been acquired, or the
	// process leaks a compute backend.
	store := model.NewStore()
	cleanup := func() {
		store.Finalize()
		backend.Finalize()
	}
	if err := hfModel.LoadStore(backend, store); err != nil {
		cleanup()
		return nil, loadErr(err, fmt.Sprintf("loading weights for %s onto %s", descriptor.DisplayName(), backendName))
	}

	// seqLen is derived inside the graph from the pad token rather than passed as
	// nil. This is not an optimization: seqLen feeds the attention mask, and with
	// nil the padded rows are attended to, silently dropping cosine similarity
	// against the reference implementation to 0.93 at 64 tokens and 0.87 at 128.
	// Derived, it matches to 1.000000.
	exec, err := model.NewExec(backend, store,
		func(scope *model.Scope, tokens *graph.Node) *graph.Node {
			pad := graph.Scalar(tokens.Graph(), tokens.DType(), float64(padID))
			seqLen := graph.ReduceSum(graph.ConvertDType(graph.NotEqual(tokens, pad), dtypes.Int32), 1)
			return graph.ConvertDType(hfModel.SentenceEmbeddingGraph(scope, tokens, seqLen), dtypes.Float32)
		})
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("building inference graph for %s: %w", descriptor.DisplayName(), err)
	}
	// Sizing the cache to the closed shape set is a correctness guard, not a
	// memory one: GoMLX never evicts, it refuses ("maximum cache size of N
	// reached"). So a bucketize regression that emits an unplanned shape fails
	// loudly instead of growing the cache. Compilation stays lazy — precompiling
	// all 21 shapes costs 6.2s on XLA, paid on every command invocation.
	exec.SetMaxCache(len(seqB) * len(batchB))

	// The prefix belongs to the Embedder, not to shared: the document side must
	// stay unprefixed even when a query-side view shares the same weights.
	prefix := ""
	if opts.ApplyQueryPrefix {
		prefix = descriptor.QueryPrefix
	}

	return &Embedder{
		owner:  true,
		prefix: prefix,
		shared: &shared{
			tokenizer:   tokenizer,
			backend:     backend,
			store:       store,
			exec:        exec,
			descriptor:  descriptor,
			queryPrefix: descriptor.QueryPrefix,
			backendName: backendName,
			padID:       int32(padID), // #nosec G115 -- padID is a tokenizer vocab id, always well under int32 range
			dim:         dim,
			maxSeqLen:   maxSeqLen,
			seqB:        seqB,
			batchB:      batchB,
			sem:         semaphore.NewWeighted(defaultConcurrency),
			compiled:    make(map[[2]int]bool, len(seqB)*len(batchB)),
		},
	}, nil
}

// checkDimensions reconciles the configured width with the model's own.
//
// The model is the source of truth. For a registry model a mismatch means the
// registry is stale; for a raw repository id it means embedding_dimensions is
// wrong, and that value is load-bearing because it decides the store's vector
// column width.
func checkDimensions(m Model, configured, actual int) error {
	if actual <= 0 {
		return fmt.Errorf("%w: %s reports a hidden size of %d", ErrDimensionMismatch, m.DisplayName(), actual)
	}
	if m.Registered {
		if m.Dimensions != 0 && m.Dimensions != actual {
			return fmt.Errorf("%w: registry says %s is %d-dimensional but the model reports %d",
				ErrDimensionMismatch, m.DisplayName(), m.Dimensions, actual)
		}
		return nil
	}
	if configured == 0 {
		return fmt.Errorf("%w: %s is not in the registry, so set embedding_dimensions to %d",
			ErrDimensionMismatch, m.DisplayName(), actual)
	}
	if configured != actual {
		return fmt.Errorf("%w: embedding_dimensions is %d but %s reports %d",
			ErrDimensionMismatch, configured, m.DisplayName(), actual)
	}
	return nil
}

// resolveMaxSeqLen clamps the requested sequence length to what the model's
// position embeddings can represent.
func resolveMaxSeqLen(requested, modelMax int) int {
	if modelMax <= 0 {
		modelMax = 512
	}
	if requested <= 0 || requested > modelMax {
		return modelMax
	}
	return requested
}

// WithPrefix returns a view that prepends prefix to every text it embeds.
//
// The view shares the weights, the compute backend, the graph cache, and the
// concurrency limit, so `codamigo serve` can hold a document embedder and a
// query embedder without loading the model twice.
//
// Only the [Embedder] from [New] owns those resources: Close on a view is a
// no-op, and the owner's Close drains work issued through every view — which is
// precisely why the concurrency limit is shared rather than duplicated. A
// caller that needs only the query side must therefore use
// [Options.ApplyQueryPrefix] instead, or it would hold a view whose Close does
// nothing and never free the backend.
func (e *Embedder) WithPrefix(prefix string) *Embedder {
	return &Embedder{shared: e.shared, prefix: prefix, owner: false}
}

// QueryPrefix returns the model's query instruction prefix, empty for a
// symmetric model. Pass it to [Embedder.WithPrefix] to build the query side.
func (e *Embedder) QueryPrefix() string { return e.shared.queryPrefix }

// Dim returns the embedding dimensionality, read from the loaded model rather
// than from configuration.
func (e *Embedder) Dim() int { return e.shared.dim }

// BackendName returns the compute backend that was actually selected, for
// diagnostics. "go" means the pure-Go fallback, which is roughly 12x slower than
// XLA.
func (e *Embedder) BackendName() string { return e.shared.backendName }

// MaxSeqLen returns the token limit in effect. Longer inputs are truncated.
func (e *Embedder) MaxSeqLen() int { return e.shared.maxSeqLen }

// Embed embeds a single text.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, errs := e.EmbedBatchPartial(ctx, []string{text})
	if errs[0] != nil {
		return nil, errs[0]
	}
	return vectors[0], nil
}

// EmbedBatch embeds every text, failing as a whole if any text fails.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	vectors, errs := e.EmbedBatchPartial(ctx, texts)
	if joined := errors.Join(errs...); joined != nil {
		return nil, joined
	}
	return vectors, nil
}

// EmbedBatchPartial embeds texts with per-text failure isolation, satisfying
// embedder.Embedder's contract: len(vectors) == len(errs) == len(texts), and
// vectors[i] is non-nil exactly when errs[i] is nil. The indexer enforces this,
// so every return path below assigns exactly one of the two for every index.
//
// Texts are tokenized first, then grouped into a closed set of padded shapes,
// then executed. A failure is attributed to exactly the texts in the batch that
// failed, the same isolation the HTTP embedder provides per sub-batch.
func (e *Embedder) EmbedBatchPartial(ctx context.Context, texts []string) ([][]float32, []error) {
	vectors := make([][]float32, len(texts))
	errs := make([]error, len(texts))
	if len(texts) == 0 {
		return vectors, errs
	}

	s := e.shared
	if !s.enter() {
		for i := range errs {
			errs[i] = ErrClosed
		}
		return vectors, errs
	}
	defer s.leave()

	if err := ctx.Err(); err != nil {
		for i := range errs {
			errs[i] = err
		}
		return vectors, errs
	}

	// Validation and tokenization happen before the graph is touched, so the
	// error paths above and below need no model. Whitespace-only text is rejected
	// rather than embedded: the model would happily return the vector of a
	// sequence with no content words, and a silent low-signal vector in the index
	// is worse than a per-chunk error.
	work := make([]int, 0, len(texts))
	tokens := make([][]int32, 0, len(texts))
	lengths := make([]int, 0, len(texts))
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			errs[i] = fmt.Errorf("text %d is empty or whitespace-only", i)
			continue
		}
		row := s.encode(e.prefix + text)
		work = append(work, i)
		tokens = append(tokens, row)
		lengths = append(lengths, len(row))
	}
	if len(work) == 0 {
		return vectors, errs
	}

	batches, err := bucketize(lengths, s.batchB, s.seqB)
	if err != nil {
		for _, i := range work {
			errs[i] = err
		}
		return vectors, errs
	}

	// Each batch writes only the indices it owns, and bucketize guarantees those
	// index sets are disjoint, so there is no shared mutable state and no lock.
	var wg sync.WaitGroup
	for _, b := range batches {
		wg.Go(func() {
			out, err := s.runBatch(ctx, b, tokens)
			for k, p := range b.indices {
				i := work[p]
				if err != nil {
					errs[i] = err
					continue
				}
				vectors[i] = out[k]
			}
		})
	}
	wg.Wait()
	return vectors, errs
}

// encode tokenizes one text under the tokenizer lock, truncating to maxSeqLen.
func (s *shared) encode(text string) []int32 {
	s.tokMu.Lock()
	ids := s.tokenizer.Encode(text)
	s.tokMu.Unlock()

	if len(ids) > s.maxSeqLen {
		ids = ids[:s.maxSeqLen]
	}
	row := make([]int32, len(ids))
	for i, id := range ids {
		row[i] = int32(id) // #nosec G115 -- id is a tokenizer vocab id, always well under int32 range
	}
	return row
}

// ensureCompiled builds the graph for b's shape if it has not been built yet,
// holding compileMu so no two shapes are ever compiled concurrently. Shapes
// already compiled take the read path and do not serialize.
//
// Compilation is lazy by design: eagerly building all 21 shapes costs about 6.2s
// on XLA, which every `search` or `callers` invocation would pay.
func (s *shared) ensureCompiled(b batch) error {
	key := [2]int{b.rows, b.seqLen}

	s.compileMu.RLock()
	done := s.compiled[key]
	s.compileMu.RUnlock()
	if done {
		return nil
	}

	s.compileMu.Lock()
	defer s.compileMu.Unlock()
	if s.compiled[key] {
		return nil
	}
	if _, err := s.exec.Compile(shapes.Make(dtypes.Int32, b.rows, b.seqLen)); err != nil {
		// GoMLX refuses rather than evicting when the cache is full, so an
		// unplanned shape from bucketize surfaces here.
		return fmt.Errorf("%w: compiling batch %d x seq %d: %w", ErrUnexpectedShape, b.rows, b.seqLen, err)
	}
	s.compiled[key] = true
	return nil
}

// runBatch executes one padded batch and returns one vector per real row.
//
// Every tensor it allocates is finalized before it returns: GoMLX buffers are
// freed explicitly, not by the garbage collector, and under XLA they are device
// buffers whose ceiling is VRAM.
func (s *shared) runBatch(ctx context.Context, b batch, tokens [][]int32) (out [][]float32, err error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		// Nothing was acquired, so nothing is released here.
		return nil, fmt.Errorf("waiting for an inference slot: %w", err)
	}
	defer s.sem.Release(1)

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := s.ensureCompiled(b); err != nil {
		return nil, err
	}

	rows := make([][]int32, b.rows)
	for i := range rows {
		row := make([]int32, b.seqLen)
		for j := range row {
			row[j] = s.padID
		}
		// Surplus rows stay all-padding; their results are discarded below.
		if i < len(b.indices) {
			copy(row, tokens[b.indices[i]])
		}
		rows[i] = row
	}

	input := tensors.FromValue(rows)
	defer func() {
		if ferr := input.FinalizeAll(); ferr != nil && err == nil {
			err = fmt.Errorf("releasing input tensor: %w", ferr)
		}
	}()

	results, err := s.exec.Exec(input)
	if err != nil {
		return nil, fmt.Errorf("inference on %s (batch %d x seq %d): %w", s.backendName, b.rows, b.seqLen, err)
	}
	result := results[0]
	defer func() {
		if ferr := result.FinalizeAll(); ferr != nil && err == nil {
			err = fmt.Errorf("releasing output tensor: %w", ferr)
		}
	}()

	// Copy inside the callback: ConstFlatData's slice is only valid for the
	// duration of the call, and there is no CopyFlatData to do it for us.
	out = make([][]float32, len(b.indices))
	var copyErr error
	if cerr := result.ConstFlatData(func(anyFlat any) {
		flat, ok := anyFlat.([]float32)
		if !ok {
			copyErr = fmt.Errorf("expected float32 output, got %T", anyFlat)
			return
		}
		if len(flat) != b.rows*s.dim {
			// Catches a model whose pooled output is not HiddenSize wide, e.g. one
			// with a Dense projection module we did not account for.
			copyErr = fmt.Errorf("%w: got %d floats for %d rows, want %d per row",
				ErrDimensionMismatch, len(flat), b.rows, s.dim)
			return
		}
		for k := range b.indices {
			vec := make([]float32, s.dim)
			copy(vec, flat[k*s.dim:(k+1)*s.dim])
			out[k] = vec
		}
	}); cerr != nil {
		return nil, fmt.Errorf("reading output tensor: %w", cerr)
	}
	if copyErr != nil {
		return nil, copyErr
	}
	return out, nil
}

// enter registers a call as in-flight, reporting false when the Embedder is
// closed. The lock makes the check and the registration one step, so Close
// cannot start finalizing between them.
func (s *shared) enter() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return false
	}
	s.inFlight.Add(1)
	return true
}

func (s *shared) leave() { s.inFlight.Done() }

// Close releases the compute backend, the compiled graphs, and the weights.
//
// Safe to call more than once, and safe to call on a [Embedder.WithPrefix] view,
// where it does nothing. Must not be called from inside an embedding call — it
// waits for in-flight work — which holds because it is only invoked from command
// teardown.
//
// If in-flight work does not finish within the close timeout, Close returns
// [ErrShutdownTimeout] and deliberately skips finalization: freeing buffers
// underneath a live call is undefined behaviour, while leaking them until
// process exit is harmless.
func (e *Embedder) Close() error {
	if !e.owner {
		return nil
	}
	s := e.shared
	s.closeOnce.Do(func() { s.closeErr = s.close() })
	return s.closeErr
}

func (s *shared) close() error {
	// Reject new calls first, so the set of in-flight calls can only shrink.
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()

	drained := make(chan struct{})
	go func() {
		s.inFlight.Wait()
		close(drained)
	}()

	timer := time.NewTimer(defaultCloseTimeout)
	defer timer.Stop()
	select {
	case <-drained:
	case <-timer.C:
		return fmt.Errorf("%w after %s; native resources left for process exit",
			ErrShutdownTimeout, defaultCloseTimeout)
	}

	// Reverse construction order.
	s.exec.Finalize()
	s.store.Finalize()
	s.backend.Finalize()
	return nil
}
