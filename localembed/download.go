package localembed

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
)

// DownloadOptions configures [Download].
type DownloadOptions struct {
	// Model is the model to fetch, from [Lookup].
	Model Model
	// ModelDir is the destination directory, from [ModelDir].
	ModelDir string
	// Token authenticates against HuggingFace. Optional: the registry models
	// download anonymously. Only needed for gated or private repositories.
	Token string
	// Endpoint overrides the HuggingFace base URL. Used by tests; empty means
	// the default (honouring HF_ENDPOINT as go-huggingface does).
	Endpoint string
	// Force re-downloads files that are already present and verified.
	Force bool
	// Progress enables go-huggingface's progress output on stdout.
	Progress bool
}

// DownloadResult reports what [Download] did.
type DownloadResult struct {
	// ModelDir is the directory that now holds the model.
	ModelDir string
	// Downloaded and Skipped list manifest paths, in manifest order.
	Downloaded []string
	Skipped    []string
	// Bytes is the total on-disk size of the manifest files.
	Bytes int64
	// Verified reports whether checksums were compared. False for an unpinned
	// repository id, where there is nothing to compare against.
	Verified bool
	// CommitHash is the upstream revision the files came from, resolved from
	// the repository info. Recorded in the pin file so later loads need no
	// network round trip to rediscover it.
	CommitHash string
	// Dimensions is the model's hidden size, read from the downloaded
	// config.json. Zero when it could not be read. Reported so the caller can
	// print a real embedding_dimensions value instead of a placeholder.
	Dimensions int
}

// Download fetches every manifest file for opts.Model into opts.ModelDir and,
// for a pinned model, verifies each file's SHA256 against the value recorded
// for its revision.
//
// Verification is a supply-chain check, not a corruption check: go-huggingface
// caches by ETag and never compares a content hash, and HuggingFace publishes
// real SHA256 sums only for LFS files — not for small ones like tokenizer.json.
// Comparing our own pinned constants against a pinned revision is what makes
// the download reproducible.
//
// Download is idempotent: a file already present with the right size and hash
// is skipped unless opts.Force is set. A file that fails verification is
// removed before the error is returned, so a retry starts clean.
func Download(ctx context.Context, opts DownloadOptions) (*DownloadResult, error) {
	m := opts.Model
	if opts.ModelDir == "" {
		return nil, errors.New("ModelDir must be set")
	}
	if len(m.Files) == 0 {
		return nil, fmt.Errorf("%w: %s has an empty manifest", ErrUnknownModel, m.DisplayName())
	}
	if err := os.MkdirAll(opts.ModelDir, 0o750); err != nil {
		return nil, fmt.Errorf("creating model directory: %w", err)
	}

	repo := hub.New(m.RepoID).WithCacheDir(opts.ModelDir)
	if m.Revision != "" {
		repo = repo.WithRevision(m.Revision)
	}
	if opts.Token != "" {
		repo = repo.WithAuth(opts.Token)
	}
	if opts.Endpoint != "" {
		repo = repo.WithEndpoint(opts.Endpoint)
	}
	repo = repo.WithProgressBar(opts.Progress)
	if opts.Progress {
		repo.Verbosity = 1
	} else {
		repo.Verbosity = 0
	}
	// One file at a time: the manifest is ordered so the small metadata files
	// fail before the multi-megabyte weights start, and sequential progress
	// output is readable.
	repo.MaxParallelDownload = 1

	// DownloadInfo resolves the revision and the file list. It does not take a
	// context upstream, so Ctrl-C during this step is only noticed at the next
	// ctx check below.
	//
	// Forced unconditionally: download-model is the one command that is meant to
	// consult upstream, and DownloadInfo otherwise reuses a cached info file and
	// would never observe that the tracked branch had moved. Loads never reach
	// this code — they answer the same query from the pin file instead.
	if err := repo.DownloadInfo(true); err != nil {
		return nil, downloadError(err, m, opts.Token)
	}

	// Resolve the concrete commit hash once, right after DownloadInfo wrote the
	// info file, and before the per-file loop. SnapshotDir now requires a
	// concrete hash — existingFile (via fetchFile) would otherwise be handed
	// m's unresolved "main" for an unpinned model and silently treat every file
	// as absent, turning an idempotent second run into a full re-download.
	rev := m.Revision
	if rev == "" {
		rev = "main"
	}
	hash, infoBody, err := readDownloadInfo(opts.ModelDir, m, rev)
	if err != nil {
		return nil, err
	}
	opts.Model.Revision = hash

	// An unpinned repository id's manifest may still be missing files that
	// only modules.json itself reveals (a Dense projection, a Normalize
	// step) — discover them now, before the loop below decides the download
	// is complete. Registry models keep their hand-verified manifest as is.
	if !m.Registered {
		expanded, err := expandUnpinnedManifest(ctx, repo, opts, m, infoBody)
		if err != nil {
			return nil, err
		}
		m.Files = expanded
		opts.Model.Files = expanded
	}

	result := &DownloadResult{ModelDir: opts.ModelDir, Verified: m.Pinned()}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		path, downloaded, err := fetchFile(ctx, repo, opts, f)
		if err != nil {
			return nil, err
		}
		if downloaded {
			result.Downloaded = append(result.Downloaded, f.Path)
		} else {
			result.Skipped = append(result.Skipped, f.Path)
		}
		if info, err := os.Stat(path); err == nil {
			result.Bytes += info.Size()
		}
	}

	pin, err := writeDownloadPin(opts, rev, hash, infoBody)
	if err != nil {
		return nil, err
	}
	result.CommitHash = pin.CommitHash
	result.Dimensions = pin.Dimensions

	return result, nil
}

// readDownloadInfo reads and parses the repository info file go-huggingface
// just wrote for rev, returning the resolved commit hash and the raw body as
// read from disk. Reading the file directly rather than calling repo.Info()
// avoids a nil dereference nilaway would reject. The body is what becomes the
// pin's RepoInfo; note that WritePin's json.Marshal compacts and HTML-escapes
// it, so what the shim later replays is semantically equivalent, not
// byte-identical, to what is read here.
func readDownloadInfo(modelDir string, m Model, rev string) (hash string, body []byte, err error) {
	repoDir := filepath.Join(modelDir, flatRepoDir(m.RepoID))
	infoPath := filepath.Join(repoDir, "info", rev)
	// #nosec G304 -- infoPath is inside the model directory this call just populated
	body, err = os.ReadFile(infoPath)
	if err != nil {
		return "", nil, fmt.Errorf("reading repository info for %s at %s: %w", m.DisplayName(), infoPath, err)
	}
	var meta struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(body, &meta); err != nil {
		return "", nil, fmt.Errorf("parsing repository info for %s at %s: %w", m.DisplayName(), infoPath, err)
	}
	if !isCommitHash(meta.SHA) {
		return "", nil, fmt.Errorf("repository info for %s at %s has no valid sha", m.DisplayName(), infoPath)
	}
	return meta.SHA, body, nil
}

// writeDownloadPin records the revision this download resolved to, so loading
// the model later needs no network access. hash and body are the values
// [readDownloadInfo] already parsed from the info file before the per-file
// loop, reused here rather than read a second time.
func writeDownloadPin(opts DownloadOptions, rev, hash string, body []byte) (Pin, error) {
	m := opts.Model
	repoDir := filepath.Join(opts.ModelDir, flatRepoDir(m.RepoID))
	pin := Pin{
		RepoID:       m.RepoID,
		CommitHash:   hash,
		ResolvedFrom: rev,
		ResolvedAt:   time.Now().UTC(),
		Dimensions:   snapshotHiddenSize(repoDir, hash),
		RepoInfo:     body,
	}
	if err := WritePin(opts.ModelDir, pin); err != nil {
		return Pin{}, err
	}
	return pin, nil
}

// snapshotHiddenSize reads hidden_size from the downloaded config.json.
//
// This is the same quantity New cross-checks against embedding_dimensions via
// the loaded model's Config.HiddenSize, so recording it introduces no second
// notion of width. Failure returns 0: the value is only used to print a helpful
// default, and a download must not fail because of it.
func snapshotHiddenSize(repoDir, commitHash string) int {
	// #nosec G304 -- path is inside the model directory this call just populated
	body, err := os.ReadFile(filepath.Join(repoDir, "snapshots", commitHash, "config.json"))
	if err != nil {
		return 0
	}
	var cfg struct {
		HiddenSize int `json:"hidden_size"`
	}
	if err := json.Unmarshal(body, &cfg); err != nil {
		return 0
	}
	return cfg.HiddenSize
}

// expandUnpinnedManifest fetches modules.json ahead of the main download loop
// below and folds in whatever extra module files it declares, via
// [expandManifest]. Fetching it early — rather than waiting for the main loop
// to reach it in manifest order — is what lets the loop's single pass over
// m.Files download everything a Dense projection or Normalize module needs,
// instead of only ever knowing about the plain Transformer+Pooling pair.
//
// The fetch here is not wasted work: fetchFile is idempotent, so the main loop
// downloading modules.json again a moment later just finds it already present.
func expandUnpinnedManifest(ctx context.Context, repo *hub.Repo, opts DownloadOptions, m Model, infoBody []byte) ([]ManifestFile, error) {
	modulesFile, ok := findManifestFile(m.Files, "modules.json")
	if !ok {
		return m.Files, nil
	}
	path, _, err := fetchFile(ctx, repo, opts, modulesFile)
	if err != nil {
		return nil, err
	}
	// #nosec G304 -- path is inside the model directory this call just populated
	modulesJSON, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading modules.json to discover extra model files: %w", err)
	}
	return expandManifest(modulesJSON, infoBody, m.Files)
}

// findManifestFile looks up one manifest entry by path.
func findManifestFile(files []ManifestFile, path string) (ManifestFile, bool) {
	for _, f := range files {
		if f.Path == path {
			return f, true
		}
	}
	return ManifestFile{}, false
}

// fetchFile downloads one manifest file unless an acceptable copy is already
// present, and verifies it before returning. It reports whether a download
// actually happened so the caller can distinguish a no-op run.
func fetchFile(ctx context.Context, repo *hub.Repo, opts DownloadOptions, f ManifestFile) (path string, downloaded bool, err error) {
	if !opts.Force {
		if p, ok := existingFile(opts.ModelDir, opts.Model, f); ok {
			return p, false, nil
		}
	}
	path, err = repo.DownloadFileCtx(ctx, f.Path)
	if err != nil {
		// A cancelled context surfaces as a transport error upstream; report the
		// cause rather than a confusing download failure.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", false, ctxErr
		}
		if isAccessDenied(err) {
			return "", false, accessDeniedError(err, opts.Model, opts.Token)
		}
		return "", false, fmt.Errorf("downloading %s from %s: %w", f.Path, opts.Model.RepoID, err)
	}
	if err := verifyFile(path, f); err != nil {
		return "", false, err
	}
	return path, true, nil
}

// existingFile reports whether f is already on disk and acceptable, so the
// download can be skipped. A wrong size or hash counts as absent: re-fetching
// is cheaper than making the user work out how to clear the cache by hand.
func existingFile(modelDir string, m Model, f ManifestFile) (string, bool) {
	snapshot, err := SnapshotDir(modelDir, m)
	if err != nil {
		return "", false
	}
	path := filepath.Join(snapshot, filepath.FromSlash(f.Path))
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return "", false
	}
	if f.Size > 0 && info.Size() != f.Size {
		return "", false
	}
	if f.SHA256 != "" {
		sum, err := sha256File(path)
		if err != nil || !strings.EqualFold(sum, f.SHA256) {
			return "", false
		}
	}
	return path, true
}

// verifyFile compares path against f's pinned hash and size, removing the file
// and the blob it links to on mismatch so a retry starts clean.
func verifyFile(path string, f ManifestFile) error {
	if f.SHA256 == "" && f.Size == 0 {
		return nil // unpinned: nothing to compare against
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat %s: %w", f.Path, err)
	}
	if f.Size > 0 && info.Size() != f.Size {
		removeDownloaded(path)
		return fmt.Errorf("%w for %s: got %d bytes, want %d", ErrChecksumMismatch, f.Path, info.Size(), f.Size)
	}
	if f.SHA256 == "" {
		return nil
	}
	sum, err := sha256File(path)
	if err != nil {
		return fmt.Errorf("hashing %s: %w", f.Path, err)
	}
	if !strings.EqualFold(sum, f.SHA256) {
		removeDownloaded(path)
		return fmt.Errorf("%w for %s: got %s, want %s", ErrChecksumMismatch, f.Path, sum, f.SHA256)
	}
	return nil
}

// removeDownloaded deletes a snapshot entry and the blob it points at, so a
// rejected file is not silently reused on the next run. Errors are ignored:
// this runs on a failure path whose own error is the one worth reporting.
func removeDownloaded(path string) {
	if target, err := os.Readlink(path); err == nil {
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(path), target)
		}
		_ = os.Remove(target)
	}
	_ = os.Remove(path)
}

func sha256File(path string) (string, error) {
	// #nosec G304 -- path is a file this process just downloaded into its own model cache dir, not external input
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }() // the file is only being read
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// downloadError annotates a repository-level failure.
func downloadError(err error, m Model, token string) error {
	if !isAccessDenied(err) {
		return fmt.Errorf("downloading %s: %w", m.RepoID, err)
	}
	return accessDeniedError(err, m, token)
}

// accessDeniedError adds the token hint only when no token was supplied. Told to
// set a token by a tool that was already given one, a user would go looking in
// the wrong place.
func accessDeniedError(err error, m Model, token string) error {
	if token != "" {
		return fmt.Errorf("%w for %s (the supplied token does not grant access): %w",
			ErrAccessDenied, m.RepoID, err)
	}
	return fmt.Errorf("%w for %s: it appears to be gated or private. "+
		"Set CODAMIGO_HF_TOKEN or HF_TOKEN, or embedding_hf_token in "+
		"~/.codamigo/global_settings.yml, then retry: %w", ErrAccessDenied, m.RepoID, err)
}

// isAccessDenied reports whether err looks like an HTTP 401 or 403.
//
// go-huggingface returns plain formatted strings rather than typed errors
// ("bad status code 403: ...", "failed with the following message: \"403
// Forbidden\""), so substring matching is the only option. It is deliberately
// additive: a false negative just means the caller sees the raw error, and a
// false positive adds a hint that is merely unhelpful.
func isAccessDenied(err error) bool {
	msg := err.Error()
	for _, needle := range []string{"401", "403", "Unauthorized", "Forbidden"} {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}
