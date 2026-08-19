package localembed_test

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ieshan/codamigo/localembed"
)

// deadEndpoint returns the URL of a server that has already been shut down, so
// any attempt to reach it fails immediately rather than hanging.
func deadEndpoint(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	return url
}

// TestNew_LoadsWithoutNetwork is the regression test for this whole feature.
//
// HF_ENDPOINT is read by hub.New, so pointing it at a closed port means any
// outbound HuggingFace request fails. New succeeding under that condition
// proves nothing but the local shim was consulted. Before the pin-and-shim
// change this failed with "config.json is missing or failed to download from
// repo" even though every file was already on disk.
//
// This test actually loads the model to completion (unlike its siblings in
// this file, which fail before backend selection), so it runs under both the
// test-race and bench-localembed CI jobs. It uses Backend: "auto" rather than
// "go" for the same reason newTestEmbedder skips "go" under -race: the pure-Go
// backend trips checkptr inside GoMLX's matmul, which aborts the whole test
// binary under -race. It also quiets the backend-selection log the same way
// newTestEmbedder does, since it is the only test in this file that reaches it.
func TestNew_LoadsWithoutNetwork(t *testing.T) {
	const model = "bge-small-en-v1.5"
	root := modelsRoot(t, model) // skips when the model is not downloaded
	t.Setenv("HF_ENDPOINT", deadEndpoint(t))

	prev := slog.Default()
	slog.SetDefault(slog.New(slog.DiscardHandler))
	t.Cleanup(func() { slog.SetDefault(prev) })

	emb, err := localembed.New(localembed.Options{
		Model:      model,
		ModelsRoot: root,
		Backend:    "auto",
	})
	if err != nil {
		t.Fatalf("New with no reachable hub: %v", err)
	}
	defer func() { _ = emb.Close() }()

	// Loading is the point, but a real embedding proves the weights and
	// tokenizer resolved too, not just the config.
	vec, err := emb.Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 384 {
		t.Errorf("Embed returned width %d, want 384 for %s", len(vec), model)
	}
}

func TestNew_UnpinnedWithoutPinIsActionable(t *testing.T) {
	// An unpinned repo id with nothing on disk must say what to run, not
	// produce an opaque HTTP error.
	t.Setenv("HF_ENDPOINT", deadEndpoint(t))
	_, err := localembed.New(localembed.Options{
		Model:      "org/not-downloaded",
		ModelsRoot: t.TempDir(),
		Dimensions: 8,
	})
	if err == nil {
		t.Fatal("New succeeded for a model that is not downloaded")
	}
	if !strings.Contains(err.Error(), "download-model") {
		t.Errorf("error %q does not tell the user how to recover", err)
	}
}

// TestNew_FileOutsideTheSnapshotIsNamed covers the shim-404 translation in New.
//
// standardManifest is deliberately a subset of what some repositories publish,
// so a model can pass MissingFiles and still have LoadStore ask for a file the
// local snapshot never had — a shard of a sharded model, for instance.
// LoadStore reaches that file through repo.IterFileNames, which iterates the
// pin's repo_info siblings, so naming an absent .safetensors sibling reproduces
// the case exactly. Without the translation the user sees a raw HTTP 404
// quoting a 127.0.0.1:<port> loopback URL.
//
// This is the one branch that cannot be reached hermetically. Every file
// LoadModel probes is in standardManifest, so MissingFiles rejects its absence
// before the shim is ever consulted; the only requester of a non-manifest file
// is LoadStore, which runs after GetTokenizer and backend selection and so
// needs a real tokenizer and real weights. Rather than assert against a
// synthetic model that never gets that far, this builds a temp models root that
// symlinks the real snapshot — read-only — and adds the missing sibling there.
func TestNew_FileOutsideTheSnapshotIsNamed(t *testing.T) {
	const (
		model      = "bge-small-en-v1.5"
		repoID     = "BAAI/bge-small-en-v1.5"
		flatRepo   = "models--BAAI--bge-small-en-v1.5"
		absentFile = "model-00002-of-00002.safetensors"
	)
	realRoot := modelsRoot(t, model) // skips when the model is not downloaded
	descriptor, err := localembed.Lookup(model)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	revision := descriptor.Revision

	// A models root of our own, so the pin we are about to write cannot land in
	// the user's real cache. The snapshot is a symlink to the real one: the test
	// only ever reads it, and copying 133 MB per run would be gratuitous.
	root := t.TempDir()
	repoDir := filepath.Join(root, model, flatRepo)
	for _, d := range []string{"info", "blobs", "snapshots"} {
		if err := os.MkdirAll(filepath.Join(repoDir, d), 0o750); err != nil {
			t.Fatalf("MkdirAll %s: %v", d, err)
		}
	}
	realSnapshot := filepath.Join(realRoot, model, flatRepo, "snapshots", revision)
	if err := os.Symlink(realSnapshot, filepath.Join(repoDir, "snapshots", revision)); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	// The siblings list must be faithful for the files the loader resolves
	// through it — GetTokenizer looks tokenizer.json up in the repo index, not on
	// disk — so it carries the real manifest plus the one shard that is absent.
	// That absent shard is what LoadStore reaches for and the shim refuses.
	siblings := make([]string, 0, len(descriptor.Files)+1)
	for _, f := range descriptor.Files {
		siblings = append(siblings, `{"rfilename":"`+f.Path+`"}`)
	}
	siblings = append(siblings, `{"rfilename":"`+absentFile+`"}`)
	repoInfo := `{"id":"` + repoID + `","sha":"` + revision +
		`","siblings":[` + strings.Join(siblings, ",") + `]}`
	if err := localembed.WritePin(filepath.Join(root, model), localembed.Pin{
		RepoID:       repoID,
		CommitHash:   revision,
		ResolvedFrom: revision,
		RepoInfo:     json.RawMessage(repoInfo),
	}); err != nil {
		t.Fatalf("WritePin: %v", err)
	}

	prev := slog.Default()
	slog.SetDefault(slog.New(slog.DiscardHandler))
	t.Cleanup(func() { slog.SetDefault(prev) })

	t.Setenv("HF_ENDPOINT", deadEndpoint(t))
	emb, err := localembed.New(localembed.Options{
		Model:      model,
		ModelsRoot: root,
		Backend:    "auto",
	})
	if err == nil {
		_ = emb.Close()
		t.Fatal("New succeeded even though a sibling is absent from the snapshot")
	}
	if !errors.Is(err, localembed.ErrModelNotDownloaded) {
		t.Errorf("New error = %v, want it to wrap ErrModelNotDownloaded", err)
	}
	if !strings.Contains(err.Error(), absentFile) {
		t.Errorf("error %q does not name the absent file %q", err, absentFile)
	}
	if !strings.Contains(err.Error(), "download-model") {
		t.Errorf("error %q does not tell the user how to recover", err)
	}
	// The whole point of the translation: no loopback URL in the user's face.
	if strings.Contains(err.Error(), "127.0.0.1") {
		t.Errorf("error %q leaks the shim's loopback address", err)
	}
}

// TestNew_DamagedPinRepoInfoIsRejected is hermetic: it never touches
// ~/.codamigo/models.
//
// ReadPin only validates CommitHash, never RepoInfo. Without a guard, a pin
// with a valid hash but empty/mismatched RepoInfo would reach
// go-huggingface's DownloadInfo, whose LockedDownload unlinks info/<hash>
// before discovering the replacement is unusable — destroying the on-disk
// cache the offline path depends on. This asserts New refuses such a pin
// before anything can unlink that file, and that the pre-existing info file
// on disk is untouched afterward.
func TestNew_DamagedPinRepoInfoIsRejected(t *testing.T) {
	const testHash = "0123456789abcdef0123456789abcdef01234567"
	tests := []struct {
		name     string
		repoInfo json.RawMessage
	}{
		{"empty repo_info", nil},
		{"sha mismatch", json.RawMessage(`{"sha":"ffffffffffffffffffffffffffffffffffffffff","siblings":[]}`)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			repoDir := filepath.Join(dir, "models--org--damaged")
			infoDir := filepath.Join(repoDir, "info")
			if err := os.MkdirAll(infoDir, 0o750); err != nil {
				t.Fatalf("MkdirAll: %v", err)
			}

			// A legitimate info file already on disk, as if left over from a
			// previous successful load. If the guard regresses, go-huggingface
			// would unlink and try to rewrite this before failing offline.
			goodInfo := `{"id":"org/damaged","sha":"` + testHash + `","siblings":[]}`
			infoPath := filepath.Join(infoDir, "main")
			if err := os.WriteFile(infoPath, []byte(goodInfo), 0o600); err != nil {
				t.Fatalf("WriteFile info: %v", err)
			}

			snapshot := filepath.Join(repoDir, "snapshots", testHash)
			if err := os.MkdirAll(filepath.Join(snapshot, "1_Pooling"), 0o750); err != nil {
				t.Fatalf("MkdirAll snapshot: %v", err)
			}
			for _, f := range []string{
				"config.json", "config_sentence_transformers.json", "modules.json",
				"tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json", "model.safetensors",
			} {
				if err := os.WriteFile(filepath.Join(snapshot, f), []byte("{}"), 0o600); err != nil {
					t.Fatalf("WriteFile %s: %v", f, err)
				}
			}

			if err := localembed.WritePin(dir, localembed.Pin{
				RepoID:       "org/damaged",
				CommitHash:   testHash,
				ResolvedFrom: "main",
				RepoInfo:     tt.repoInfo,
			}); err != nil {
				t.Fatalf("WritePin: %v", err)
			}

			_, err := localembed.New(localembed.Options{
				Model:      "org/damaged",
				ModelsRoot: dir,
				Dimensions: 8,
			})
			if !errors.Is(err, localembed.ErrModelNotDownloaded) {
				t.Fatalf("New error = %v, want ErrModelNotDownloaded", err)
			}
			if !strings.Contains(err.Error(), "download-model") {
				t.Errorf("error %q does not tell the user how to recover", err)
			}

			got, err := os.ReadFile(infoPath)
			if err != nil {
				t.Fatalf("info file missing after New: %v", err)
			}
			if string(got) != goodInfo {
				t.Errorf("info file changed by a rejected load: got %q, want %q", got, goodInfo)
			}
		})
	}
}
