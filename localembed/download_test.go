package localembed_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ieshan/codamigo/localembed"
)

// fakeHub serves the three endpoints go-huggingface actually uses, so download
// behaviour can be exercised without touching huggingface.co:
//
//	GET  /api/models/<id>/revision/<rev>   repository info (siblings + sha)
//	HEAD /<id>/resolve/<sha>/<file>        ETag and size, used to name the blob
//	GET  /<id>/resolve/<sha>/<file>        the bytes
type fakeHub struct {
	*httptest.Server
	// status, when non-zero, is returned for every request instead of content.
	status int
	// requests counts GETs of file content, so an idempotent second run can be
	// distinguished from a re-download.
	requests map[string]int
	// lfsSHA256, keyed by rfilename, is reported as that sibling's lfs.sha256
	// in the repository info, the way HuggingFace does for large files.
	lfsSHA256 map[string]string
}

const fakeRevision = "0123456789abcdef0123456789abcdef01234567"

func newFakeHub(t *testing.T, repoID string, files map[string]string) *fakeHub {
	t.Helper()
	hub := &fakeHub{requests: map[string]int{}}
	mux := http.NewServeMux()

	mux.HandleFunc("/api/models/"+repoID+"/revision/", func(w http.ResponseWriter, r *http.Request) {
		if hub.status != 0 {
			http.Error(w, fmt.Sprintf("%d denied", hub.status), hub.status)
			return
		}
		siblings := make([]map[string]any, 0, len(files))
		for name, content := range files {
			sib := map[string]any{"rfilename": name, "size": len(content)}
			if sum, ok := hub.lfsSHA256[name]; ok {
				sib["lfs"] = map[string]any{"sha256": sum, "size": len(content)}
			}
			siblings = append(siblings, sib)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_id":      "test",
			"id":       repoID,
			"sha":      fakeRevision,
			"siblings": siblings,
		})
	})

	mux.HandleFunc("/"+repoID+"/resolve/", func(w http.ResponseWriter, r *http.Request) {
		if hub.status != 0 {
			http.Error(w, fmt.Sprintf("%d denied", hub.status), hub.status)
			return
		}
		// Path is /<id>/resolve/<sha>/<file...>; everything after the sha is the
		// file name, which may itself contain a slash (1_Pooling/config.json).
		prefix := "/" + repoID + "/resolve/"
		rest := strings.TrimPrefix(r.URL.Path, prefix)
		_, name, ok := strings.Cut(rest, "/")
		if !ok {
			http.NotFound(w, r)
			return
		}
		content, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		// A distinct ETag per file, since go-huggingface names the blob by it.
		sum := sha256.Sum256([]byte(name))
		w.Header().Set("ETag", `"`+hex.EncodeToString(sum[:])+`"`)
		w.Header().Set("Content-Length", fmt.Sprint(len(content)))
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		hub.requests[name]++
		_, _ = w.Write([]byte(content))
	})

	hub.Server = httptest.NewServer(mux)
	t.Cleanup(hub.Close)
	return hub
}

// pinnedModel builds a Model whose manifest matches the served content exactly,
// so a successful download verifies rather than merely completes.
func pinnedModel(repoID string, files map[string]string, order []string) localembed.Model {
	manifest := make([]localembed.ManifestFile, 0, len(order))
	for _, name := range order {
		sum := sha256.Sum256([]byte(files[name]))
		manifest = append(manifest, localembed.ManifestFile{
			Path:   name,
			Size:   int64(len(files[name])),
			SHA256: hex.EncodeToString(sum[:]),
		})
	}
	return localembed.Model{
		Name:       "test-model",
		RepoID:     repoID,
		Revision:   fakeRevision,
		Dimensions: 8,
		Files:      manifest,
		Registered: true,
	}
}

func testFiles() (map[string]string, []string) {
	files := map[string]string{
		"config.json":           `{"hidden_size": 8}`,
		"modules.json":          `[]`,
		"1_Pooling/config.json": `{"pooling_mode_cls_token": true}`,
		"model.safetensors":     strings.Repeat("w", 512),
	}
	return files, []string{"config.json", "modules.json", "1_Pooling/config.json", "model.safetensors"}
}

func TestDownload_Success(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	dir := t.TempDir()

	res, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	})
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if len(res.Downloaded) != len(order) {
		t.Errorf("Downloaded = %v, want all %d files", res.Downloaded, len(order))
	}
	if len(res.Skipped) != 0 {
		t.Errorf("Skipped = %v, want none on a fresh download", res.Skipped)
	}
	if !res.Verified {
		t.Error("Verified = false for a pinned model")
	}
	var wantBytes int64
	for _, f := range m.Files {
		wantBytes += f.Size
	}
	if res.Bytes != wantBytes {
		t.Errorf("Bytes = %d, want %d", res.Bytes, wantBytes)
	}
	ok, err := localembed.IsDownloaded(dir, m)
	if err != nil {
		t.Fatalf("IsDownloaded: %v", err)
	}
	if !ok {
		t.Error("IsDownloaded = false right after a successful download")
	}
	// Nested manifest paths must land in a subdirectory, not a flattened name.
	snapshot, err := localembed.SnapshotDir(dir, m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(snapshot, "1_Pooling", "config.json")); err != nil {
		t.Errorf("nested manifest file not where expected: %v", err)
	}
}

func TestDownload_IdempotentSkip(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	dir := t.TempDir()
	opts := localembed.DownloadOptions{Model: m, ModelDir: dir, Endpoint: hub.URL}

	if _, err := localembed.Download(t.Context(), opts); err != nil {
		t.Fatalf("first Download: %v", err)
	}
	firstCounts := map[string]int{}
	for k, v := range hub.requests {
		firstCounts[k] = v
	}

	res, err := localembed.Download(t.Context(), opts)
	if err != nil {
		t.Fatalf("second Download: %v", err)
	}
	if len(res.Skipped) != len(order) {
		t.Errorf("Skipped = %v, want all %d files on a repeat run", res.Skipped, len(order))
	}
	if len(res.Downloaded) != 0 {
		t.Errorf("Downloaded = %v, want none on a repeat run", res.Downloaded)
	}
	for name, count := range hub.requests {
		if count != firstCounts[name] {
			t.Errorf("%s was re-fetched (%d then %d requests)", name, firstCounts[name], count)
		}
	}
}

// TestDownload_IdempotentSkipUnpinned is the unpinned twin of
// TestDownload_IdempotentSkip: a repository id tracking "main" has
// Model.Revision == "main" going into Download, which SnapshotDir now rejects
// outright. A second run must still resolve the concrete commit hash before
// checking what is already on disk, or every file gets reported as
// downloaded again instead of skipped.
func TestDownload_IdempotentSkipUnpinned(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/unpinned-model"
	hub := newFakeHub(t, repoID, files)
	manifest := make([]localembed.ManifestFile, 0, len(order))
	for _, name := range order {
		manifest = append(manifest, localembed.ManifestFile{Path: name})
	}
	m := localembed.Model{RepoID: repoID, Revision: "main", Files: manifest}
	dir := t.TempDir()
	opts := localembed.DownloadOptions{Model: m, ModelDir: dir, Endpoint: hub.URL}

	if _, err := localembed.Download(t.Context(), opts); err != nil {
		t.Fatalf("first Download: %v", err)
	}
	firstCounts := map[string]int{}
	for k, v := range hub.requests {
		firstCounts[k] = v
	}

	res, err := localembed.Download(t.Context(), opts)
	if err != nil {
		t.Fatalf("second Download: %v", err)
	}
	if len(res.Skipped) != len(order) {
		t.Errorf("Skipped = %v, want all %d files on a repeat run", res.Skipped, len(order))
	}
	if len(res.Downloaded) != 0 {
		t.Errorf("Downloaded = %v, want none on a repeat run", res.Downloaded)
	}
	for name, count := range hub.requests {
		if count != firstCounts[name] {
			t.Errorf("%s was re-fetched (%d then %d requests)", name, firstCounts[name], count)
		}
	}
}

func TestDownload_ForceRefetches(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	dir := t.TempDir()

	if _, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	}); err != nil {
		t.Fatalf("first Download: %v", err)
	}
	res, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL, Force: true,
	})
	if err != nil {
		t.Fatalf("forced Download: %v", err)
	}
	if len(res.Downloaded) != len(order) {
		t.Errorf("Downloaded = %v, want all %d files with Force", res.Downloaded, len(order))
	}
}

// TestDownload_ChecksumMismatchLeavesNothingBehind is the important half of
// verification: a rejected file must not be silently reused next run.
func TestDownload_ChecksumMismatchLeavesNothingBehind(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	// Pin a hash the server will never serve.
	m.Files[0].SHA256 = strings.Repeat("a", 64)
	dir := t.TempDir()

	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	})
	if !errors.Is(err, localembed.ErrChecksumMismatch) {
		t.Fatalf("Download = %v, want ErrChecksumMismatch", err)
	}
	if !strings.Contains(err.Error(), m.Files[0].Path) {
		t.Errorf("error should name the offending file, got: %v", err)
	}

	// Neither the snapshot entry nor its blob may survive, and no partial file
	// may be left in the tree.
	snapshot, serr := localembed.SnapshotDir(dir, m)
	if serr == nil {
		if _, err := os.Stat(filepath.Join(snapshot, filepath.FromSlash(m.Files[0].Path))); !os.IsNotExist(err) {
			t.Errorf("rejected file still present: %v", err)
		}
	}
	var leftovers []string
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil //nolint:nilerr // a walk error just means nothing to inspect here
		}
		if strings.Contains(d.Name(), ".part") || strings.Contains(d.Name(), ".tmp") {
			leftovers = append(leftovers, path)
		}
		return nil
	})
	if len(leftovers) > 0 {
		t.Errorf("partial download files left behind: %v", leftovers)
	}
}

func TestDownload_WrongSizeIsRejected(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	m.Files[0].Size = m.Files[0].Size + 100

	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL,
	})
	if !errors.Is(err, localembed.ErrChecksumMismatch) {
		t.Fatalf("Download = %v, want ErrChecksumMismatch for a size mismatch", err)
	}
}

// TestDownload_AccessDeniedWithoutToken asserts the hint fires when a missing
// token is actually the likely cause.
func TestDownload_AccessDeniedWithoutToken(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	hub.status = http.StatusForbidden
	m := pinnedModel(repoID, files, order)

	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL,
	})
	if !errors.Is(err, localembed.ErrAccessDenied) {
		t.Fatalf("Download = %v, want ErrAccessDenied", err)
	}
	for _, want := range []string{"CODAMIGO_HF_TOKEN", "HF_TOKEN", "embedding_hf_token", "gated"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q, got: %v", want, err)
		}
	}
}

// TestDownload_AccessDeniedWithToken asserts the hint does NOT fire when a token
// was supplied: telling someone to set a token they already set sends them
// looking in the wrong place.
func TestDownload_AccessDeniedWithToken(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	hub.status = http.StatusForbidden
	m := pinnedModel(repoID, files, order)

	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL, Token: "hf_test",
	})
	if !errors.Is(err, localembed.ErrAccessDenied) {
		t.Fatalf("Download = %v, want ErrAccessDenied", err)
	}
	if strings.Contains(err.Error(), "CODAMIGO_HF_TOKEN") {
		t.Errorf("error should not suggest setting a token when one was supplied, got: %v", err)
	}
	if !strings.Contains(err.Error(), "does not grant access") {
		t.Errorf("error should say the supplied token was rejected, got: %v", err)
	}
}

func TestDownload_Unauthorized(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	hub.status = http.StatusUnauthorized
	m := pinnedModel(repoID, files, order)

	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL,
	})
	if !errors.Is(err, localembed.ErrAccessDenied) {
		t.Fatalf("Download = %v, want ErrAccessDenied for 401", err)
	}
}

func TestDownload_CancelledContext(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := localembed.Download(ctx, localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Download with a cancelled context = %v, want context.Canceled", err)
	}
}

func TestDownload_RequiresModelDir(t *testing.T) {
	files, order := testFiles()
	m := pinnedModel("test-org/test-model", files, order)
	if _, err := localembed.Download(t.Context(), localembed.DownloadOptions{Model: m}); err == nil {
		t.Error("Download without ModelDir = nil error, want error")
	}
}

func TestDownload_EmptyManifest(t *testing.T) {
	_, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model:    localembed.Model{RepoID: "org/x", Revision: "main"},
		ModelDir: t.TempDir(),
	})
	if !errors.Is(err, localembed.ErrUnknownModel) {
		t.Errorf("Download with an empty manifest = %v, want ErrUnknownModel", err)
	}
}

// TestDownload_UnpinnedIsNotVerified makes the trade-off explicit: a raw
// repository id can be downloaded, but nothing checks what arrived.
func TestDownload_UnpinnedIsNotVerified(t *testing.T) {
	files := map[string]string{"config.json": `{"hidden_size": 8}`}
	repoID := "test-org/unpinned"
	hub := newFakeHub(t, repoID, files)
	m := localembed.Model{
		RepoID:   repoID,
		Revision: "main",
		Files:    []localembed.ManifestFile{{Path: "config.json"}},
	}
	res, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: t.TempDir(), Endpoint: hub.URL,
	})
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if res.Verified {
		t.Error("Verified = true for an unpinned model")
	}
}

func TestDownload_WritesPin(t *testing.T) {
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	m := pinnedModel(repoID, files, order)
	dir := t.TempDir()

	res, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	})
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if res.CommitHash != fakeRevision {
		t.Errorf("CommitHash = %q, want %q", res.CommitHash, fakeRevision)
	}
	// testFiles() serves config.json as {"hidden_size": 8}.
	if res.Dimensions != 8 {
		t.Errorf("Dimensions = %d, want 8", res.Dimensions)
	}

	pin, err := localembed.ReadPin(dir)
	if err != nil {
		t.Fatalf("ReadPin after Download: %v", err)
	}
	if pin.CommitHash != fakeRevision {
		t.Errorf("pin.CommitHash = %q, want %q", pin.CommitHash, fakeRevision)
	}
	if pin.RepoID != repoID {
		t.Errorf("pin.RepoID = %q, want %q", pin.RepoID, repoID)
	}
	if pin.Dimensions != 8 {
		t.Errorf("pin.Dimensions = %d, want 8", pin.Dimensions)
	}
	if pin.ResolvedAt.IsZero() {
		t.Error("pin.ResolvedAt is zero")
	}
	// The shim replays RepoInfo to go-huggingface, so it must be parseable as
	// the repository info and carry the same sha.
	var meta struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(pin.RepoInfo, &meta); err != nil {
		t.Fatalf("pin.RepoInfo is not valid JSON: %v", err)
	}
	if meta.SHA != fakeRevision {
		t.Errorf("pin.RepoInfo sha = %q, want %q", meta.SHA, fakeRevision)
	}
}

// TestDownload_DiscoversExtraModuleFiles is the regression test for
// google/embeddinggemma-300m: an unpinned repository id whose modules.json
// declares a Dense projection module must have that module's files
// downloaded too, not just the plain Transformer+Pooling pair standardManifest
// assumes.
func TestDownload_DiscoversExtraModuleFiles(t *testing.T) {
	files := map[string]string{
		"config.json":                       `{"hidden_size": 8}`,
		"config_sentence_transformers.json": `{}`,
		"modules.json": `[
			{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
			{"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
			{"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}
		]`,
		"tokenizer.json":            `{}`,
		"tokenizer_config.json":     `{}`,
		"1_Pooling/config.json":     `{"pooling_mode_cls_token": true}`,
		"model.safetensors":         strings.Repeat("w", 512),
		"2_Dense/config.json":       `{"in_features": 8, "out_features": 8}`,
		"2_Dense/model.safetensors": strings.Repeat("d", 64),
	}
	repoID := "test-org/dense-model"
	hub := newFakeHub(t, repoID, files)
	hub.lfsSHA256 = map[string]string{
		"2_Dense/model.safetensors": func() string {
			sum := sha256.Sum256([]byte(files["2_Dense/model.safetensors"]))
			return hex.EncodeToString(sum[:])
		}(),
	}

	base := make([]localembed.ManifestFile, 0)
	for _, p := range []string{
		"config.json", "config_sentence_transformers.json", "modules.json",
		"tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json", "model.safetensors",
	} {
		base = append(base, localembed.ManifestFile{Path: p})
	}
	m := localembed.Model{RepoID: repoID, Revision: "main", Files: base}
	dir := t.TempDir()

	res, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	})
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	wantDownloaded := []string{"2_Dense/config.json", "2_Dense/model.safetensors"}
	for _, want := range wantDownloaded {
		if !slices.Contains(res.Downloaded, want) {
			t.Errorf("Downloaded = %v, want it to include %q", res.Downloaded, want)
		}
	}

	snapshot, err := localembed.SnapshotDir(dir, localembed.Model{RepoID: repoID, Revision: fakeRevision})
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	for _, name := range []string{"2_Dense/config.json", "2_Dense/model.safetensors"} {
		if _, err := os.Stat(filepath.Join(snapshot, filepath.FromSlash(name))); err != nil {
			t.Errorf("%s not on disk after Download: %v", name, err)
		}
	}

	// The model is now actually complete: MissingFiles on the same discovery
	// must agree, which is what New relies on via ResolvePin at load time.
	resolved, _, err := localembed.ResolvePin(dir, m)
	if err != nil {
		t.Fatalf("ResolvePin: %v", err)
	}
	missing, err := localembed.MissingFiles(dir, resolved)
	if err != nil {
		t.Fatalf("MissingFiles: %v", err)
	}
	if len(missing) != 0 {
		t.Errorf("MissingFiles = %v, want none after Download discovered the Dense module", missing)
	}
}

func TestDownload_WritesPinForUnpinnedRepo(t *testing.T) {
	// The case that motivated all of this: a bare repository id tracking "main".
	files, order := testFiles()
	repoID := "test-org/test-model"
	hub := newFakeHub(t, repoID, files)
	manifest := make([]localembed.ManifestFile, 0, len(order))
	for _, name := range order {
		manifest = append(manifest, localembed.ManifestFile{Path: name})
	}
	m := localembed.Model{RepoID: repoID, Revision: "main", Files: manifest}
	dir := t.TempDir()

	if _, err := localembed.Download(t.Context(), localembed.DownloadOptions{
		Model: m, ModelDir: dir, Endpoint: hub.URL,
	}); err != nil {
		t.Fatalf("Download: %v", err)
	}
	pin, err := localembed.ReadPin(dir)
	if err != nil {
		t.Fatalf("ReadPin: %v", err)
	}
	if pin.CommitHash != fakeRevision {
		t.Errorf("pin.CommitHash = %q, want %q", pin.CommitHash, fakeRevision)
	}
	if pin.ResolvedFrom != "main" {
		t.Errorf("pin.ResolvedFrom = %q, want \"main\"", pin.ResolvedFrom)
	}
	// And the resolved pin must now hand back a concrete revision.
	resolved, _, err := localembed.ResolvePin(dir, m)
	if err != nil {
		t.Fatalf("ResolvePin: %v", err)
	}
	if resolved.Revision != fakeRevision {
		t.Errorf("resolved Revision = %q, want %q", resolved.Revision, fakeRevision)
	}
}
