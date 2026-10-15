package localembed

import (
	"strings"
	"testing"
)

// baseUnpinnedFiles mirrors what Lookup builds from standardManifest for a
// raw repository id.
func baseUnpinnedFiles() []ManifestFile {
	files := make([]ManifestFile, len(standardManifest))
	for i, p := range standardManifest {
		files[i] = ManifestFile{Path: p}
	}
	return files
}

// TestExpandManifest_DiscoversDenseModule is the regression test for
// google/embeddinggemma-300m: modules.json declares a Dense projection beyond
// the plain Transformer+Pooling pair, and its files must be added even though
// standardManifest has never heard of them.
func TestExpandManifest_DiscoversDenseModule(t *testing.T) {
	modulesJSON := `[
		{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
		{"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
		{"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
		{"idx": 3, "name": "3", "path": "4_Normalize", "type": "sentence_transformers.models.Normalize"}
	]`
	repoInfo := `{"siblings": [
		{"rfilename": "config.json", "size": 100},
		{"rfilename": "2_Dense/config.json", "size": 134},
		{"rfilename": "2_Dense/model.safetensors", "size": 9437272,
		 "lfs": {"sha256": "c327f2acb00149676ade24a75e11eb6ebbd367f9ee050267ba56829d2979f702"}},
		{"rfilename": "model.safetensors", "size": 1211486072}
	]}`

	got, err := expandManifest([]byte(modulesJSON), []byte(repoInfo), baseUnpinnedFiles())
	if err != nil {
		t.Fatalf("expandManifest: %v", err)
	}

	byPath := make(map[string]ManifestFile, len(got))
	for _, f := range got {
		byPath[f.Path] = f
	}
	for _, want := range []string{"2_Dense/config.json", "2_Dense/model.safetensors"} {
		if _, ok := byPath[want]; !ok {
			t.Errorf("expanded manifest is missing %q; got %v", want, got)
		}
	}
	dense := byPath["2_Dense/model.safetensors"]
	if dense.Size != 9437272 {
		t.Errorf("2_Dense/model.safetensors Size = %d, want 9437272", dense.Size)
	}
	if dense.SHA256 != "c327f2acb00149676ade24a75e11eb6ebbd367f9ee050267ba56829d2979f702" {
		t.Errorf("2_Dense/model.safetensors SHA256 = %q, want the sibling's lfs.sha256", dense.SHA256)
	}
	// 4_Normalize has no siblings of its own in this fixture; it must not
	// invent a file that was never listed.
	if _, ok := byPath["4_Normalize/config.json"]; ok {
		t.Error("expanded manifest invented a file for a module with no matching sibling")
	}

	// The new files must land before model.safetensors, not after it.
	denseIdx, weightsIdx := -1, -1
	for i, f := range got {
		switch f.Path {
		case "2_Dense/model.safetensors":
			denseIdx = i
		case "model.safetensors":
			weightsIdx = i
		}
	}
	if denseIdx == -1 || weightsIdx == -1 || denseIdx >= weightsIdx {
		t.Errorf("2_Dense/model.safetensors (idx %d) must come before model.safetensors (idx %d)", denseIdx, weightsIdx)
	}
}

// TestExpandManifest_NoExtraModulesIsNoOp confirms a plain sentence-transformers
// model — the common case — gets back exactly the manifest it was given.
func TestExpandManifest_NoExtraModulesIsNoOp(t *testing.T) {
	modulesJSON := `[
		{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
		{"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}
	]`
	base := baseUnpinnedFiles()
	got, err := expandManifest([]byte(modulesJSON), []byte(`{"siblings":[]}`), base)
	if err != nil {
		t.Fatalf("expandManifest: %v", err)
	}
	if len(got) != len(base) {
		t.Errorf("expandManifest added files for a model with no extra modules: got %v", got)
	}
}

// TestExpandManifest_MalformedModulesJSON asserts a parse failure is reported
// rather than silently ignored, so a corrupt local file cannot masquerade as
// "no extra modules".
func TestExpandManifest_MalformedModulesJSON(t *testing.T) {
	if _, err := expandManifest([]byte("not json"), []byte(`{}`), baseUnpinnedFiles()); err == nil {
		t.Error("expandManifest with malformed modules.json = nil error, want error")
	}
}

// TestRegistry_EveryEntryIsPinned is the invariant that makes Download a
// supply-chain check rather than a corruption check: an entry with a missing or
// invented checksum would either skip verification silently or reject a
// legitimate file.
//
// It reads the registry directly rather than through an exported accessor —
// this asserts a property of the package's own data, and exporting a getter
// only tests would call would be shipping API for a test's benefit.
func TestRegistry_EveryEntryIsPinned(t *testing.T) {
	if len(registry) == 0 {
		t.Fatal("registry is empty")
	}
	for _, m := range registry {
		t.Run(m.Name, func(t *testing.T) {
			if m.Name == "" {
				t.Error("registry entry has no short name")
			}
			if !strings.Contains(m.RepoID, "/") {
				t.Errorf("RepoID %q is not an owner/name repository id", m.RepoID)
			}
			if len(m.Revision) != 40 {
				t.Errorf("Revision %q is not a 40-character commit hash; a moving "+
					"revision cannot be checksum-pinned", m.Revision)
			}
			if m.Dimensions <= 0 {
				t.Errorf("Dimensions = %d, want positive", m.Dimensions)
			}
			if !m.Pinned() {
				t.Error("entry is not fully pinned")
			}
			for _, f := range m.Files {
				if len(f.SHA256) != 64 {
					t.Errorf("%s: SHA256 %q is not 64 hex characters", f.Path, f.SHA256)
				}
				if f.Size <= 0 {
					t.Errorf("%s: Size = %d, want positive", f.Path, f.Size)
				}
			}
		})
	}
}
