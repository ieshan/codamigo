package localembed

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// Model describes a sentence-transformers model that can run locally.
//
// Registry entries are pinned: a fixed revision plus a manifest whose sizes and
// SHA256 sums were measured against that revision, which is what makes
// [Download] a supply-chain check rather than a corruption check. Entries
// resolved from a bare HuggingFace repository id are unpinned — see [Lookup].
type Model struct {
	// Name is the short name users write in embedding_model. Empty for unpinned
	// models, which are named by their repository id.
	Name string
	// RepoID is the HuggingFace repository, e.g. "BAAI/bge-small-en-v1.5".
	RepoID string
	// Revision is the git revision to download. "main" for unpinned models.
	Revision string
	// Dimensions is the embedding width. Zero for unpinned models, where the
	// caller must supply it via embedding_dimensions. Even when set this is only
	// a cross-check: the authoritative value comes from the loaded model's
	// hidden size.
	Dimensions int
	// QueryPrefix is prepended to queries and never to documents. Empty for
	// symmetric models, which embed both sides identically.
	QueryPrefix string
	// Files is the manifest of repository files needed to load the model.
	// SHA256 and Size are empty/zero for unpinned models.
	Files []ManifestFile
	// Registered reports whether this came from the registry (and is therefore
	// pinned) rather than from a raw repository id.
	Registered bool
}

// ManifestFile is one repository file, with the size and hash pinned for the
// model's revision. Size 0 and an empty SHA256 mean "unpinned, accept whatever
// the server sends" — see [Model.Registered].
type ManifestFile struct {
	Path   string
	Size   int64
	SHA256 string
}

// standardManifest is the sentence-transformers file set. Every model in the
// registry uses it, and it is also what an unpinned repository id is assumed to
// provide. model.safetensors is last so the small metadata files fail fast
// before the multi-megabyte download starts.
var standardManifest = []string{
	"config.json",
	"config_sentence_transformers.json",
	"modules.json",
	"tokenizer.json",
	"tokenizer_config.json",
	"1_Pooling/config.json",
	"model.safetensors",
}

// expandManifest augments base with the extra files modulesJSON declares
// beyond the plain Transformer+Pooling pair, using the sizes and hashes
// repoInfo already reports.
//
// standardManifest — what base is for an unpinned repository id — only covers
// module 0 (the bare Transformer, path "") and module 1 (1_Pooling). A model
// such as google/embeddinggemma-300m adds a Dense projection module ("2_Dense")
// or a Normalize step, each with its own files at "<path>/<name>" inside the
// same repository; without this, Download reports success while
// transformer.LoadModel still needs a file that was never fetched, and
// MissingFiles reports the model ready when it is not (see the
// pinned-model-revision design doc's "Known limitations"). Widening
// standardManifest itself was rejected there: every registry entry is
// hand-verified against exactly the plain set, so this only ever runs for a
// model discovered from a raw repository id, which has no such guarantee to
// begin with.
//
// Extra files are inserted before base's model.safetensors, if present, so a
// small metadata file added here still fails fast rather than waiting behind
// whatever multi-hundred-megabyte download was already in flight — the same
// reasoning that orders standardManifest itself.
func expandManifest(modulesJSON, repoInfo []byte, base []ManifestFile) ([]ManifestFile, error) {
	var modules []struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(modulesJSON, &modules); err != nil {
		return nil, fmt.Errorf("parsing modules.json: %w", err)
	}

	extraModules := make(map[string]bool)
	for _, mod := range modules {
		if mod.Path == "" || mod.Path == "1_Pooling" {
			continue
		}
		extraModules[mod.Path] = true
	}
	if len(extraModules) == 0 {
		return base, nil
	}

	var info struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
			Size      int64  `json:"size"`
			LFS       *struct {
				SHA256 string `json:"sha256"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.Unmarshal(repoInfo, &info); err != nil {
		return nil, fmt.Errorf("parsing repository info: %w", err)
	}

	known := make(map[string]bool, len(base))
	for _, f := range base {
		known[f.Path] = true
	}
	var extraFiles []ManifestFile
	for _, sib := range info.Siblings {
		if known[sib.RFilename] {
			continue
		}
		dir, _, ok := strings.Cut(sib.RFilename, "/")
		if !ok || !extraModules[dir] {
			continue
		}
		f := ManifestFile{Path: sib.RFilename, Size: sib.Size}
		if sib.LFS != nil {
			f.SHA256 = sib.LFS.SHA256
		}
		extraFiles = append(extraFiles, f)
		known[sib.RFilename] = true
	}
	if len(extraFiles) == 0 {
		return base, nil
	}

	insertAt := len(base)
	for i, f := range base {
		if f.Path == "model.safetensors" {
			insertAt = i
			break
		}
	}
	files := make([]ManifestFile, 0, len(base)+len(extraFiles))
	files = append(files, base[:insertAt]...)
	files = append(files, extraFiles...)
	files = append(files, base[insertAt:]...)
	return files, nil
}

// DefaultModel is the model used when embedding_model names no other. It is
// small (133 MB), 384-dimensional, and asymmetric — its query instruction
// prefix maps onto codamigo's existing index/query embedder split.
const DefaultModel = "bge-small-en-v1.5"

// registry holds the models whose revisions and checksums have been verified.
// Adding an entry means downloading it, recording the real sizes and SHA256
// sums for a fixed revision, and confirming it loads and embeds — an entry with
// invented checksums is worse than no entry, because Download would reject a
// legitimate file.
var registry = []Model{
	{
		Name:        "bge-small-en-v1.5",
		RepoID:      "BAAI/bge-small-en-v1.5",
		Revision:    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
		Dimensions:  384,
		QueryPrefix: "Represent this sentence for searching relevant passages: ",
		Files: []ManifestFile{
			{"config.json", 743, "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"},
			{"config_sentence_transformers.json", 124, "940d5f50db195fa6e5e6a4f122c095f77880de259d74b14a65779ed48bdd7c56"},
			{"modules.json", 349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"},
			{"tokenizer.json", 711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"},
			{"tokenizer_config.json", 366, "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3"},
			{"1_Pooling/config.json", 190, "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c"},
			{"model.safetensors", 133466304, "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"},
		},
	},
	{
		// Symmetric: no query prefix, so queries and documents embed identically.
		Name:       "all-MiniLM-L6-v2",
		RepoID:     "sentence-transformers/all-MiniLM-L6-v2",
		Revision:   "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
		Dimensions: 384,
		Files: []ManifestFile{
			{"config.json", 612, "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"},
			{"config_sentence_transformers.json", 116, "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3"},
			{"modules.json", 349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"},
			{"tokenizer.json", 466247, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"},
			{"tokenizer_config.json", 350, "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"},
			{"1_Pooling/config.json", 190, "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23"},
			{"model.safetensors", 90868376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"},
		},
	},
	// nomic-ai/nomic-embed-text-v1.5 is deliberately absent. go-huggingface
	// v0.4.1 cannot load it: transformer.LoadModel fails with "cannot unmarshal
	// number into Go struct field wrapper.rope_parameters of type
	// transformer.RoPEParams". Users can still name it as a repository id, but it
	// will fail at construction, so listing it here would only promise more than
	// the loader delivers.
}

// RegistryNames returns the short names of the pinned models.
func RegistryNames() []string {
	names := make([]string, len(registry))
	for i, m := range registry {
		names[i] = m.Name
	}
	return names
}

// Lookup resolves a model name.
//
// A registry short name returns a pinned entry. A name containing "/" is taken
// as a HuggingFace repository id and returns an unpinned entry: revision
// "main", the standard sentence-transformers manifest, no checksums, and zero
// Dimensions — so the caller must declare the width via embedding_dimensions,
// and [Download] cannot verify what it fetched.
//
// A bare name that is neither returns [ErrUnknownModel] rather than being
// guessed at, so a typo fails locally instead of after a network round trip.
func Lookup(name string) (Model, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Lookup(DefaultModel)
	}
	for _, m := range registry {
		if m.Name == name || m.RepoID == name {
			m.Files = slices.Clone(m.Files)
			m.Registered = true
			return m, nil
		}
	}
	if !strings.Contains(name, "/") {
		return Model{}, fmt.Errorf("%w %q: known models are %s (or give a full HuggingFace repository id such as %q)",
			ErrUnknownModel, name, strings.Join(RegistryNames(), ", "), "BAAI/bge-small-en-v1.5")
	}
	if err := validRepoID(name); err != nil {
		return Model{}, err
	}
	files := make([]ManifestFile, len(standardManifest))
	for i, p := range standardManifest {
		files[i] = ManifestFile{Path: p}
	}
	return Model{
		RepoID:   name,
		Revision: "main",
		Files:    files,
	}, nil
}

// validRepoID rejects repository ids that would produce a surprising cache
// directory. HuggingFace ids are "<owner>/<name>", both restricted to
// alphanumerics, '-', '_' and '.'; anything else is either a typo or an attempt
// to steer the on-disk path.
func validRepoID(id string) error {
	owner, name, ok := strings.Cut(id, "/")
	if !ok || owner == "" || name == "" || strings.Contains(name, "/") {
		return fmt.Errorf("%w %q: expected the form owner/name", ErrUnknownModel, id)
	}
	for _, part := range []string{owner, name} {
		if part == "." || part == ".." {
			return fmt.Errorf("%w %q: %q is not a valid path segment", ErrUnknownModel, id, part)
		}
		for _, r := range part {
			switch {
			case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			case r == '-', r == '_', r == '.':
			default:
				return fmt.Errorf("%w %q: character %q is not allowed in a repository id", ErrUnknownModel, id, r)
			}
		}
	}
	return nil
}

// DisplayName returns the name to show in diagnostics: the short name for a
// registry model, otherwise the repository id.
func (m Model) DisplayName() string {
	if m.Name != "" {
		return m.Name
	}
	return m.RepoID
}

// DirName returns the directory under the models root that holds this model.
// A registry model uses its short name; an unpinned repository id has its "/"
// replaced with "_" so the whole model stays inside one removable directory.
func (m Model) DirName() string {
	if m.Name != "" {
		return m.Name
	}
	return strings.ReplaceAll(m.RepoID, "/", "_")
}

// Pinned reports whether every manifest entry carries a checksum, meaning
// [Download] can verify what it fetched.
func (m Model) Pinned() bool {
	if len(m.Files) == 0 {
		return false
	}
	for _, f := range m.Files {
		if f.SHA256 == "" {
			return false
		}
	}
	return true
}
