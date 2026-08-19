package localembed

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// pinFileName is the pin file's name. It lives directly under ModelDir rather
// than inside the models--<id> tree, because everything below ModelDir belongs
// to go-huggingface (see the layout comment in cache.go).
const pinFileName = "codamigo-pin.json"

// Pin records the exact upstream revision a model directory holds, so loading
// the model needs no network round trip to discover it.
//
// RepoInfo is the RepoInfo JSON as it arrived from HuggingFace, semantically
// preserved (compacted, HTML-escaped by json.Marshal) rather than byte-for-byte,
// because [startInfoShim] serves it straight back to go-huggingface, which must
// be able to unmarshal it into its own hub.RepoInfo. Storing our own copy is not
// redundancy: go-huggingface's LockedDownload deletes its info/<revision> file
// before re-fetching it, so a shim reading that file would be reading something
// that had just been unlinked.
type Pin struct {
	RepoID string `json:"repo_id"`
	// CommitHash is the resolved 40-hex git commit the snapshot directory is
	// named after.
	CommitHash string `json:"commit_hash"`
	// ResolvedFrom is the revision that was asked for, "main" for an unpinned
	// model. Kept for diagnostics: it is what distinguishes "pinned upstream"
	// from "whatever main happened to be".
	ResolvedFrom string `json:"resolved_from"`
	// ResolvedAt is when download-model last consulted upstream.
	ResolvedAt time.Time `json:"resolved_at"`
	// Dimensions is the model's hidden size, recorded so download-model can
	// print a real embedding_dimensions value. Display only: it is never
	// consulted at load time.
	Dimensions int             `json:"dimensions,omitempty"`
	RepoInfo   json.RawMessage `json:"repo_info"`
}

// PinPath returns the pin file's path for a model directory.
func PinPath(modelDir string) string {
	return filepath.Join(modelDir, pinFileName)
}

// WritePin writes the pin file, replacing any existing one.
func WritePin(modelDir string, p Pin) error {
	if modelDir == "" {
		return errors.New("model directory must not be empty")
	}
	body, err := json.Marshal(p)
	if err != nil {
		return fmt.Errorf("encoding pin for %s: %w", p.RepoID, err)
	}
	if err := os.WriteFile(PinPath(modelDir), body, 0o600); err != nil {
		return fmt.Errorf("writing pin for %s: %w", p.RepoID, err)
	}
	return nil
}

// ReadPin reads the pin file.
//
// Absent, unreadable, unparseable and semantically unusable files all return
// [ErrNoPin] rather than distinct errors, because every caller treats them
// identically: fall through to deriving the revision from disk.
func ReadPin(modelDir string) (Pin, error) {
	// #nosec G304 -- modelDir is derived from the configured models root, not external input
	body, err := os.ReadFile(PinPath(modelDir))
	if err != nil {
		return Pin{}, fmt.Errorf("%w: %s: %w", ErrNoPin, PinPath(modelDir), err)
	}
	var p Pin
	if err := json.Unmarshal(body, &p); err != nil {
		return Pin{}, fmt.Errorf("%w: %s is not valid JSON: %w", ErrNoPin, PinPath(modelDir), err)
	}
	if !isCommitHash(p.CommitHash) {
		return Pin{}, fmt.Errorf("%w: %s has no valid commit_hash", ErrNoPin, PinPath(modelDir))
	}
	return p, nil
}

// isCommitHash reports whether s is a full 40-character hex git object id.
// Anything shorter is ambiguous and anything else is not a revision at all.
func isCommitHash(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9', r >= 'a' && r <= 'f':
		default:
			return false
		}
	}
	return true
}

// derivePin reconstructs a pin from what go-huggingface already left on disk,
// for a model directory downloaded before pin files existed. It reads the info
// file for the requested revision, takes its "sha", and confirms a snapshot of
// that name is actually present.
//
// The info file's bytes become the pin's RepoInfo. They are read here, before
// the shim starts and before any load begins, which is what keeps them safe
// from LockedDownload unlinking the file later.
func derivePin(modelDir string, m Model) (Pin, error) {
	rev := m.Revision
	if rev == "" {
		rev = "main"
	}
	repoDir := filepath.Join(modelDir, flatRepoDir(m.RepoID))
	infoPath := filepath.Join(repoDir, "info", rev)
	// #nosec G304 -- infoPath is built from the configured models root and the model's own repo id
	body, err := os.ReadFile(infoPath)
	if err != nil {
		return Pin{}, fmt.Errorf("%w: %s has no pin file and no cached info at %s. "+
			"Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, m.DisplayName(), infoPath, m.DisplayName())
	}
	var meta struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(body, &meta); err != nil {
		return Pin{}, fmt.Errorf("%w: cached info at %s is not valid JSON: %w. "+
			"Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, infoPath, err, m.DisplayName())
	}
	if !isCommitHash(meta.SHA) {
		return Pin{}, fmt.Errorf("%w: cached info at %s has no valid sha. "+
			"Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, infoPath, m.DisplayName())
	}
	snapshot := filepath.Join(repoDir, "snapshots", meta.SHA)
	if info, err := os.Stat(snapshot); err != nil || !info.IsDir() {
		return Pin{}, fmt.Errorf("%w: cached info names revision %s but %s is not present. "+
			"Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, meta.SHA, snapshot, m.DisplayName())
	}
	return Pin{
		RepoID:       m.RepoID,
		CommitHash:   meta.SHA,
		ResolvedFrom: rev,
		RepoInfo:     body,
	}, nil
}

// ResolvePin returns m with Revision set to the concrete commit hash the model
// directory actually holds, together with the pin that decided it.
//
// A registry model keeps its compiled-in Revision: that hash is the
// supply-chain pin, so a pin file must never be able to redirect it. A pin file
// that contradicts it means the directory and the binary disagree, which is
// reported rather than silently resolved in either direction.
func ResolvePin(modelDir string, m Model) (Model, Pin, error) {
	p, err := ReadPin(modelDir)
	if err != nil {
		if p, err = derivePin(modelDir, m); err != nil {
			return Model{}, Pin{}, err
		}
	}
	if m.Registered {
		if p.CommitHash != m.Revision {
			return Model{}, Pin{}, fmt.Errorf(
				"%s is pinned to revision %s but %s holds %s. "+
					"Run: codamigo download-model --model %s --force",
				m.DisplayName(), m.Revision, modelDir, p.CommitHash, m.DisplayName())
		}
		return m, p, nil
	}
	m.Revision = p.CommitHash
	m.Files = expandManifestFromSnapshot(modelDir, m, p)
	return m, p, nil
}

// expandManifestFromSnapshot is the load-time twin of [expandUnpinnedManifest]:
// no network access is available here, so it reads modules.json from the
// snapshot directory Revision now names, instead of fetching it, and reuses
// the repository info the pin already carries.
//
// Any failure to read or parse falls back to m.Files unchanged rather than
// failing ResolvePin outright: an unreadable or absent modules.json is exactly
// what [MissingFiles] already reports on the base manifest, and a malformed
// repoInfo is diagnosed later, when the caller actually needs it, rather than
// here where the only consequence would be an incomplete manifest.
func expandManifestFromSnapshot(modelDir string, m Model, p Pin) []ManifestFile {
	snapshot, err := SnapshotDir(modelDir, m)
	if err != nil {
		return m.Files
	}
	// #nosec G304 -- path is inside the model directory ResolvePin was asked about
	modulesJSON, err := os.ReadFile(filepath.Join(snapshot, "modules.json"))
	if err != nil {
		return m.Files
	}
	expanded, err := expandManifest(modulesJSON, p.RepoInfo, m.Files)
	if err != nil {
		return m.Files
	}
	return expanded
}
