package localembed_test

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ieshan/codamigo/localembed"
)

func TestModelDir_RegistryModel(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	dir, err := localembed.ModelDir("/models", m)
	if err != nil {
		t.Fatalf("ModelDir: %v", err)
	}
	if want := filepath.Join("/models", "bge-small-en-v1.5"); dir != want {
		t.Errorf("ModelDir = %q, want %q", dir, want)
	}
}

func TestModelDir_RejectsBadRoot(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	for _, root := range []string{"", "relative/path", "./models"} {
		t.Run(root, func(t *testing.T) {
			if _, err := localembed.ModelDir(root, m); err == nil {
				t.Errorf("ModelDir(%q) = nil error, want rejection", root)
			}
		})
	}
}

// TestModelDir_RejectsUnsafeDirName covers the path that actually touches the
// filesystem. Lookup already constrains names, so these are defence in depth
// against a future caller building a Model by hand.
func TestModelDir_RejectsUnsafeDirName(t *testing.T) {
	for _, name := range []string{
		".",
		"..",
		"../escape",
		"a/b",
		"/absolute",
		"nested/../..",
		"with\x00null",
		"with\nnewline",
		"tab\there",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := localembed.ModelDir("/models", localembed.Model{Name: name, RepoID: "org/x"})
			if err == nil {
				t.Errorf("ModelDir with name %q = nil error, want rejection", name)
			}
		})
	}
}

// A Model with neither a name nor a repository id has no directory to resolve.
func TestModelDir_RejectsEmptyModel(t *testing.T) {
	if _, err := localembed.ModelDir("/models", localembed.Model{}); err == nil {
		t.Error("ModelDir on a zero Model = nil error, want rejection")
	}
}

func TestModelDir_EscapeAttemptStaysInsideRoot(t *testing.T) {
	// Belt and braces: even if a name slipped through, the result must not
	// resolve above the root.
	m := localembed.Model{Name: "..", RepoID: "org/x"}
	if dir, err := localembed.ModelDir("/models", m); err == nil {
		if rel, relErr := filepath.Rel("/models", dir); relErr == nil && rel == ".." {
			t.Errorf("ModelDir escaped the root: %q", dir)
		}
	}
}

func TestSnapshotDir_PinnedRevision(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	dir, err := localembed.SnapshotDir("/models/bge-small-en-v1.5", m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	want := filepath.Join("/models/bge-small-en-v1.5",
		"models--BAAI--bge-small-en-v1.5", "snapshots", m.Revision)
	if dir != want {
		t.Errorf("SnapshotDir = %q, want %q", dir, want)
	}
}

func TestSnapshotDir_UsesRevisionDirectly(t *testing.T) {
	root := t.TempDir()
	m := localembed.Model{RepoID: "org/model", Revision: testHash}
	want := filepath.Join(root, "models--org--model", "snapshots", testHash)

	got, err := localembed.SnapshotDir(root, m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	if got != want {
		t.Errorf("SnapshotDir = %q, want %q", got, want)
	}
}

func TestSnapshotDir_DoesNotRequireTheDirectoryToExist(t *testing.T) {
	// SnapshotDir names a path; MissingFiles is what decides whether the files
	// are there. Several revisions side by side is no longer ambiguous, because
	// the pin says which one to use.
	root := t.TempDir()
	repoDir := filepath.Join(root, "models--org--model", "snapshots")
	for _, rev := range []string{testHash, "89abcdef0123456789abcdef0123456789abcdef"} {
		if err := os.MkdirAll(filepath.Join(repoDir, rev), 0o750); err != nil {
			t.Fatalf("MkdirAll: %v", err)
		}
	}
	got, err := localembed.SnapshotDir(root, localembed.Model{RepoID: "org/model", Revision: testHash})
	if err != nil {
		t.Fatalf("SnapshotDir with two revisions present: %v", err)
	}
	if got != filepath.Join(repoDir, testHash) {
		t.Errorf("SnapshotDir = %q, want the pinned revision %q", got, filepath.Join(repoDir, testHash))
	}
}

func TestMissingFiles_NothingDownloaded(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	missing, err := localembed.MissingFiles(t.TempDir(), m)
	if err != nil {
		t.Fatalf("MissingFiles: %v", err)
	}
	if len(missing) != len(m.Files) {
		t.Errorf("MissingFiles reported %d of %d files missing, want all", len(missing), len(m.Files))
	}
	ok, err := localembed.IsDownloaded(t.TempDir(), m)
	if err != nil {
		t.Fatalf("IsDownloaded: %v", err)
	}
	if ok {
		t.Error("IsDownloaded = true for an empty directory")
	}
}

func TestMissingFiles_PartialAndComplete(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	root := t.TempDir()
	snapshot, err := localembed.SnapshotDir(root, m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}

	// Write all but the last manifest entry, at their pinned sizes.
	for _, f := range m.Files[:len(m.Files)-1] {
		writeSized(t, filepath.Join(snapshot, filepath.FromSlash(f.Path)), f.Size)
	}
	last := m.Files[len(m.Files)-1]

	missing, err := localembed.MissingFiles(root, m)
	if err != nil {
		t.Fatalf("MissingFiles: %v", err)
	}
	if !slices.Equal(missing, []string{last.Path}) {
		t.Errorf("MissingFiles = %v, want [%s]", missing, last.Path)
	}

	writeSized(t, filepath.Join(snapshot, filepath.FromSlash(last.Path)), last.Size)
	ok, err := localembed.IsDownloaded(root, m)
	if err != nil {
		t.Fatalf("IsDownloaded: %v", err)
	}
	if !ok {
		t.Error("IsDownloaded = false with the full manifest present")
	}
}

// TestMissingFiles_WrongSizeCountsAsMissing means a truncated download is
// reported as missing rather than loaded and failing deeper in GoMLX.
func TestMissingFiles_WrongSizeCountsAsMissing(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	root := t.TempDir()
	snapshot, err := localembed.SnapshotDir(root, m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	for _, f := range m.Files {
		writeSized(t, filepath.Join(snapshot, filepath.FromSlash(f.Path)), f.Size)
	}
	// Truncate one file.
	truncated := filepath.Join(snapshot, filepath.FromSlash(m.Files[0].Path))
	if err := os.WriteFile(truncated, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	missing, err := localembed.MissingFiles(root, m)
	if err != nil {
		t.Fatalf("MissingFiles: %v", err)
	}
	if !slices.Contains(missing, m.Files[0].Path) {
		t.Errorf("MissingFiles = %v, want it to include the truncated %s", missing, m.Files[0].Path)
	}
}

// TestMissingFiles_DanglingSymlink covers go-huggingface's real layout, where
// snapshot entries are symlinks into blobs/. A dangling link must count as
// missing, which is why the implementation uses Stat rather than Lstat.
func TestMissingFiles_DanglingSymlink(t *testing.T) {
	m, err := localembed.Lookup("bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	root := t.TempDir()
	snapshot, err := localembed.SnapshotDir(root, m)
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}
	for _, f := range m.Files {
		writeSized(t, filepath.Join(snapshot, filepath.FromSlash(f.Path)), f.Size)
	}
	broken := filepath.Join(snapshot, filepath.FromSlash(m.Files[0].Path))
	if err := os.Remove(broken); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if err := os.Symlink(filepath.Join(root, "does-not-exist"), broken); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	missing, err := localembed.MissingFiles(root, m)
	if err != nil {
		t.Fatalf("MissingFiles: %v", err)
	}
	if !slices.Contains(missing, m.Files[0].Path) {
		t.Errorf("MissingFiles = %v, want it to include the dangling %s", missing, m.Files[0].Path)
	}
}

func TestSupersededSnapshots(t *testing.T) {
	const stale = "89abcdef0123456789abcdef0123456789abcdef"
	root := t.TempDir()
	snapshots := filepath.Join(root, "models--org--model", "snapshots")
	for _, rev := range []string{testHash, stale} {
		if err := os.MkdirAll(filepath.Join(snapshots, rev), 0o750); err != nil {
			t.Fatalf("MkdirAll: %v", err)
		}
		if err := os.WriteFile(filepath.Join(snapshots, rev, "w.bin"), []byte("12345"), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
	}
	got, err := localembed.SupersededSnapshots(root, localembed.Model{RepoID: "org/model"}, testHash)
	if err != nil {
		t.Fatalf("SupersededSnapshots: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d superseded snapshots, want 1: %+v", len(got), got)
	}
	if got[0].Path != filepath.Join(snapshots, stale) {
		t.Errorf("Path = %q, want %q", got[0].Path, filepath.Join(snapshots, stale))
	}
	if got[0].Bytes != 5 {
		t.Errorf("Bytes = %d, want 5", got[0].Bytes)
	}
}

func TestSupersededSnapshots_NoneWhenOnlyKeepExists(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "models--org--model", "snapshots", testHash), 0o750); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	got, err := localembed.SupersededSnapshots(root, localembed.Model{RepoID: "org/model"}, testHash)
	if err != nil {
		t.Fatalf("SupersededSnapshots: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %+v, want none", got)
	}
}

func TestSupersededSnapshots_MissingDirIsNotAnError(t *testing.T) {
	got, err := localembed.SupersededSnapshots(t.TempDir(), localembed.Model{RepoID: "org/model"}, testHash)
	if err != nil {
		t.Fatalf("SupersededSnapshots on empty root: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %+v, want none", got)
	}
}

// TestSupersededSnapshots_SharedBlobsNotCounted covers go-huggingface's real
// layout: snapshot entries are symlinks into a shared blobs/ directory,
// deduplicated by etag. Deleting a superseded snapshot only removes its
// symlinks, not blobs still referenced by the kept snapshot, so those bytes
// are not actually reclaimable and must not be counted. Without the fix this
// reports the sum of both blobs (shared + unique) instead of only the unique
// one.
func TestSupersededSnapshots_SharedBlobsNotCounted(t *testing.T) {
	const stale = "89abcdef0123456789abcdef0123456789abcdef"
	root := t.TempDir()
	repoDir := filepath.Join(root, "models--org--model")
	blobs := filepath.Join(repoDir, "blobs")
	if err := os.MkdirAll(blobs, 0o750); err != nil {
		t.Fatalf("MkdirAll blobs: %v", err)
	}
	sharedBlob := filepath.Join(blobs, "shared")
	keepOnlyBlob := filepath.Join(blobs, "keep-only")
	staleOnlyBlob := filepath.Join(blobs, "stale-only")
	for path, content := range map[string]string{
		sharedBlob:    "shared-weights-unchanged",
		keepOnlyBlob:  "new-config",
		staleOnlyBlob: "only-in-stale",
	} {
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatalf("WriteFile %s: %v", path, err)
		}
	}

	snapshots := filepath.Join(repoDir, "snapshots")
	keepDir := filepath.Join(snapshots, testHash)
	staleDir := filepath.Join(snapshots, stale)
	if err := os.MkdirAll(keepDir, 0o750); err != nil {
		t.Fatalf("MkdirAll keep: %v", err)
	}
	if err := os.MkdirAll(staleDir, 0o750); err != nil {
		t.Fatalf("MkdirAll stale: %v", err)
	}

	// The kept snapshot's weights did not change (still points at the shared
	// blob) but its config did (points at a blob the stale snapshot never
	// referenced). This link uses a relative target, matching what
	// go-huggingface actually writes (e.g. "../../blobs/<etag>"), so
	// resolveSymlink's relative-path branch is exercised too, not just the
	// absolute-path one the other three links below use.
	sharedBlobRel, err := filepath.Rel(keepDir, sharedBlob)
	if err != nil {
		t.Fatalf("Rel: %v", err)
	}
	if err := os.Symlink(sharedBlobRel, filepath.Join(keepDir, "model.bin")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Symlink(keepOnlyBlob, filepath.Join(keepDir, "config.json")); err != nil {
		t.Fatalf("Symlink: %v", err)
	}
	// The stale snapshot shares the weights blob but has its own extra file.
	if err := os.Symlink(sharedBlob, filepath.Join(staleDir, "model.bin")); err != nil {
		t.Fatalf("Symlink: %v", err)
	}
	if err := os.Symlink(staleOnlyBlob, filepath.Join(staleDir, "extra.bin")); err != nil {
		t.Fatalf("Symlink: %v", err)
	}

	got, err := localembed.SupersededSnapshots(root, localembed.Model{RepoID: "org/model"}, testHash)
	if err != nil {
		t.Fatalf("SupersededSnapshots: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d superseded snapshots, want 1: %+v", len(got), got)
	}
	want := int64(len("only-in-stale"))
	if got[0].Bytes != want {
		t.Errorf("Bytes = %d, want %d (only the blob not shared with the kept snapshot)", got[0].Bytes, want)
	}
}

// writeSized creates a file of exactly size bytes, so size checks can be
// exercised without materializing real weights.
func writeSized(t *testing.T, path string, size int64) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create %s: %v", path, err)
	}
	defer func() { _ = f.Close() }()
	if size > 0 {
		if err := f.Truncate(size); err != nil {
			t.Fatalf("Truncate %s: %v", path, err)
		}
	}
}
