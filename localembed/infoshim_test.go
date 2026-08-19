package localembed

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func testShimPin() (Model, Pin) {
	const sha = "0123456789abcdef0123456789abcdef01234567"
	m := Model{RepoID: "org/model", Revision: sha}
	p := Pin{
		RepoID:     "org/model",
		CommitHash: sha,
		RepoInfo:   json.RawMessage(`{"id":"org/model","sha":"` + sha + `","siblings":[]}`),
	}
	return m, p
}

func TestInfoShim_ServesPinnedRevision(t *testing.T) {
	m, p := testShimPin()
	shim, err := startInfoShim(m, p)
	if err != nil {
		t.Fatalf("startInfoShim: %v", err)
	}
	defer func() { _ = shim.Close() }()

	url := shim.URL + "/api/models/org/model/revision/" + p.CommitHash + "?blobs=true"
	resp, err := http.Get(url) // #nosec G107 -- loopback URL from this test's own shim
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(body) != string(p.RepoInfo) {
		t.Errorf("body = %s, want the pin's RepoInfo verbatim %s", body, p.RepoInfo)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestInfoShim_ServedBytesAreIndependentOfTheCaller(t *testing.T) {
	// The shim exists because LockedDownload unlinks its destination before
	// fetching, so what it serves must not be reachable by anyone else.
	// json.RawMessage is a []byte: `[]byte(p.RepoInfo)` aliases the caller's
	// array rather than copying it, so without a defensive copy a later write
	// through the pin would change what the shim serves mid-flight.
	m, p := testShimPin()
	original := string(p.RepoInfo)
	shim, err := startInfoShim(m, p)
	if err != nil {
		t.Fatalf("startInfoShim: %v", err)
	}
	defer func() { _ = shim.Close() }()

	// Scribble over the caller's copy.
	for i := range p.RepoInfo {
		p.RepoInfo[i] = 'X'
	}

	resp, err := http.Get(shim.URL + "/api/models/org/model/revision/" + p.CommitHash) // #nosec G107 -- loopback URL from this test's own shim
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(body) != original {
		t.Errorf("body = %s, want %s; the shim is aliasing the caller's buffer", body, original)
	}
}

func TestInfoShim_UnknownPathsAre404AndRecorded(t *testing.T) {
	m, p := testShimPin()
	shim, err := startInfoShim(m, p)
	if err != nil {
		t.Fatalf("startInfoShim: %v", err)
	}
	defer func() { _ = shim.Close() }()

	for _, path := range []string{
		"/org/model/resolve/" + p.CommitHash + "/model.safetensors",
		"/api/models/org/model/revision/89abcdef0123456789abcdef0123456789abcdef",
	} {
		resp, err := http.Get(shim.URL + path) // #nosec G107 -- loopback URL from this test's own shim
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s status = %d, want 404", path, resp.StatusCode)
		}
		// The body must name what was asked for, so the surfaced error is
		// actionable rather than a bare HTTP failure.
		if !strings.Contains(string(body), path) {
			t.Errorf("404 body %q does not name the requested path %q", body, path)
		}
	}
	missed := shim.missedPaths()
	if len(missed) != 2 {
		t.Errorf("missedPaths() = %v, want 2 entries", missed)
	}
}

func TestInfoShim_HasReadHeaderTimeout(t *testing.T) {
	// gosec G112: a server without ReadHeaderTimeout is a Slowloris risk.
	m, p := testShimPin()
	shim, err := startInfoShim(m, p)
	if err != nil {
		t.Fatalf("startInfoShim: %v", err)
	}
	defer func() { _ = shim.Close() }()
	if shim.srv.ReadHeaderTimeout <= 0 {
		t.Error("ReadHeaderTimeout is unset")
	}
}
