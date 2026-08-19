package localembed

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// infoShim answers the one HuggingFace API call go-huggingface insists on
// making, from data already on this machine.
//
// go-huggingface has no offline mode. hub.Repo.readCommitHashForRevision
// computes forceDownload from !revisionHashRefreshed, which is false on every
// freshly constructed Repo, so DownloadInfo re-fetches the repository info over
// the network and ignores the copy cached on disk. Its only product is the name
// of the snapshot directory. There is no way to inject a transport — the
// http.Client is built inline in the library's internal/downloader — so the
// only public lever is WithEndpoint. Pointing it at this shim turns that
// mandatory call into a loopback request that always succeeds and always
// returns the pinned revision.
//
// The shim serves the pin's in-memory RepoInfo rather than reading a file per
// request, because LockedDownload deletes its destination before fetching it:
// the info file on disk is unlinked while this request is in flight.
type infoShim struct {
	URL string

	srv      *http.Server
	listener net.Listener
	done     chan struct{}

	mu     sync.Mutex
	missed []string
}

// startInfoShim binds a loopback listener and starts serving. The caller must
// Close it; it is only needed for the span of [New], since every *hub.Repo
// access happens during loading.
func startInfoShim(m Model, p Pin) (*infoShim, error) {
	if err := validatePinRepoInfo(m, p); err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("starting local model info server for %s: %w", m.DisplayName(), err)
	}
	s := &infoShim{
		URL:      "http://" + listener.Addr().String(),
		listener: listener,
		done:     make(chan struct{}),
	}
	// The path go-huggingface builds is
	// <endpoint>/api/models/<id>/revision/<rev>, optionally with ?blobs=true.
	wantPath := "/api/models/" + m.RepoID + "/revision/" + p.CommitHash
	// Copied, not aliased: json.RawMessage is a []byte, so []byte(p.RepoInfo)
	// would share the caller's array and let a later write change what is
	// served while a request is in flight.
	body := append([]byte(nil), p.RepoInfo...)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != wantPath {
			s.recordMiss(r.URL.Path)
			// Naming the path makes the eventual load failure legible: it is
			// almost always a manifest file that is not in the local snapshot.
			http.Error(w, fmt.Sprintf("%s is not available from the local model cache", r.URL.Path),
				http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	})

	s.srv = &http.Server{
		Handler: mux,
		// gosec G112.
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		defer close(s.done)
		// Discarded deliberately. The one failure worth reporting is a bind
		// failure, and net.Listen above has already returned it; every other
		// Serve error — including the expected ErrServerClosed from Close —
		// surfaces as the load error the caller is about to report anyway.
		_ = s.srv.Serve(listener)
	}()
	return s, nil
}

// validatePinRepoInfo confirms p.RepoInfo actually describes the commit p
// claims, before anything downstream can act on it.
//
// ReadPin only validates that CommitHash looks like a commit hash — it never
// inspects RepoInfo. Without this check, a pin with a valid hash but an
// empty, unparseable, or mismatched RepoInfo would reach go-huggingface's
// DownloadInfo, whose LockedDownload unlinks info/<hash> before finding out
// the replacement is unusable: the model becomes unloadable offline until a
// network re-download, destroying the exact file the offline path depends
// on. Running this before startInfoShim's listener starts guarantees nothing
// downstream gets the chance to unlink that file.
func validatePinRepoInfo(m Model, p Pin) error {
	if len(p.RepoInfo) == 0 {
		return fmt.Errorf("%w: pin for %s has no repo_info. Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, m.DisplayName(), m.DisplayName())
	}
	var meta struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(p.RepoInfo, &meta); err != nil {
		return fmt.Errorf("%w: pin for %s has unparseable repo_info: %v. Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, m.DisplayName(), err, m.DisplayName())
	}
	if meta.SHA != p.CommitHash {
		return fmt.Errorf("%w: pin for %s names commit %s but repo_info sha is %s. Run: codamigo download-model --model %s",
			ErrModelNotDownloaded, m.DisplayName(), p.CommitHash, meta.SHA, m.DisplayName())
	}
	return nil
}

func (s *infoShim) recordMiss(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.missed = append(s.missed, path)
}

// missedPaths returns the paths the shim refused, in request order. A non-empty
// result means something asked for a file the local snapshot does not have.
func (s *infoShim) missedPaths() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.missed...)
}

// Close stops the server and waits for its goroutine to finish. The returned
// error is whatever closing the listener reported; http.Server.Close never
// reports ErrServerClosed, so there is nothing to filter out.
func (s *infoShim) Close() error {
	err := s.srv.Close()
	<-s.done
	return err
}
