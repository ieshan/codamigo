package localembed

import "errors"

var (
	// ErrModelNotDownloaded reports that the model directory is missing files
	// from the manifest. Recover by running `codamigo download-model`.
	ErrModelNotDownloaded = errors.New("model not downloaded")

	// ErrChecksumMismatch reports that a downloaded file's SHA256 does not match
	// the pinned value for the model's revision. The file is removed before the
	// error is returned, so a retry starts clean.
	ErrChecksumMismatch = errors.New("checksum mismatch")

	// ErrAccessDenied reports that HuggingFace refused the download with 401 or
	// 403, which usually means the repository is gated or private.
	ErrAccessDenied = errors.New("access denied")

	// ErrUnsupportedBackend reports that no compute backend could be
	// constructed, either because the requested one is unavailable or because
	// every candidate in the "auto" chain failed.
	ErrUnsupportedBackend = errors.New("unsupported compute backend")

	// ErrDimensionMismatch reports that the model's true embedding
	// dimensionality differs from the configured EmbeddingDimensions. Only
	// checked for repositories that are not in the registry, where the config is
	// the only declaration available.
	ErrDimensionMismatch = errors.New("embedding dimension mismatch")

	// ErrClosed reports use of an Embedder after Close.
	ErrClosed = errors.New("embedder is closed")

	// ErrShutdownTimeout reports that Close gave up waiting for in-flight calls.
	// Native resources are deliberately left unfinalized in this case: freeing
	// buffers underneath a live call is undefined behaviour, whereas leaking
	// them until process exit is harmless.
	ErrShutdownTimeout = errors.New("timed out waiting for in-flight embeddings")

	// ErrUnexpectedShape reports that bucketize produced a shape outside the
	// closed set the graph cache is sized for. This is an internal invariant
	// violation rather than a user error.
	ErrUnexpectedShape = errors.New("unexpected batch shape")

	// ErrUnknownModel reports that a model short name is not in the registry and
	// is not a usable HuggingFace repository id.
	ErrUnknownModel = errors.New("unknown model")

	// ErrNoPin reports that a model directory has no usable pin file. It is an
	// internal signal, not a user-facing failure: callers fall through to
	// deriving the revision from go-huggingface's cached info file.
	ErrNoPin = errors.New("no usable pin file")
)
