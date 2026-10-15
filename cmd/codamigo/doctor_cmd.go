package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"slices"

	"github.com/urfave/cli/v3"

	"github.com/ieshan/codamigo/config"
	"github.com/ieshan/codamigo/localembed"
	"github.com/ieshan/codamigo/walker"
	"github.com/ieshan/go-code-chunker/langs"
	"github.com/ieshan/go-embedder"
)

func doctorCmd() *cli.Command {
	return &cli.Command{
		Name:  "doctor",
		Usage: "diagnose config, store, and embedding model health",
		Flags: slices.Concat(commonFlags, []cli.Flag{
			&cli.BoolFlag{
				Name:  "quick",
				Usage: "skip the live embedding smoke-test",
			},
		}),
		Action: func(ctx context.Context, cmd *cli.Command) error {
			quick := cmd.Bool("quick")

			// ── 1. Global config ───────────────────────────────────────────────
			globalPath, globalPathErr := config.GlobalConfigPath()
			var globalErr error
			if globalPathErr != nil {
				fmt.Printf("[FAIL] Cannot resolve global config path: %v\n", globalPathErr)
			} else {
				_, globalErr = config.Load(globalPath)
			}
			switch {
			case errors.Is(globalErr, fs.ErrNotExist):
				fmt.Printf("[FAIL] Global config not found: %s — run 'codamigo init'\n", globalPath)
			case globalErr != nil:
				fmt.Printf("[FAIL] Global config parse error: %v\n", globalErr)
			default:
				fmt.Printf("[OK]  Global config: %s\n", globalPath)
			}

			// ── 2. Project config ──────────────────────────────────────────────
			// Resolve project root for home config path lookup (non-fatal in doctor).
			doctorProjectRoot := ""
			if cmd.IsSet("project-root") {
				doctorProjectRoot = cmd.String("project-root")
			}
			if doctorProjectRoot == "" {
				if wd, err := os.Getwd(); err == nil {
					doctorProjectRoot = wd
				}
			}
			if doctorProjectRoot != "" {
				if homeProjectPath, err := config.HomeProjectConfigPath(doctorProjectRoot); err == nil {
					if _, err := os.Stat(homeProjectPath); err == nil {
						fmt.Printf("[OK]  Home project config: %s\n", homeProjectPath)
					} else {
						fmt.Printf("[--]  Home project config not found (using in-project)\n")
					}
				}
			}
			projectPath := config.ProjectConfigPath()
			_, projectErr := config.Load(projectPath)
			switch {
			case errors.Is(projectErr, fs.ErrNotExist):
				fmt.Printf("[OK]  In-project config not found (using defaults)\n")
			case projectErr != nil:
				fmt.Printf("[FAIL] In-project config parse error: %v\n", projectErr)
			default:
				fmt.Printf("[OK]  In-project config: %s\n", projectPath)
			}

			// Load the merged config for subsequent checks.
			cfg, err := loadConfig(cmd)
			if err != nil {
				return fmt.Errorf("loading config: %w", err)
			}

			// Build the embedder before the store, because the store must be
			// opened with the embedder's dimension, not the configured one — see
			// storeDim. Failing to construct is non-fatal so every later section
			// still reports.
			emb, embErr := newEmbedder(cfg, roleQuery)
			if embErr != nil {
				fmt.Printf("[FAIL] Invalid embedder config: %v\n", embErr)
			} else {
				defer closeEmbedder(emb)
			}

			// ── 3. Provider ────────────────────────────────────────────────────
			reportProvider(cfg, emb)

			// ── 4. Store ───────────────────────────────────────────────────────
			storePath, err := config.DefaultStorePath(cfg.ProjectRoot)
			if err != nil {
				return fmt.Errorf("resolving store path: %w", err)
			}
			storeExists := false
			if _, err := os.Stat(storePath); err == nil {
				fmt.Printf("[OK]  Store: %s\n", storePath)
				storeExists = true
			} else {
				fmt.Printf("[FAIL] Store not found — run 'codamigo index'\n")
			}

			// ── 5. Index stats (only if store exists) ──────────────────────────
			if storeExists {
				s, err := buildStore(storePath, cfg.EmbeddingModel, storeDim(emb, cfg))
				if err != nil {
					fmt.Printf("[FAIL] Store open error: %v\n", err)
				} else {
					defer func() { _ = s.Close() }() // best-effort cleanup; the process is exiting either way
					stats, err := s.Stats(ctx)
					if err != nil {
						fmt.Printf("[FAIL] Stats error: %v\n", err)
					} else {
						fmt.Printf("       Chunks: %6d\n", stats.ChunkCount)
						fmt.Printf("        Files: %6d\n", stats.FileCount)
						if len(stats.Languages) > 0 {
							fmt.Println("    Languages:")
							type langCount struct {
								lang  string
								count int
							}
							langs := make([]langCount, 0, len(stats.Languages))
							for l, c := range stats.Languages {
								langs = append(langs, langCount{l, c})
							}
							slices.SortFunc(langs, func(a, b langCount) int {
								return cmp.Compare(b.count, a.count) // descending
							})
							for _, lc := range langs {
								fmt.Printf("      %12s: %6d\n", lc.lang, lc.count)
							}
						}
					}
				}
			}

			// ── 6. Walker preview ──────────────────────────────────────────────
			// Uses the same filter construction as buildComponents so this count
			// matches what "codamigo index" actually processes. Keep in sync if
			// the language selection in buildComponents ever changes.
			if cfg.ProjectRoot != "" {
				filter := buildExtensionFilter(langs.AllLanguages())
				w, err := walker.New(cfg.ProjectRoot, cfg, walker.WithFileFilter(filter))
				if err != nil {
					fmt.Printf("[FAIL] Walker error: %v\n", err)
				} else {
					defer func() { _ = w.Close() }() // best-effort cleanup; the process is exiting either way
					count := 0
					errCount := 0
					for _, err := range w.Walk(ctx) {
						if err != nil {
							errCount++
							slog.Warn("walker error", slog.Any("error", err))
							continue
						}
						count++
					}
					fmt.Printf("Files matched by walker: %d\n", count)
					if errCount > 0 {
						fmt.Printf("[WARN] Walker encountered %d errors (see above)\n", errCount)
					}
				}
			}

			// ── 7. Embedding smoke-test ────────────────────────────────────────
			// For the local provider this is pure computation: New already loaded
			// the weights, so there is nothing to reach over the network.
			if !quick && embErr == nil {
				vec, err := emb.Embed(ctx, "codamigo doctor test")
				switch {
				case err != nil && cfg.EmbeddingProvider == localProvider:
					fmt.Printf("[FAIL] Local embedding failed: %v\n", err)
				case err != nil:
					fmt.Printf("[FAIL] Embedding model unreachable: %v\n", err)
				case cfg.EmbeddingProvider == localProvider:
					fmt.Printf("[OK]  Local embedding works (model: %s, dims: %d)\n", cfg.EmbeddingModel, len(vec))
				default:
					fmt.Printf("[OK]  Embedding model reachable (model: %s, dims: %d)\n", cfg.EmbeddingModel, len(vec))
				}
			}

			return nil
		},
	}
}

// storeDim returns the embedding dimensionality to validate the store against.
//
// The embedder's own dimension wins: for the local provider the model, not the
// config, is the source of truth, and config.Defaults() sets 1536 — so using the
// configured value made doctor report a spurious "[FAIL] Store open error" for
// any 384-dimensional local model. Falls back to the configured value when the
// embedder could not be constructed, so doctor can still report on the store.
func storeDim(emb embedder.Embedder, cfg *config.Config) int {
	if emb != nil {
		return emb.Dim()
	}
	return cfg.EmbeddingDimensions
}

// reportProvider prints the embedding provider section, including the local
// provider's model directory and resolved compute backend. Each gap prints the
// command that fixes it.
func reportProvider(cfg *config.Config, emb embedder.Embedder) {
	if cfg.EmbeddingProvider != localProvider {
		fmt.Printf("[OK]  Provider: %s (remote API at %s)\n", cfg.EmbeddingProvider, cfg.EmbeddingBaseURL)
		if cfg.EmbeddingAPIKey == "" {
			fmt.Printf("[WARN] No API key set; most remote providers require one\n")
		}
		return
	}

	fmt.Printf("[OK]  Provider: local (in-process, no network)\n")

	model, err := localembed.Lookup(cfg.EmbeddingModel)
	if err != nil {
		fmt.Printf("[FAIL] Model: %v\n", err)
		return
	}
	root, err := localModelsRoot(cfg)
	if err != nil {
		fmt.Printf("[FAIL] Models directory: %v\n", err)
		return
	}
	fmt.Printf("       Model: %s (%s)\n", model.DisplayName(), model.RepoID)
	if !model.Pinned() {
		fmt.Printf("[WARN] %s is not a built-in model, so its files are not checksum-verified\n", model.DisplayName())
	}

	if dir, err := localembed.ModelDir(root, model); err == nil {
		switch pin, err := localembed.ReadPin(dir); {
		case err == nil:
			fmt.Printf("       Revision: %s (pinned %s)\n", pin.CommitHash, pin.ResolvedFrom)
		case errors.Is(err, localembed.ErrNoPin):
			fmt.Printf("       Revision: no pin file; derived from the cached repository info\n")
			fmt.Printf("       Run 'codamigo download-model' to record one.\n")
		default:
			fmt.Printf("[WARN] Revision: %v\n", err)
		}

		resolved, _, err := localembed.ResolvePin(dir, model)
		if err != nil {
			warnIfModelMissing(root, model)
		} else if missing, err := localembed.MissingFiles(dir, resolved); err == nil && len(missing) == 0 {
			fmt.Printf("[OK]  Model files present: %s\n", dir)
		} else {
			warnIfModelMissing(root, model)
		}
	}

	if cfg.EmbeddingHFToken != "" {
		fmt.Printf("       HuggingFace token: set (only needed for gated models)\n")
	}

	local, ok := emb.(*localembed.Embedder)
	if !ok {
		return
	}
	fmt.Printf("       Max sequence length: %d tokens (longer chunks are truncated)\n", local.MaxSeqLen())
	if local.BackendName() == "go" {
		// Worth shouting about: measured at 3.5 embeddings/sec versus 44 on XLA,
		// which is the difference between a usable and an unusable index run.
		fmt.Printf("[WARN] Compute backend: go (pure Go, ~12x slower than XLA)\n")
		fmt.Printf("       Install the faster backend with: codamigo download-model --xla\n")
	} else {
		fmt.Printf("[OK]  Compute backend: %s\n", local.BackendName())
	}
}
