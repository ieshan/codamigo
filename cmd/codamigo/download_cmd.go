package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/gomlx/go-xla/installer"
	"github.com/urfave/cli/v3"

	"github.com/ieshan/codamigo/localembed"
)

func downloadModelCmd() *cli.Command {
	return &cli.Command{
		Name:  "download-model",
		Usage: "download a local embedding model, and optionally the XLA compute plugin",
		Description: "Fetches the model's weights and tokenizer from HuggingFace into\n" +
			"~/.codamigo/models and verifies each file against a pinned checksum.\n" +
			"No HuggingFace token is needed for the built-in models.\n\n" +
			"Known models: " + strings.Join(localembed.RegistryNames(), ", ") + "\n" +
			"Any HuggingFace repository id also works, but is downloaded unverified\n" +
			"and requires embedding_dimensions to be set.",
		Flags: slices.Concat(commonFlags, []cli.Flag{
			&cli.BoolFlag{
				Name:  "xla",
				Usage: "also install the XLA (PJRT) compute plugin, which is 12-25x faster than the pure-Go backend",
			},
			&cli.StringFlag{
				Name:  "cuda",
				Usage: "install the CUDA PJRT plugin for this CUDA major version (\"12\" or \"13\"); linux/amd64 only",
			},
			&cli.BoolFlag{
				Name:  "plugin-only",
				Usage: "install the compute plugin without downloading any model (for container builds)",
			},
			&cli.BoolFlag{
				Name:  "force",
				Usage: "re-download files that are already present and verified",
			},
		}),
		Action: func(ctx context.Context, cmd *cli.Command) error {
			cfg, err := loadConfig(cmd)
			if err != nil {
				return err
			}

			if cmd.Bool("plugin-only") {
				if !cmd.Bool("xla") && cmd.String("cuda") == "" {
					return errors.New("--plugin-only needs --xla or --cuda to say which plugin to install")
				}
				return installPlugins(cmd)
			}

			// The model name comes from the normal config chain, so --model and
			// embedding_model behave the same here as everywhere else. Fall back to
			// the default local model when the configured one is a remote API model
			// that this command cannot download.
			modelName := cfg.EmbeddingModel
			model, err := localembed.Lookup(modelName)
			if err != nil {
				if cfg.EmbeddingProvider != localProvider && !strings.Contains(modelName, "/") {
					// e.g. the default text-embedding-3-small: the user has not
					// switched to the local provider yet, so download the default.
					model, err = localembed.Lookup(localembed.DefaultModel)
				}
				if err != nil {
					return err
				}
			}

			root, err := localModelsRoot(cfg)
			if err != nil {
				return err
			}
			modelDir, err := localembed.ModelDir(root, model)
			if err != nil {
				return err
			}

			if !model.Pinned() {
				fmt.Printf("[WARN] %s is not one of the built-in models, so its files cannot be\n"+
					"       checksum-verified and its revision is not pinned.\n", model.DisplayName())
			}

			fmt.Printf("Downloading %s (%s) into %s\n", model.DisplayName(), model.RepoID, modelDir)
			res, err := localembed.Download(ctx, localembed.DownloadOptions{
				Model:    model,
				ModelDir: modelDir,
				Token:    cfg.EmbeddingHFToken,
				Force:    cmd.Bool("force"),
				Progress: true,
			})
			if err != nil {
				return err
			}

			fmt.Printf("\n%d file(s) downloaded, %d already present, %s total\n",
				len(res.Downloaded), len(res.Skipped), humanBytes(res.Bytes))
			if res.Verified {
				fmt.Printf("[OK]  All files verified against the pinned checksums for revision %s\n", model.Revision)
			}
			fmt.Printf("      Model directory: %s\n", res.ModelDir)

			if stale, err := localembed.SupersededSnapshots(modelDir, model, res.CommitHash); err == nil && len(stale) > 0 {
				fmt.Printf("\n[WARN] %d superseded snapshot(s) remain in this model directory.\n", len(stale))
				fmt.Print("       They are no longer used. Remove them by hand if you want the space:\n")
				for _, s := range stale {
					fmt.Printf("         %s (%s)\n", s.Path, humanBytes(s.Bytes))
				}
			}

			if cmd.Bool("xla") || cmd.String("cuda") != "" {
				if err := installPlugins(cmd); err != nil {
					// Not fatal: the pure-Go backend still works, just slowly.
					fmt.Printf("[WARN] Could not install the compute plugin: %v\n", err)
					fmt.Print("       codamigo will fall back to the pure-Go backend, which is\n" +
						"       roughly 12x slower. Re-run with --xla to try again.\n")
				}
			}

			printLocalConfigSnippet(model, res.Dimensions)
			return nil
		},
	}
}

// installPlugins installs the requested PJRT compute plugins through go-xla's
// exported Go API — no shell pipeline, no curl.
func installPlugins(cmd *cli.Command) error {
	// go-xla downloads GitHub release assets without comparing a checksum
	// (installer/cpu.go still carries a "no hash for github releases" TODO), so
	// say so rather than implying the plugin is verified the way the weights are.
	fmt.Println("\nInstalling the XLA (PJRT) compute plugin.")
	fmt.Print("Note: unlike the model files, the plugin download is not checksum-verified\n" +
		"      upstream. Skip --xla to stay on the pure-Go backend, which needs no plugin.\n")

	if cudaVersion := cmd.String("cuda"); cudaVersion != "" {
		if err := installCUDAPlugin(cudaVersion); err != nil {
			return err
		}
	}
	if cmd.Bool("xla") {
		// installPath "" resolves to the user-local library directory, which is
		// also where GoMLX looks for plugins at runtime. AutoInstall covers CPU
		// always, and CUDA too when it can see an NVIDIA GPU.
		if err := installer.AutoInstall("", true, installer.Normal); err != nil {
			return err
		}
	}
	fmt.Println("[OK]  Plugin installed. Run 'codamigo doctor' to confirm it is picked up.")
	return nil
}

// printLocalConfigSnippet tells the user exactly what to add to switch over,
// including embedding_dimensions, which must match the model or the store will
// refuse to open. dimensions comes from the model's own config.json, so the
// unpinned case no longer leaves the user to work it out.
func printLocalConfigSnippet(model localembed.Model, dimensions int) {
	name := model.DisplayName()
	fmt.Printf("\nTo use it, add this to ~/.codamigo/global_settings.yml:\n\n")
	fmt.Printf("  embedding_provider: local\n")
	fmt.Printf("  embedding_model: %s\n", name)
	switch {
	case model.Dimensions > 0:
		fmt.Printf("  embedding_dimensions: %d\n", model.Dimensions)
	case dimensions > 0:
		fmt.Printf("  embedding_dimensions: %d\n", dimensions)
	default:
		fmt.Printf("  embedding_dimensions: <the model's hidden size — codamigo will tell you>\n")
	}
	fmt.Printf("\nThe index stores its vector width, so switch providers with:\n")
	fmt.Printf("  codamigo reset && codamigo index\n")
}

// humanBytes formats a byte count for the summary line. Kept local rather than
// pulling in a dependency for four lines.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	value := float64(n)
	for _, suffix := range []string{"KiB", "MiB", "GiB", "TiB"} {
		value /= unit
		if value < unit {
			return fmt.Sprintf("%.1f %s", value, suffix)
		}
	}
	return fmt.Sprintf("%.1f PiB", value)
}

// isModelDownloaded reports whether every manifest file for model is present
// under the models root.
func isModelDownloaded(root string, model localembed.Model) (bool, error) {
	dir, err := localembed.ModelDir(root, model)
	if err != nil {
		return false, err
	}
	resolved, _, err := localembed.ResolvePin(dir, model)
	if err != nil {
		// No resolvable revision means nothing usable is on disk.
		return false, nil
	}
	return localembed.IsDownloaded(dir, resolved)
}

// warnIfModelMissing prints an actionable hint when the local provider is
// selected but its model has not been downloaded. Used by init and doctor.
func warnIfModelMissing(root string, model localembed.Model) {
	dir, err := localembed.ModelDir(root, model)
	if err != nil {
		return
	}
	resolved, _, err := localembed.ResolvePin(dir, model)
	if err != nil {
		fmt.Printf("[FAIL] Model %s is not downloaded (%v)\n", model.DisplayName(), err)
		fmt.Printf("       Run: codamigo download-model --model %s\n", model.DisplayName())
		return
	}
	missing, err := localembed.MissingFiles(dir, resolved)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			fmt.Printf("[WARN] Could not inspect %s: %v\n", dir, err)
		}
		return
	}
	if len(missing) == 0 {
		return
	}
	fmt.Printf("[FAIL] Model %s is not downloaded (%d file(s) missing under %s)\n",
		model.DisplayName(), len(missing), dir)
	fmt.Printf("       Run: codamigo download-model --model %s\n", model.DisplayName())
}
