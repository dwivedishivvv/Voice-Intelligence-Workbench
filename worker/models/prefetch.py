#!/usr/bin/env python3
"""Download all models listed in REGISTRY.yaml to their local_dir paths and the HF hub cache.

Idempotent: skips any model whose local_dir is already populated.
Run once before the worker starts; models persist in the Docker volume across restarts.

Gated models (pyannote diarization) require a Hugging Face token:
  1. Accept the model's terms at huggingface.co
  2. Set HF_TOKEN=<your token> in .env
"""
import os
import sys
from pathlib import Path

import yaml


def _is_ready(p: str) -> bool:
    d = Path(p)
    return d.exists() and any(d.iterdir())


def main():
    reg_path = Path(__file__).parent / "REGISTRY.yaml"
    reg = yaml.safe_load(reg_path.read_text())

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    failed: list[str] = []

    for name, spec in reg["models"].items():
        local_dir = spec["local_dir"]
        repo_id = spec["id"]
        revision = spec.get("revision", "main")
        gated = spec.get("gated", False)
        tok = token if (gated or token) else None

        # --- primary download into local_dir ---
        if _is_ready(local_dir):
            print(f"  [ok]    {name:20s} {local_dir}", flush=True)
        else:
            if gated and not token:
                print(
                    f"  [skip]  {name} — gated model, HF_TOKEN not set.\n"
                    f"          Accept terms at huggingface.co/{repo_id} then set HF_TOKEN in .env",
                    file=sys.stderr,
                )
                failed.append(name)
                continue
            print(f"  [fetch] {name:20s} {repo_id} -> {local_dir}", flush=True)
            try:
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, token=tok)
            except Exception as exc:
                print(f"  [error] {name}: {exc}", file=sys.stderr)
                failed.append(name)
                continue

        # --- also populate HF hub cache for sub-model resolution ---
        # pyannote loads segmentation/wespeaker from HF_HOME/hub at runtime;
        # speechbrain resolves its pretrained_path the same way.
        for dep in [repo_id] + spec.get("depends_on", []):
            try:
                snapshot_download(repo_id=dep, token=tok, local_files_only=True)
            except Exception:
                # not in cache yet — download it
                try:
                    snapshot_download(repo_id=dep, token=tok)
                except Exception as exc2:
                    print(f"  [warn]  cache {dep}: {exc2}", file=sys.stderr)

    if failed:
        print(f"\nCould not fetch: {', '.join(failed)}", file=sys.stderr)
        print("Worker will fail to start until these models are available.", file=sys.stderr)
        sys.exit(1)

    print("All models ready.", flush=True)


if __name__ == "__main__":
    main()
