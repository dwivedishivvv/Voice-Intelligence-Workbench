"""Build-time model downloader. Runs once during docker build; fails the build if any
model can't be fetched, so a missing model is a build failure, not a runtime surprise."""
import sys
import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def prefetch(registry_path: str, token: str | None):
    reg = yaml.safe_load(Path(registry_path).read_text())
    failures = []
    for name, spec in reg["models"].items():
        dest = Path(spec["local_dir"])
        print(f"[prefetch] {name}: {spec['id']} -> {dest}", flush=True)
        try:
            snapshot_download(repo_id=spec["id"], revision=spec.get("revision", "main"),
                               local_dir=dest, token=token if spec.get("gated") else None,
                               max_workers=4)
            # Nested repo-id references inside a pipeline's config.yaml (e.g. pyannote's
            # segmentation submodel) get resolved at runtime via hf_hub_download using
            # repo id, not a filesystem path — so they must land in the *standard* HF
            # cache (HF_HOME), not a flat local_dir copy, or offline lookup fails.
            for dep_id in spec.get("depends_on", []):
                print(f"[prefetch] {name} dependency: {dep_id}", flush=True)
                snapshot_download(repo_id=dep_id, token=token if spec.get("gated") else None,
                                   max_workers=4)
        except Exception as e:
            failures.append((name, str(e)))
            continue
        if not any(dest.iterdir()):
            failures.append((name, "empty directory after download"))

    if failures:
        for n, e in failures:
            print(f"[prefetch] FAILED {n}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[prefetch] all {len(reg['models'])} models cached", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="models/REGISTRY.yaml")
    p.add_argument("--token", default=None)
    a = p.parse_args()
    prefetch(a.registry, a.token)
