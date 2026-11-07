"""Download every Prompt 2 source into data/raw/ and record data/MANIFEST.json.

Idempotent and resumable: files that already exist with the expected size (or
sha256) are skipped. Sources and pinned revisions come from configs/data.yaml.

- hf_file:            hf_hub_download at a pinned revision.
- url_file:           plain HTTPS download, verified against a published sha256.
- hf_parquet_sample:  evenly spaced row groups streamed from remote parquet
                      files until a compressed-byte cap (FineWeb-2 ben_Beng).

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.download [--only SOURCE ...]
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path
from typing import Any

from data.common import load_config, peak_rss_mb, repo_path, setup_env, sha256_file, write_json


def _record(path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Manifest entry for one local file."""
    return {
        "path": path.relative_to(repo_path("")).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def download_hf_files(name: str, src: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    """Download pinned files from a HF dataset repo."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(src["repo"], revision=src["revision"], files_metadata=True)
    remote_sizes = {s.rfilename: s.size for s in info.siblings or []}
    entries = []
    for fname in src["files"]:
        target = raw_dir / name / fname
        expected = remote_sizes.get(fname)
        if expected is None:
            raise RuntimeError(f"{src['repo']}@{src['revision']} has no file {fname!r}")
        if not (target.is_file() and target.stat().st_size == expected):
            print(f"  downloading {src['repo']}/{fname} ({expected / 1e6:,.1f} MB)", flush=True)
            hf_hub_download(
                src["repo"], fname, repo_type="dataset", revision=src["revision"],
                local_dir=raw_dir / name,
            )
        else:
            print(f"  present  {fname}", flush=True)
        if target.stat().st_size != expected:
            raise RuntimeError(f"size mismatch for {target}: {target.stat().st_size} != {expected}")
        entries.append(_record(target, {"repo": src["repo"], "revision": src["revision"], "remote_path": fname}))
    return entries


def download_url_file(name: str, src: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    """Download one file over HTTPS and verify its published sha256."""
    target = raw_dir / name / src["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.is_file() and sha256_file(target) == src["sha256"]):
        print(f"  downloading {src['url']}", flush=True)
        tmp = target.with_suffix(target.suffix + ".part")
        req = urllib.request.Request(src["url"], headers={"User-Agent": "tokjhallm-data/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while block := resp.read(2**20):
                out.write(block)
        tmp.replace(target)
    digest = sha256_file(target)
    if digest != src["sha256"]:
        raise RuntimeError(f"sha256 mismatch for {target}: {digest} != {src['sha256']}")
    return [_record(target, {"url": src["url"], "published_sha256": src["sha256"]})]


def sample_parquet_row_groups(name: str, src: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    """Stream evenly spaced row groups from remote parquet files up to a compressed-byte cap.

    Row groups are chosen round-robin across files at evenly spaced positions, so
    the sample is not biased toward the start of any one file (crawl order).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_dir = raw_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_SAMPLE_COMPLETE.json"
    if done_marker.is_file():
        print("  present  (sample complete)", flush=True)
        return [_record(p, {"repo": src["repo"], "revision": src["revision"]}) for p in sorted(out_dir.glob("part-*.parquet"))]

    fs = HfFileSystem()
    cap = int(src["cap_compressed_bytes"])
    columns = list(src["columns"])
    handles = []
    for fname in src["files"]:
        path = f"datasets/{src['repo']}@{src['revision']}/{fname}"
        f = fs.open(path, "rb", block_size=2**20)
        pf = pq.ParquetFile(f)
        col_idx = [pf.schema_arrow.get_field_index(c) for c in columns]
        md = pf.metadata
        sizes = [sum(md.row_group(rg).column(i).total_compressed_size for i in col_idx) for rg in range(md.num_row_groups)]
        handles.append((fname, f, pf, sizes))

    mean_rg = float(np.mean([s for _, _, _, sizes in handles for s in sizes]))
    per_file = int(np.ceil(cap / mean_rg / len(handles))) + 1
    rng = np.random.default_rng(int(src["seed"]))
    plan: list[list[int]] = []
    for _, _, _, sizes in handles:
        n = len(sizes)
        positions = np.linspace(0, n - 1, num=min(per_file, n)).astype(int)
        jitter = rng.integers(0, max(1, n // max(1, per_file)), size=len(positions))
        plan.append(sorted(set(int(min(n - 1, p + j)) for p, j in zip(positions, jitter))))

    total = 0
    part = 0
    taken: list[dict[str, Any]] = []
    entries = []
    for step in range(max(len(p) for p in plan)):
        for (fname, _, pf, sizes), rgs in zip(handles, plan):
            if step >= len(rgs) or total >= cap:
                continue
            rg = rgs[step]
            table = pf.read_row_group(rg, columns=columns)
            table = table.append_column("source_file", pa.array([fname] * table.num_rows))
            table = table.append_column("row_group", pa.array([rg] * table.num_rows, type=pa.int32()))
            out = out_dir / f"part-{part:05d}.parquet"
            pq.write_table(table, out, compression="zstd")
            total += sizes[rg]
            taken.append({"file": fname, "row_group": rg, "compressed_bytes": sizes[rg], "rows": table.num_rows})
            part += 1
            if part % 25 == 0:
                print(f"  {part} row groups, {total / 1e6:,.0f} / {cap / 1e6:,.0f} MB compressed, peak RSS {peak_rss_mb():,.0f} MB", flush=True)
        if total >= cap:
            break
    for _, f, _, _ in handles:
        f.close()
    for p in sorted(out_dir.glob("part-*.parquet")):
        entries.append(_record(p, {"repo": src["repo"], "revision": src["revision"]}))
    write_json(done_marker, {"compressed_bytes_read": total, "row_groups": taken})
    print(f"  sampled {part} row groups = {total / 1e6:,.1f} MB compressed", flush=True)
    return entries


HANDLERS = {
    "hf_file": download_hf_files,
    "url_file": download_url_file,
    "hf_parquet_sample": sample_parquet_row_groups,
}


def main() -> int:
    """Download all (or selected) sources and write data/MANIFEST.json."""
    parser = argparse.ArgumentParser(description="Download Prompt 2 data sources.")
    parser.add_argument("--only", nargs="*", help="source names to download (default: all)")
    args = parser.parse_args()

    cfg = load_config()
    setup_env(cfg)
    raw_dir = repo_path(cfg["paths"]["raw_dir"])
    manifest_path = repo_path(cfg["paths"]["manifest"])
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("sources", {})

    for name, src in cfg["sources"].items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        print(f"[{name}] {src['kind']}", flush=True)
        entries = HANDLERS[src["kind"]](name, src, raw_dir)
        manifest["sources"][name] = {
            "kind": src["kind"], "role": src["role"], "license": src["license"],
            "url": src.get("url") or src.get("page"), "files": entries,
            "bytes": sum(e["bytes"] for e in entries),
        }
        print(f"  done in {time.time() - t0:,.0f}s", flush=True)

    manifest["total_download_bytes"] = sum(s["bytes"] for s in manifest["sources"].values())
    manifest["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(manifest_path, manifest)
    print(f"total on disk: {manifest['total_download_bytes'] / 1e9:.3f} GB; peak RSS {peak_rss_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
