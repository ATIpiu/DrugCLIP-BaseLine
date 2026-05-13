"""预缓存 benchmark 全部分子的 RDKit tokenization 到 SQLite。

与 checkpoint 无关，只要跑一次，后续任何 checkpoint 的推理都跳过 RDKit。

用法:
    python scripts/build_mol_cache.py
    python scripts/build_mol_cache.py --benchmark-dir data/benchmark/benchmark --output-dir output
    python scripts/build_mol_cache.py --workers 4   # 并行进程数
"""

import argparse
import csv
import hashlib
import os
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from train.data.utils import prepare_molecule


# ── SQLite helpers（与 InferenceEngine 保持一致）────────────────────

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mol_tokens (
            smiles_hash TEXT PRIMARY KEY,
            data        BLOB NOT NULL
        )
    """)
    conn.commit()
    return conn


def smiles_key(smiles: str) -> str:
    return hashlib.md5(smiles.encode()).hexdigest()


def already_cached(conn: sqlite3.Connection, keys: list[str]) -> set[str]:
    """返回已在 DB 中的 hash 集合（批量查询，避免逐条 SELECT）。"""
    result = set()
    chunk = 900  # SQLite 变量上限 999
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT smiles_hash FROM mol_tokens WHERE smiles_hash IN ({placeholders})",
            batch,
        ).fetchall()
        result.update(r[0] for r in rows)
    return result


def insert_batch(conn: sqlite3.Connection, entries: list[tuple]):
    """批量写入 (smiles_hash, blob) 列表。"""
    conn.executemany(
        "INSERT OR IGNORE INTO mol_tokens (smiles_hash, data) VALUES (?, ?)",
        entries,
    )
    conn.commit()


# ── Worker function（在子进程中运行）────────────────────────────────

def _worker(smiles: str) -> tuple[str, bytes | None]:
    """返回 (smiles_hash, blob) 或 (smiles_hash, None) 若 RDKit 失败。"""
    mol_data = prepare_molecule(smiles, fast=True, atom_dict=None)
    key = smiles_key(smiles)
    if mol_data is None:
        return key, None
    blob = pickle.dumps(
        {"t": mol_data["tokens"], "d": mol_data["distances"], "e": mol_data["edge_types"]},
        protocol=4,
    )
    return key, blob


# ── 收集全部唯一 SMILES ───────────────────────────────────────────

def collect_all_smiles(benchmark_dir: Path) -> dict[str, str]:
    """返回 {smiles_hash: smiles} 的去重字典。"""
    manifest_path = benchmark_dir / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"找不到 manifest.jsonl: {manifest_path}")

    import json
    tasks = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    all_smiles: dict[str, str] = {}  # hash → smiles
    for task in tasks:
        task_id = task["task_id"]
        ligands_csv = benchmark_dir / "tasks" / task_id / task["ligand_file"]
        if not ligands_csv.exists():
            print(f"  [skip] {task_id}: {ligands_csv} 不存在")
            continue
        with open(ligands_csv, newline="") as f:
            for row in csv.DictReader(f):
                smi = row["smiles"]
                key = smiles_key(smi)
                if key not in all_smiles:
                    all_smiles[key] = smi

    return all_smiles


# ── 主流程 ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="预缓存 benchmark 分子 RDKit tokenization")
    p.add_argument("--benchmark-dir", default="data/benchmark/benchmark")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="并行 RDKit worker 进程数（默认 CPU 核心数 - 1）")
    p.add_argument("--batch-size", type=int, default=2000,
                   help="每批提交给进程池的 SMILES 数量")
    args = p.parse_args()

    benchmark_dir = ROOT / args.benchmark_dir
    cache_dir = ROOT / args.output_dir / "mol_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "tokens.db"

    print(f"Benchmark : {benchmark_dir}")
    print(f"SQLite DB : {db_path}")
    print(f"Workers   : {args.workers}")
    print()

    # 1. 收集全部唯一 SMILES
    print("扫描 benchmark ligands.csv ...")
    t0 = time.time()
    all_smiles = collect_all_smiles(benchmark_dir)
    print(f"  共 {len(all_smiles):,} 个唯一 SMILES（耗时 {time.time()-t0:.1f}s）\n")

    # 2. 查哪些已经缓存了
    conn = open_db(db_path)
    all_keys = list(all_smiles.keys())
    cached_keys = already_cached(conn, all_keys)
    todo = [(k, all_smiles[k]) for k in all_keys if k not in cached_keys]

    print(f"已缓存: {len(cached_keys):,} / {len(all_smiles):,}")
    if not todo:
        print("全部已缓存，无需重新计算。")
        conn.close()
        return

    print(f"待计算: {len(todo):,} 个 SMILES")
    print(f"开始 RDKit 3D 构象生成（{args.workers} 进程）...\n")

    # 3. 并行计算并写入 DB
    batch_size = args.batch_size
    total = len(todo)
    done = 0
    failed = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = todo[batch_start: batch_start + batch_size]
        smiles_batch = [smi for _, smi in batch]

        to_insert = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, smi): smi for smi in smiles_batch}
            for f in as_completed(futures):
                key, blob = f.result()
                if blob is not None:
                    to_insert.append((key, blob))
                else:
                    failed += 1

        insert_batch(conn, to_insert)
        done += len(batch)

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  [{done:>7,}/{total:,}] "
            f"成功 {done-failed:,} 失败 {failed:,} | "
            f"{rate:.0f} mol/s | ETA {eta/60:.1f} min"
        )

    elapsed = time.time() - t_start
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\n完成！共写入 {done-failed:,} 条，失败 {failed:,} 条")
    print(f"耗时 {elapsed/60:.1f} 分钟，DB 大小 {db_size_mb:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
