"""预缓存 benchmark 全部分子的 RDKit tokenization 到 SQLite。

与 checkpoint 无关，只要跑一次，后续任何 checkpoint 的推理都跳过 RDKit。

用法:
    python scripts/build_mol_cache.py
    python scripts/build_mol_cache.py --benchmark-dir data/benchmark/benchmark --output-dir output
    python scripts/build_mol_cache.py --workers 4        # 并行线程数
    python scripts/build_mol_cache.py --max-smiles 2000  # 快速测试：只缓存前 N 个
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from train.data.utils import prepare_molecule, load_atom_dict


# ── Atom dict ────────────────────────────────────────────────────

def _load_mol_dict() -> dict:
    dict_path = ROOT / "data" / "dict_mol.txt"
    if dict_path.exists():
        d = load_atom_dict(str(dict_path))
        print(f"Atom dict : {dict_path}  ({d['num_types']} types)")
        return d
    print("WARNING: data/dict_mol.txt not found, using default ATOM_LIST")
    return None


# ── SQLite helpers ───────────────────────────────────────────────

def _dict_fingerprint(mol_dict: dict) -> str:
    if not mol_dict:
        return "default"
    items = json.dumps(sorted(mol_dict["atom_to_idx"].items()))
    return hashlib.md5(items.encode()).hexdigest()[:12]


def open_db(db_path: Path, dict_fp: str = "") -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS db_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mol_tokens (
            smiles_hash TEXT PRIMARY KEY,
            data        BLOB NOT NULL
        )
    """)
    conn.commit()

    if dict_fp:
        row = conn.execute(
            "SELECT value FROM db_meta WHERE key = 'dict_fingerprint'"
        ).fetchone()
        stored = row[0] if row else ""
        if stored != dict_fp:
            deleted = conn.execute("DELETE FROM mol_tokens").rowcount
            conn.execute(
                "INSERT OR REPLACE INTO db_meta (key, value) VALUES ('dict_fingerprint', ?)",
                (dict_fp,)
            )
            conn.commit()
            if deleted:
                print(f"Atom dict changed → cleared {deleted:,} stale cache entries")

    return conn


def smiles_key(smiles: str) -> str:
    return hashlib.md5(smiles.encode()).hexdigest()


def already_cached(conn: sqlite3.Connection, keys: list) -> set:
    result = set()
    chunk = 900
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT smiles_hash FROM mol_tokens WHERE smiles_hash IN ({placeholders})",
            batch,
        ).fetchall()
        result.update(r[0] for r in rows)
    return result


def insert_batch(conn: sqlite3.Connection, entries: list):
    conn.executemany(
        "INSERT OR IGNORE INTO mol_tokens (smiles_hash, data) VALUES (?, ?)",
        entries,
    )
    conn.commit()


# ── Worker ───────────────────────────────────────────────────────

def make_worker(mol_dict):
    def _worker(args):
        key, smiles = args
        mol_data = prepare_molecule(smiles, fast=True, atom_dict=mol_dict)
        if mol_data is None:
            return key, None
        blob = pickle.dumps(
            {"t": mol_data["tokens"], "d": mol_data["distances"], "e": mol_data["edge_types"]},
            protocol=4,
        )
        return key, blob
    return _worker


# ── 流式读取 SMILES（不一次性加载全部到内存）────────────────────

def iter_smiles_chunks(benchmark_dir: Path, chunk_size: int, max_smiles: int = 0):
    """逐块 yield (key, smiles) 列表，seen_hashes 集合用于全局去重。"""
    manifest_path = benchmark_dir / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"找不到 manifest.jsonl: {manifest_path}")

    tasks = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    seen_hashes: set = set()
    chunk: list = []
    total_yielded = 0

    for task in tasks:
        task_id = task["task_id"]
        ligands_csv = benchmark_dir / "tasks" / task_id / task["ligand_file"]
        if not ligands_csv.exists():
            print(f"  [skip] {task_id}: ligands.csv not found")
            continue

        with open(ligands_csv, newline="") as f:
            for row in csv.DictReader(f):
                smi = row["smiles"]
                key = smiles_key(smi)
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)
                chunk.append((key, smi))
                total_yielded += 1

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

                if max_smiles > 0 and total_yielded >= max_smiles:
                    if chunk:
                        yield chunk
                    return

    if chunk:
        yield chunk


# ── 主流程 ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="预缓存 benchmark 分子 RDKit tokenization")
    p.add_argument("--benchmark-dir", default="data/benchmark/benchmark")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    p.add_argument("--batch-size", type=int, default=2000,
                   help="每批处理的 SMILES 数量（控制峰值内存）")
    p.add_argument("--max-smiles", type=int, default=0,
                   help="最多缓存 N 个 SMILES（0=全部，用于快速测试）")
    args = p.parse_args()

    benchmark_dir = ROOT / args.benchmark_dir
    cache_dir = ROOT / args.output_dir / "mol_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "tokens.db"

    mol_dict = _load_mol_dict()
    dict_fp  = _dict_fingerprint(mol_dict)
    _worker  = make_worker(mol_dict)

    print(f"Benchmark : {benchmark_dir}")
    print(f"SQLite DB : {db_path}")
    print(f"Workers   : {args.workers} threads  |  Batch: {args.batch_size}")
    if args.max_smiles:
        print(f"Max SMILES: {args.max_smiles} (quick test mode)")
    print()

    conn = open_db(db_path, dict_fp)

    total_new   = 0
    total_skip  = 0
    total_fail  = 0
    t_start     = time.time()
    batch_idx   = 0

    for chunk in iter_smiles_chunks(benchmark_dir, args.batch_size, args.max_smiles):
        batch_idx += 1

        # 查缓存
        keys = [k for k, _ in chunk]
        cached = already_cached(conn, keys)
        todo   = [(k, smi) for k, smi in chunk if k not in cached]
        total_skip += len(cached)

        if not todo:
            continue

        # 并行 RDKit
        to_insert: list = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key, blob in pool.map(_worker, todo):
                if blob is not None:
                    to_insert.append((key, blob))
                else:
                    total_fail += 1

        insert_batch(conn, to_insert)
        total_new += len(to_insert)

        # 显式释放，避免 blob 在内存中堆积
        del to_insert
        del todo
        gc.collect()

        elapsed = time.time() - t_start
        done    = total_new + total_skip
        rate    = done / elapsed if elapsed > 0 else 0
        print(
            f"  [batch {batch_idx:>3}] 已处理 {done:>8,} "
            f"(新写 {total_new:,} 跳过 {total_skip:,} 失败 {total_fail}) "
            f"| {rate:.0f} mol/s"
        )

    elapsed = time.time() - t_start
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\n完成！新写 {total_new:,} 条，跳过 {total_skip:,} 条，失败 {total_fail} 条")
    print(f"耗时 {elapsed:.1f}s，DB 大小 {db_size_mb:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
