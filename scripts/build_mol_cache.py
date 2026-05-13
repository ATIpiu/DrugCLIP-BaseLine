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
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from train.data.utils import prepare_molecule, load_atom_dict


# ── 加载 atom dict（与 InferenceEngine 一致）────────────────────────

def _load_mol_dict() -> dict:
    dict_path = ROOT / "data" / "dict_mol.txt"
    if dict_path.exists():
        d = load_atom_dict(str(dict_path))
        print(f"Atom dict : {dict_path}  ({d['num_types']} types)")
        return d
    print("WARNING: data/dict_mol.txt not found, using default ATOM_LIST")
    return None


# ── SQLite helpers（与 InferenceEngine 保持一致）────────────────────

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

    # 指纹不匹配时清空旧缓存（与 InferenceEngine 行为一致）
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


# ── Worker（线程中运行，atom_dict 通过闭包传入）─────────────────────

def make_worker(mol_dict):
    def _worker(smiles: str):
        mol_data = prepare_molecule(smiles, fast=True, atom_dict=mol_dict)
        key = smiles_key(smiles)
        if mol_data is None:
            return key, None
        blob = pickle.dumps(
            {"t": mol_data["tokens"], "d": mol_data["distances"], "e": mol_data["edge_types"]},
            protocol=4,
        )
        return key, blob
    return _worker


# ── 收集全部唯一 SMILES ───────────────────────────────────────────

def collect_all_smiles(benchmark_dir: Path, max_smiles: int = 0) -> dict:
    manifest_path = benchmark_dir / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"找不到 manifest.jsonl: {manifest_path}")

    tasks = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    all_smiles = {}  # hash → smiles
    for task in tasks:
        task_id = task["task_id"]
        ligands_csv = benchmark_dir / "tasks" / task_id / task["ligand_file"]
        if not ligands_csv.exists():
            print(f"  [skip] {task_id}: {ligands_csv} not found")
            continue
        with open(ligands_csv, newline="") as f:
            for row in csv.DictReader(f):
                smi = row["smiles"]
                key = smiles_key(smi)
                if key not in all_smiles:
                    all_smiles[key] = smi
                    if max_smiles > 0 and len(all_smiles) >= max_smiles:
                        return all_smiles

    return all_smiles


# ── 主流程 ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="预缓存 benchmark 分子 RDKit tokenization")
    p.add_argument("--benchmark-dir", default="data/benchmark/benchmark")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2),
                   help="并行线程数（默认 4）")
    p.add_argument("--batch-size", type=int, default=2000,
                   help="每批提交给线程池的 SMILES 数量")
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
    print(f"Workers   : {args.workers} threads")
    if args.max_smiles:
        print(f"Max SMILES: {args.max_smiles} (quick test mode)")
    print()

    # 1. 收集全部唯一 SMILES
    print("扫描 benchmark ligands.csv ...")
    t0 = time.time()
    all_smiles = collect_all_smiles(benchmark_dir, args.max_smiles)
    print(f"  共 {len(all_smiles):,} 个唯一 SMILES（{time.time()-t0:.1f}s）\n")

    # 2. 查已缓存
    conn = open_db(db_path, dict_fp)
    all_keys = list(all_smiles.keys())
    cached_keys = already_cached(conn, all_keys)
    todo = [(k, all_smiles[k]) for k in all_keys if k not in cached_keys]

    print(f"已缓存: {len(cached_keys):,} / {len(all_smiles):,}")
    if not todo:
        print("全部已缓存，无需重新计算。")
        conn.close()
        return

    print(f"待计算: {len(todo):,} 个 SMILES")
    print(f"开始 RDKit 3D 构象生成（{args.workers} 线程）...\n")

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
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
            f"成功 {done-failed:,} 失败 {failed} | "
            f"{rate:.0f} mol/s | ETA {eta/60:.1f} min"
        )

    elapsed = time.time() - t_start
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\n完成！写入 {done-failed:,} 条，失败 {failed} 条")
    print(f"耗时 {elapsed:.1f}s，DB 大小 {db_size_mb:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
