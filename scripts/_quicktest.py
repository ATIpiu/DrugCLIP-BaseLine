"""快速端到端流程测试：预加载 + 推理"""
if __name__ != "__main__":
    raise SystemExit("Run as script: python -m scripts._quicktest")
import warnings, os, json, time
warnings.filterwarnings("ignore")
os.environ["DRUGCLIP_SILENT"] = "1"
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from pathlib import Path
from train.config import Config
from train.inference import InferenceEngine

TASK_ID   = "litpcba_TP53"
CKPT      = "output/models/drugclip/checkpoints/best.pt"
BENCH_DIR = "data/benchmark/benchmark"

config = Config()
config.data.output_dir    = "output"
config.data.benchmark_dir = BENCH_DIR

print("=" * 60)
print(f"  快速推理测试: {TASK_ID}")
print("=" * 60)

# ── 初始化引擎 ────────────────────────────────────────────────────
t0 = time.time()
engine = InferenceEngine(config, checkpoint_path=CKPT)
print(f"[引擎初始化] {time.time()-t0:.1f}s\n")

tasks     = [json.loads(l) for l in open(f"{BENCH_DIR}/manifest.jsonl")]
task_info = next(t for t in tasks if t["task_id"] == TASK_ID)
task_dir  = Path(BENCH_DIR) / "tasks" / TASK_ID

# ── 第一次：预加载（RDKit + 计算 embedding，写入两级缓存）────────
print(f"[第一次运行] 预加载  ({task_info['num_ligands']} 个配体)")
t0 = time.time()
scores = engine._score_single_task(task_dir, task_info)
dt1 = time.time() - t0
print(f"  结果: {len(scores)} 个配体已评分  耗时: {dt1:.1f}s")

# ── 第二次：全缓存命中 ────────────────────────────────────────────
print(f"\n[第二次运行] 使用缓存推理")
t0 = time.time()
scores2 = engine._score_single_task(task_dir, task_info)
dt2 = time.time() - t0
print(f"  结果: {len(scores2)} 个配体已评分  耗时: {dt2:.2f}s  加速: {dt1/dt2:.0f}x")

# ── 输出 Top-5 ────────────────────────────────────────────────────
top5 = sorted(scores, key=lambda x: -x[1])[:5]
print(f"\n[Top-5 配体 by score]")
for lig_id, sc in top5:
    print(f"  {lig_id:40s}  score={sc:.4f}")

# ── 缓存文件统计 ──────────────────────────────────────────────────
cache_dir = Path("output/mol_cache")
db_path   = cache_dir / "tokens.db"
db_kb = db_path.stat().st_size // 1024 if db_path.exists() else 0
emb_files = list((cache_dir / "emb").glob(f"**/{TASK_ID}.pkl"))
emb_kb = emb_files[0].stat().st_size // 1024 if emb_files else 0
print(f"\n[缓存文件]")
print(f"  SQLite token DB : {db_kb:,} KB  ({db_path})")
if emb_files:
    print(f"  Embedding pkl   : {emb_kb:,} KB  ({emb_files[0]})")
print()
