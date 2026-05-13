#!/usr/bin/env bash
# DrugCLIP 一键环境配置 + 快速训练启动
#
# 用法:
#   bash setup.sh                 # 使用 fasttest（10个样本）快速验证
#   bash setup.sh --full-data     # 同时下载完整 THU-ATOM_PDBbind (~1.87 GB)
#   bash setup.sh --skip-train    # 只配置环境和下载模型，不自动训练

set -e

CUDA_VERSION="cu121"   # 按需改为 cu118 / cu124 / cu128
TRAIN_DATA="data/fasttest"
FULL_DATA=false
SKIP_TRAIN=false

for arg in "$@"; do
  case $arg in
    --full-data)  FULL_DATA=true ;;
    --skip-train) SKIP_TRAIN=true ;;
  esac
done

echo "======================================================"
echo "  DrugCLIP Setup"
echo "======================================================"

# ── 1. Python 检查 ────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] python3 not found. Install Python >= 3.10 first." && exit 1
fi
PY=python3
echo "[✓] $($PY --version)"

# ── 2. 虚拟环境（若未在 conda/venv 中） ──────────────────────────────
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
  [ ! -d ".venv" ] && $PY -m venv .venv && echo "[*] 已创建虚拟环境 .venv"
  source .venv/bin/activate
  PY=python
  echo "[✓] 虚拟环境已激活"
else
  PY=python
  echo "[✓] 当前环境: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}"
fi

# ── 3. PyTorch ────────────────────────────────────────────────────────
if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[*] 安装 PyTorch (CUDA ${CUDA_VERSION}) ..."
  pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" -q
  echo "[✓] PyTorch 安装完成"
else
  echo "[✓] PyTorch $($PY -c 'import torch; print(torch.__version__)') (CUDA 可用)"
fi

# ── 4. 项目依赖 ───────────────────────────────────────────────────────
echo "[*] 安装项目依赖 ..."
pip install rdkit tqdm scipy numpy modelscope -q
pip install -r requirements.txt -q --no-deps 2>/dev/null || true
echo "[✓] 依赖安装完成"

# ── 5. 下载 Base 模型 ─────────────────────────────────────────────────
MODEL_DIR="train/model/Base"
MODEL_FILE="${MODEL_DIR}/checkpoint_best.pt"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
  echo "[*] 下载 DrugCLIP Base 模型 (~1.1 GB) ..."
  $PY - <<'PYEOF'
import os, sys
from modelscope import snapshot_download
local = snapshot_download(
    model_id='ATIpiu/DrugCLIP-Base',
    repo_type='model',
    local_dir='train/model/Base',
    allow_patterns=['checkpoint_best.pt'],
)
pt = os.path.join(local, 'checkpoint_best.pt')
if os.path.exists(pt):
    print(f'[✓] 模型下载完成: {pt}')
else:
    print('[ERROR] checkpoint_best.pt 未找到', file=sys.stderr); sys.exit(1)
PYEOF
else
  echo "[✓] Base 模型已存在: ${MODEL_FILE}"
fi

# ── 6. （可选）下载完整数据集 ────────────────────────────────────────
if $FULL_DATA; then
  DATA_DIR="data/THU-ATOM_PDBbind"
  if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "[*] 下载 THU-ATOM_PDBbind 数据集 (~1.87 GB) ..."
    $PY - <<PYEOF
from modelscope import snapshot_download
snapshot_download(
    model_id='ATIpiu/THU-ATOM_PDBbind_For_AI4S',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
)
print('[✓] 数据集下载完成: ${DATA_DIR}')
PYEOF
    TRAIN_DATA="$DATA_DIR"
  else
    echo "[✓] 数据集已存在: ${DATA_DIR}"
    TRAIN_DATA="$DATA_DIR"
  fi
fi

# ── 完成提示 ──────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  配置完成"
echo "  训练数据 : ${TRAIN_DATA}"
echo "  Base 模型: ${MODEL_FILE}"
echo "======================================================"

if $SKIP_TRAIN; then
  echo ""
  echo "手动训练命令:"
  echo "  python -m train.run --mode train \\"
  echo "    --train-data ${TRAIN_DATA} \\"
  echo "    --pretrained ${MODEL_FILE} \\"
  echo "    --epochs 50 --batch-size 32"
  exit 0
fi

# ── 7. 开始训练 ───────────────────────────────────────────────────────
echo ""
echo "[*] 开始训练 (${TRAIN_DATA}) ..."
$PY -m train.run \
  --mode train \
  --train-data "${TRAIN_DATA}" \
  --pretrained "${MODEL_FILE}" \
  --epochs 50 \
  --batch-size 32
