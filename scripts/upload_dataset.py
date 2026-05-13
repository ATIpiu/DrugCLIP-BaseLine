"""上传 THU-ATOM_PDBbind 到 ModelScope，强制 .pdb/.sdf/.mol2 走 LFS。

用法:
    python scripts/upload_dataset.py
"""
from pathlib import Path

# 在导入 modelscope 之后、调用 API 之前，把生物信息格式加入 LFS 列表
import modelscope.utils.repo_utils as _repo_utils
for _ext in ['.pdb', '.sdf', '.mol2']:
    if _ext not in _repo_utils.DATASET_LFS_SUFFIX:
        _repo_utils.DATASET_LFS_SUFFIX.append(_ext)

from modelscope.hub.api import HubApi

TOKEN   = "REMOVED_SEE_ENV"
REPO_ID = "ATIpiu/THU-ATOM_PDBbind_For_AI4S"
SRC_DIR = r"D:\vscode\DrugCLIP\data\THU-ATOM_PDBbind"

api = HubApi()
api.login(TOKEN)
print("已登录 ModelScope", flush=True)

size_gb = sum(f.stat().st_size for f in Path(SRC_DIR).rglob("*") if f.is_file()) / 1e9
print(f"数据集大小: {size_gb:.2f} GB，开始上传 ...", flush=True)

api.upload_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=SRC_DIR,
    path_in_repo="",
    commit_message="Upload THU-ATOM PDBbind (2739 complexes, ESMFold-aligned)",
    ignore_patterns=["*.cache", "__pycache__", ".ms_upload_cache"],
    max_workers=12,
    use_cache=False,
)

print(f"\n上传完成！https://www.modelscope.cn/datasets/{REPO_ID}", flush=True)
