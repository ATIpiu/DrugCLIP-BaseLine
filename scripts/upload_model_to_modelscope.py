"""上传 DrugCLIP 预训练 checkpoint 到 ModelScope 模型仓库。

.pt 文件在模型仓库中自动走 LFS，直接用 SDK upload_file 即可。

用法:
    python scripts/upload_model_to_modelscope.py
"""

import os
from pathlib import Path
from modelscope.hub.api import HubApi

TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")
REPO_ID   = "ATIpiu/DrugCLIP-Base"
LOCAL_PT  = r"D:\vscode\DrugCLIP\train\model\Base\checkpoint_best.pt"

api = HubApi()
api.login(TOKEN)
print(f"已登录 ModelScope")

print(f"上传 {Path(LOCAL_PT).name} ({Path(LOCAL_PT).stat().st_size/1024/1024:.0f} MB) → {REPO_ID}")
print("文件较大（~1.1 GB），请耐心等待 ...\n")

api.upload_file(
    path_or_fileobj=LOCAL_PT,
    path_in_repo="checkpoint_best.pt",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload DrugCLIP base pretrained checkpoint (UniMol 15-layer/512-dim)",
)

print(f"\n上传完成！")
print(f"模型地址：https://www.modelscope.cn/models/{REPO_ID}")
