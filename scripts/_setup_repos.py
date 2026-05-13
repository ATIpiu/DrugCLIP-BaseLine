"""创建 ModelScope 数据集仓库（两个 part）。"""
import os
import sys
from modelscope.hub.api import HubApi

TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")
api = HubApi()
api.login(TOKEN)
print("已登录 ModelScope", flush=True)

repos = [
    "ATIpiu/THU-ATOM_PDBbind_For_AI4S",
    "ATIpiu/THU-ATOM_PDBbind_For_AI4S_2",
]

for repo_id in repos:
    try:
        result = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
        )
        print(f"[OK] 创建成功: {result}", flush=True)
    except Exception as e:
        print(f"[SKIP/FAIL] {repo_id}: {e}", flush=True)
