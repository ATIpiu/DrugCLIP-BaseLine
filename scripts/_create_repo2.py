"""尝试创建 ModelScope 数据集仓库 ATIpiu/THU-ATOM_PDBbind_For_AI4S_2"""
import os
import sys
from modelscope.hub.api import HubApi

TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")
api = HubApi()
api.login(TOKEN)
print("已登录 ModelScope", flush=True)

methods = [
    ("create_dataset", lambda: api.create_dataset(
        dataset_name="THU-ATOM_PDBbind_For_AI4S_2",
        namespace="ATIpiu",
        is_public=True,
    )),
    ("create_repo type=dataset", lambda: api.create_repo(
        repo_id="ATIpiu/THU-ATOM_PDBbind_For_AI4S_2",
        repo_type="dataset",
        private=False,
    )),
]

for name, fn in methods:
    try:
        result = fn()
        print(f"[OK] {name} -> {result}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)

print("全部方法失败，请手动在 ModelScope 网页创建仓库", flush=True)
sys.exit(1)
