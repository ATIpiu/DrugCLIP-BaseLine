"""上传 THU-ATOM_PDBbind 到 ModelScope，拆成两个仓库各 ~935MB。

ModelScope 数据集仓库限制 1.6GB，完整数据集 1.87GB 超限，拆半后各 ~935MB 可以通过。

Repo 1: ATIpiu/THU-ATOM_PDBbind_For_AI4S   (前 1370 个复合物)
Repo 2: ATIpiu/THU-ATOM_PDBbind_For_AI4S_2 (后 1369 个复合物)

用法:
    python scripts/upload_to_modelscope.py
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")
SRC_DIR  = ROOT / "data" / "THU-ATOM_PDBbind"

REPOS = [
    ("ATIpiu/THU-ATOM_PDBbind_For_AI4S",   ROOT / "temp_ms_upload_1"),
    ("ATIpiu/THU-ATOM_PDBbind_For_AI4S_2", ROOT / "temp_ms_upload_2"),
]


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def rm_dir(path: Path):
    if path.exists():
        def rm_ro(func, p, _):
            os.chmod(p, 0o666)
            func(p)
        shutil.rmtree(str(path), onerror=rm_ro)


# ── 获取所有复合物目录，排序后二分 ───────────────────────────────
all_dirs = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])
mid = len(all_dirs) // 2
splits = [all_dirs[:mid], all_dirs[mid:]]

print(f"总目录: {len(all_dirs)}")
print(f"Part 1: {len(splits[0])} 个 ({splits[0][0].name} ~ {splits[0][-1].name})")
print(f"Part 2: {len(splits[1])} 个 ({splits[1][0].name} ~ {splits[1][-1].name})\n")

run(["git", "lfs", "install"])

for (repo_id, clone_dir), dirs in zip(REPOS, splits):
    print(f"\n{'='*60}", flush=True)
    print(f"上传 {repo_id}  ({len(dirs)} 个复合物)", flush=True)
    print(f"{'='*60}", flush=True)

    clone_url = f"https://oauth2:{TOKEN}@www.modelscope.cn/datasets/{repo_id}.git"

    # 1. 清理并克隆
    rm_dir(clone_dir)
    run(["git", "clone", clone_url, str(clone_dir)])
    os.chdir(clone_dir)

    # 2. 设置 LFS tracking
    run(["git", "lfs", "track", "*.pdb", "*.sdf", "*.mol2"])
    run(["git", "add", ".gitattributes"])
    result = subprocess.run(["git", "status", "--porcelain"],
                            capture_output=True, text=True)
    if result.stdout.strip():
        run(["git", "commit", "-m", "Add LFS tracking"])

    # 3. 复制对应的那一半目录
    print(f"复制 {len(dirs)} 个目录 ...", flush=True)
    for d in dirs:
        dst = clone_dir / d.name
        shutil.copytree(str(d), str(dst))
    print("复制完成", flush=True)

    # 4. add → commit → push
    run(["git", "add", "."])
    run(["git", "commit", "-m",
         f"Upload THU-ATOM PDBbind Part ({dirs[0].name}~{dirs[-1].name}, {len(dirs)} complexes)"])
    run(["git", "push", "origin", "master"])

    print(f"✓ {repo_id} 上传完成", flush=True)
    print(f"  https://www.modelscope.cn/datasets/{repo_id}", flush=True)

print("\n\n全部上传完成！", flush=True)
