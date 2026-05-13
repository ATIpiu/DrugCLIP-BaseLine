"""上传 THU-ATOM_PDBbind Part 2 到 ModelScope。"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT      = Path(r"D:\vscode\DrugCLIP")
TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")
SRC_DIR   = ROOT / "data" / "THU-ATOM_PDBbind"
REPO_ID   = "ATIpiu/THU-ATOM_PDBbind_For_AI4S_2"
CLONE_DIR = ROOT / "temp_ms_upload_2"


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def rm_ro(func, p, _):
    os.chmod(p, 0o666)
    func(p)


all_dirs = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])
mid = len(all_dirs) // 2
dirs = all_dirs[mid:]

print(f"总目录: {len(all_dirs)}，Part2 个数: {len(dirs)}", flush=True)
print(f"范围: {dirs[0].name} ~ {dirs[-1].name}", flush=True)

# 清理旧目录
if CLONE_DIR.exists():
    print("清理旧 clone 目录 ...", flush=True)
    shutil.rmtree(str(CLONE_DIR), onerror=rm_ro)

# Git LFS 全局安装
run(["git", "lfs", "install"])

# 克隆仓库
clone_url = f"https://oauth2:{TOKEN}@www.modelscope.cn/datasets/{REPO_ID}.git"
run(["git", "clone", clone_url, str(CLONE_DIR)])
os.chdir(CLONE_DIR)

# 设置 LFS tracking
run(["git", "lfs", "track", "*.pdb", "*.sdf", "*.mol2"])
run(["git", "add", ".gitattributes"])
result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
if result.stdout.strip():
    run(["git", "commit", "-m", "Add LFS tracking"])

# 复制数据
print(f"\n复制 {len(dirs)} 个目录 ...", flush=True)
for i, d in enumerate(dirs):
    dst = CLONE_DIR / d.name
    shutil.copytree(str(d), str(dst))
    if (i + 1) % 100 == 0:
        print(f"  已复制 {i+1}/{len(dirs)}", flush=True)
print("复制完成", flush=True)

# add → commit → push
run(["git", "add", "."])
run(["git", "commit", "-m",
     f"Upload THU-ATOM PDBbind Part2 ({dirs[0].name}~{dirs[-1].name}, {len(dirs)} complexes)"])
run(["git", "push", "origin", "master"])

print(f"\n上传完成！", flush=True)
print(f"  https://www.modelscope.cn/datasets/{REPO_ID}", flush=True)
