"""上传预缓存文件到 ModelScope。

用法:
    python scripts/upload_cache_to_modelscope.py --token YOUR_TOKEN
    python scripts/upload_cache_to_modelscope.py --token YOUR_TOKEN --repo ATIpiu/DrugCLIP-Cache
    python scripts/upload_cache_to_modelscope.py --token YOUR_TOKEN --dry-run   # 只检查不上传
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    p = argparse.ArgumentParser(description="上传 mol_cache 到 ModelScope")
    p.add_argument("--token", default=os.environ.get("MODELSCOPE_TOKEN", ""),
                   help="ModelScope token（也可设环境变量 MODELSCOPE_TOKEN）")
    p.add_argument("--repo", default="ATIpiu/DrugCLIP-Cache",
                   help="ModelScope 数据集仓库 ID（默认 ATIpiu/DrugCLIP-Cache）")
    p.add_argument("--cache-dir", default="output/mol_cache",
                   help="本地缓存目录（含 tokens.db）")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印要上传的文件，不实际上传")
    args = p.parse_args()

    token = args.token
    if not token:
        sys.exit("错误：请通过 --token 或环境变量 MODELSCOPE_TOKEN 提供 token")

    cache_dir = ROOT / args.cache_dir
    db_path   = cache_dir / "tokens.db"

    if not db_path.exists():
        sys.exit(f"缓存文件不存在：{db_path}\n请先运行 python scripts/build_mol_cache.py")

    print(f"缓存文件 : {db_path}")
    print(f"文件大小 : {human_size(db_path)}")
    print(f"目标仓库 : {args.repo}")
    print()

    if args.dry_run:
        print("[dry-run] 跳过实际上传。")
        return

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        sys.exit("请先安装 modelscope：pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")

    api = HubApi()
    api.login(token)

    # 确保仓库存在（不存在则创建）
    try:
        api.create_dataset(args.repo, exists_ok=True)
        print(f"数据集仓库已就绪：{args.repo}")
    except Exception as e:
        print(f"[warn] 创建/检查仓库时出错（可能已存在）：{e}")

    print(f"开始上传 tokens.db → {args.repo} ...")
    try:
        api.upload_file(
            path_or_fileobj=str(db_path),
            path_in_repo="tokens.db",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Upload pre-cached mol tokens (RDKit 3D conformers)",
        )
        print("上传完成！")
        print(f"数据集页面：https://www.modelscope.cn/datasets/{args.repo}")
    except AttributeError:
        # 旧版 SDK 用 push_file
        api.push_file(
            local_path=str(db_path),
            repo_id=args.repo,
            repo_type="dataset",
            filename_in_repo="tokens.db",
        )
        print("上传完成！")
        print(f"数据集页面：https://www.modelscope.cn/datasets/{args.repo}")


if __name__ == "__main__":
    main()
