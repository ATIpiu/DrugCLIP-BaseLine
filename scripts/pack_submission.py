"""Create submission zip for benchmark output."""
import shutil, zipfile, sys
from pathlib import Path

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('output/bench_official')

rl = output_dir / 'result.log'
al = output_dir / 'agent.log'
if (not rl.exists() or rl.stat().st_size == 0) and al.exists():
    shutil.copyfile(al, rl)

z = output_dir / 'result.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in ['result.csv', 'result.log']:
        fp = output_dir / fn
        if fp.exists():
            zf.write(fp, fn)

mb = z.stat().st_size / 1024**2
print(f'{z} ({mb:.1f} MB)')
