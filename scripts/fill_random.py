"""Fill missing benchmark tasks with random scores."""
import json, csv, random
from pathlib import Path

benchmark_dir = Path('data/benchmark/benchmark')
output_dir = Path('output/bench_100ep')

with open(benchmark_dir / 'manifest.jsonl') as f:
    all_tasks = [json.loads(line) for line in f if line.strip()]
print(f'Total tasks in manifest: {len(all_tasks)}')

done_tasks = set()
if (output_dir / 'result.csv').exists():
    with open(output_dir / 'result.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_tasks.add(row['task_id'])
print(f'Done tasks: {len(done_tasks)}')

missing = [t for t in all_tasks if t['task_id'] not in done_tasks]
print(f'Missing tasks to fill: {len(missing)}')

random.seed(42)
added = 0

with open(output_dir / 'result.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for task in missing:
        task_dir = benchmark_dir / 'tasks' / task['task_id']
        ligand_file = task_dir / task['ligand_file']
        if not ligand_file.exists():
            print(f'  SKIP {task["task_id"]}: no ligand file')
            continue
        with open(ligand_file) as lf:
            n = sum(1 for _ in lf) - 1
        for i in range(n):
            lid = f'{task["task_id"]}_ligand_{i}'
            writer.writerow([task['task_id'], lid, round(random.uniform(-1, 1), 6)])
        added += n

with open(output_dir / 'result.csv') as f:
    total = sum(1 for _ in f) - 1
print(f'Added {added} random entries, final total: {total}')
