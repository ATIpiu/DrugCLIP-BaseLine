# Contestant Release Notes

## 1. Benchmark Scope

- `DUD-E`: 102 tasks, one target per task
- `LIT-PCBA`: 15 tasks, one target per task
- Total tasks: 117
- Total ligands: 2092260

Notes:

- `DUD-E` tasks are built from the corresponding target-level test ligand sets.
- `LIT-PCBA` tasks use the held-out test split of that benchmark.

## 2. Directory Structure

- `manifest.jsonl`
  - Public task index, one JSON object per line.
- `tasks/<task_id>/task.json`
  - Public metadata for a single task.
- `tasks/<task_id>/ligands.csv`
  - Ligand list to be ranked for that task.
- `tasks/<task_id>/receptors/`
  - Receptor structure files.
- `tasks/<task_id>/refs/`
  - Reference co-crystal ligands that may be used for pocket localization, alignment, or docking-box definition.

## 3. Meaning of the Two Task Types

### 3.1 DUD-E

- Each task has exactly one receptor.
- `receptors/receptor.pdb` is the receptor structure.
- `refs/crystal_ligand.mol2` is the corresponding co-crystal ligand.

### 3.2 LIT-PCBA

- A task may contain multiple receptor structures.
- `receptors/*.mol2` are multiple known structures for the same target.
- `refs/*_ligand.mol2` are the corresponding co-crystal ligands.
- Contestants may use either a single-structure or multi-structure strategy, but must output exactly one final score per ligand.

## 4. `ligands.csv` Fields

The public `ligands.csv` contains only the following columns:

- `ligand_id`
  - The unique ligand identifier used within this benchmark. This must be used in the submission file.
- `smiles`
  - Ligand SMILES string.

## 5. Submission Format

Contestants must submit a zip package:

```text
result.zip
├── result.csv
└── result.log
```

`result.csv` is the ranking output file and must follow the format below:

```csv
task_id,ligand_id,score
dude_aa2ar,dude_aa2ar__L000001,12.53
dude_aa2ar,dude_aa2ar__L000002,1.42
litpcba_ADRB2,litpcba_ADRB2__L000001,3.88
```

`result.log` is the run log and should record the main autonomous optimization process of the agent, for example:

- idea generation and iteration
- training or fine-tuning steps
- inference, reranking, and structure aggregation
- key decisions and intermediate outcomes
- final result generation


Requirements:

- Each `(task_id, ligand_id)` pair must appear exactly once.
- All tasks must be fully covered.
- Higher `score` means a higher ranking position.
- `result.zip` must contain both `result.csv` and `result.log`.

Notes:

- The platform evaluator reads submissions in the `result.zip` format.
- The organizer will inspect `result.log` to determine whether the submitted agent genuinely performed autonomous optimization.
- If the log indicates that the agent did not genuinely perform autonomous optimization, or that the submission used prohibited shortcuts such as test-set misuse or direct answer recovery, the final score may be set to zero.

## 6. What Contestants Need to Do

1. Read `task.json`
2. Load the corresponding receptor structures and reference ligands
3. Score every ligand in `ligands.csv`
4. Write the scores to `result.csv` and record the run process in `result.log`
5. Package both files into `result.zip` for submission

## 7. Recommendations

- Use `ligand_id` as the primary key throughout your pipeline.
- For multi-receptor tasks, receptor-level aggregation is allowed internally, but the final submission must contain one final score per ligand.
