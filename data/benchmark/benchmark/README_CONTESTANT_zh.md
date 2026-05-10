# 选手发布版说明（中文）

## 1. 测试集范围

- `DUD-E`：共 102 个任务，每个靶点 1 个任务。
- `LIT-PCBA`：共 15 个任务，每个靶点 1 个任务。
- 总任务数：117
- 总配体数：2092260

说明：

- `DUD-E` 任务来自对应靶点的测试配体集合。
- `LIT-PCBA` 任务使用该 benchmark 的保留测试划分。

## 2. 目录结构

- `manifest.jsonl`
  - 全部任务索引，每行一个 JSON 对象。
- `tasks/<task_id>/task.json`
  - 单个任务的公开元信息。
- `tasks/<task_id>/ligands.csv`
  - 该任务的待排序配体列表。
- `tasks/<task_id>/receptors/`
  - 受体结构文件。
- `tasks/<task_id>/refs/`
  - 参考共晶配体，可用于口袋定位、对齐或定义 docking box。

## 3. 两类任务的含义

### 3.1 DUD-E

- 每个任务只有 1 个 receptor。
- `receptors/receptor.pdb` 是受体结构。
- `refs/crystal_ligand.mol2` 是该结构对应的共晶配体。

### 3.2 LIT-PCBA

- 每个任务可能有多个 receptor 结构。
- `receptors/*.mol2` 是同一 target 的多个已知结构。
- `refs/*_ligand.mol2` 是各结构对应的共晶配体。
- 选手可以使用单结构或多结构方法，但最终每个 ligand 只能输出 1 个分数。

## 4. ligands.csv 字段说明

公开版 `ligands.csv` 只保留以下字段：

- `ligand_id`
  - 本测试集内部唯一编号，提交时必须使用。
- `smiles`
  - 配体 SMILES。

## 5. 选手提交格式

选手最终需要提交一个压缩包：

```text
result.zip
├── result.csv
└── result.log
```

其中 `result.csv` 为排序结果文件，格式如下：

```csv
task_id,ligand_id,score
dude_aa2ar,dude_aa2ar__L000001,12.53
dude_aa2ar,dude_aa2ar__L000002,1.42
litpcba_ADRB2,litpcba_ADRB2__L000001,3.88
```

其中 `result.log` 为运行日志，用于记录 agent 的主要自主优化过程，例如：

- 方案设计与迭代
- 训练或微调过程
- 推理、重排序与结构聚合过程
- 关键决策与中间结果
- 最终结果生成过程

要求：

- 每个 `(task_id, ligand_id)` 必须且只能出现一次。
- 所有任务必须覆盖完整。
- `score` 越高，排序越靠前。
- `result.zip` 中必须同时包含 `result.csv` 和 `result.log`

说明：

- 平台评测脚本按 `result.zip` 格式读取提交。
- 主办方会结合 `result.log` 分析选手的 agent 是否确实进行了自主优化。
- 如果日志显示方法并未真实进行自主优化，或存在违规使用测试集、直接恢复答案等行为，最终分数可能被置零。

## 6. 选手需要做什么

对每个任务：

1. 读取 `task.json`
2. 读取对应的 receptor 与参考配体
3. 对 `ligands.csv` 中的每个 ligand 打分
4. 将结果写入 `result.csv`，并将运行过程写入 `result.log`
5. 将二者打包为 `result.zip` 后提交

## 7. 建议

- 统一以 `ligand_id` 作为主键。
- 多结构任务建议在方法内部做聚合，但提交时必须压缩为每个 ligand 一个最终分数。
