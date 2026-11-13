# 20261027v2 敏感性与亚组分析代码工程

本工程保存正式 v4 五分支 CACox（ES）敏感性分析的**冻结科学代码**、历史运行调度代码，以及后续 MPA/GPA 和肾脏受累亚组评价代码。原始工作簿、类型化患者数据、插补对象、模型权重、预测和结果没有复制。代码与实际数据、环境、运行状态分离，不能从本目录直接重放完整训练。

## 分析范围

| 分析 | 人群或比较 | 是否重新训练 |
| --- | --- | --- |
| 五分支敏感性分析 | `complete_case`、`egfr_observed`、`renal_observed`、`known_subtype`、`no_egpa`；各 10 次×5 折 | 是，按限制人群独立重新插补／筛选／拟合；完整病例无插补 |
| MPA/GPA 亚组 | 原主分析总体模型在原始 Type 明确为 MPA、GPA 患者的折外预测 | 否 |
| 肾脏受累亚组 | 原主分析总体模型在原始 `Renal_Involvement=yes/no` 患者的折外预测 | 否 |

`renal_observed` 是“肾脏受累状态有记录”的**重训分支**；`yes/no` 是原总体模型的**亚组评价**，两者不能互换。`known_subtype` 包含 Type 已知的 MPA、GPA、EGPA，单独的 MPA/GPA 对比也不能用它替代。

## 目录与源码层次

| 位置 | 用途 |
| --- | --- |
| `frozen_package/code/`、`frozen_package/vendor/` | 正式 v4 冻结的数据处理、训练、评价与上游科学模块 |
| `frozen_package/run.py`、`.cmd`、`requirements.txt` | 原可移动包的启动与环境说明；需要原包的 `input/` 和运行环境 |
| `operational_history/experiments/cacox_sensitivity_fast_v1/` | 当时双批、机会式补折、并行模型调度、版本恢复与审计脚本，以及后续独立亚组脚本；其中 benchmark、早期版本和恢复脚本均非默认正式入口 |
| `docs/FROZEN_SOURCE_VERIFICATION.csv` | 从原 `PACKAGE_MANIFEST.json` 抽取并逐项验证的 44 个冻结代码／启动文件哈希 |
| `docs/COPIED_SOURCE_INDEX.csv` | 全部复制文件的来源、大小与 SHA-256 |
| `docs/SOURCE_MAP_SUMMARY.json` | 来源包、文件数和工作簿哈希 |

## 实际流程与入口

1. 原包 `run.py` 和 Windows 启动文件负责环境检查、输入核验、启动和状态读取。正式科学模块在 `frozen_package/code/` 与 `vendor/`；`input/20261027v2.xlsx` 的 SHA-256 是 `7ed9af48f53e3fc4eebe8ff047bfc7b03a5f8fa654a0a96c758b79f640b8356e`。
2. 实际 v4 运行使用 `two_batch_controller.py` 将五分支共 250 个数据折分批准备；`fast_data.py`、`mice_worker.R`、`select_worker.R` 与冻结模块完成各折数据准备、质控和筛选。机会式补折由 `opportunistic_queue_v4.py` 等脚本协调，完成折经核验后供正式流程复用。
3. 模型阶段沿用冻结 `cacox_fold.py` 等科学代码；`overlap_model_during_data.py`、`parallel_model_after_data.py` 仅调整调度和资源并发。正式完成是各分支每折的完成标记与最终汇总，不以旧失败状态文件代替。
4. `aggregate_sensitivity.R` 汇总五分支，并使用 `evaluation.R` 的折内可比较对、患者聚类影响量和 Rubin 合并计算比较。`subgroup_evaluation.R` 评价原总体模型的 MPA/GPA；后续 `evaluate_known_subtype_mpa_gpa.R` 形成独立对比；`evaluate_main_renal_subgroups.R` 评价原总体模型在肾脏受累 yes/no 两组中的 C-index 与组间差。

正式 v4 的适应性 MICE 记录为三个 80 轮例外和一个 70 轮例外，均从原有状态继续，其他分区为 60 轮；版本及恢复脚本在 `operational_history/`，但**不作为从头运行的新流程**。详细例外与完成结果以[原正式包](C:/Users/NinoNeumann/CodexResults/cacox_sensitivity_portable_adaptive_20261108_v4/README_中文.txt)和原运行状态为准。

## 输入、结果和执行边界

原正式包位于 `C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive_20261108_v4`；正式结果根为其 `run_fast_20261106_v1/`。原主分析冻结折外预测位于该包 `input/main_oof_cacox.csv.gz`，SHA-256 `23da9bb82ad4f0e5001d3c6ecbf2e252c037eb1f0cffba46eb856e4e012c1d55`。肾脏受累亚组的结果和核验另见[后续输出](E:/CodexPorjects/PUMC-rebuttal/outputs/cacox_renal_involvement_subgroup_20261112/README.md)。

本目录只保存代码。所有复制文件已按来源 SHA-256 核对；`frozen_package/` 的 44 个代码／启动文件还通过正式包清单复核。外部数据路径和本机 Python、R、CUDA 依赖仍需在获授权的环境中配置；不要直接把本代码工程当作包含研究数据与权重的可运行完整包。
