# 20261027v2 主分析代码工程

本工程保存**已完成主分析当时归档的代码副本**，覆盖 CACox（历史代码名 CACox ES）及 RCS＋ridge-Cox。研究数据、插补对象、模型权重和结果没有复制到本工程；代码的运行仍依赖原始数据、原计算环境及历史路径配置。本工程是代码与溯源归档，不代表在此目录直接重新运行过模型。

## 已完成工作的范围

- 输入：`20261027v2.xlsx`，SHA-256 `7ed9af48f53e3fc4eebe8ff047bfc7b03a5f8fa654a0a96c758b79f640b8356e`；1140 人、29 个候选变量。
- 10 次重复五折外层验证，每折 20 份插补；CACox 与 RCS＋ridge-Cox 各有 50 折及 1000 个外层插补拟合／预测编号。
- 原完整归档：[主分析工程](C:/Users/NinoNeumann/CodexResults/main_analysis_20261027v2_10x5x20_20261108/README.md)。原工作簿位于该归档的 `results/cacox_main/input/20261027v2.xlsx`，本工程只记录路径和哈希。

## 代码位置

| 位置 | 内容 |
| --- | --- |
| `code/experiments/cox_es_rcs_newdata_10x5/` | 输入准备、MICE 与变量筛选、两模型执行及最终比较入口 |
| `code/experiments/cacox_*/` | CACox 神经网络、树与候选模型实现；具体用于正式运行的文件以源码映射表为准 |
| `code/experiments/rcs_ridge_10x5/` | RCS＋ridge-Cox 相关实现 |
| `code/upstream/` | 当时调用并归档的上游 R 数据处理、变量筛选和拟合模块 |
| `code/environment/` | 原 Python/GPU 依赖清单 |
| `docs/COPIED_SOURCE_INDEX.csv` | 每个复制文件的来源、大小和 SHA-256 |
| `docs/EXECUTED_SOURCE_MAP.csv` | 四份正式运行清单中的 553 条来源引用；41 条代码引用均按 SHA-256 定位到本工程副本，其余是输入或结果引用 |
| `docs/SOURCE_MAP_SUMMARY.json` | 四份正式运行清单的原路径及映射统计 |

## 原代码执行顺序

1. `prepare_input.R` 与 `export_input_excel.py` 建立类型化输入和固定患者级外层分折。
2. `run_data.py` 调用 `r_stage.R` 和 `export_bundle.R`：在外层开发数据内完成 MICE、Boruta＋LASSO 及 20 份数据导出；正式流程保留外层验证患者作预测评价。
3. `run_i20.py` 调用 CACox 网络与树实现，完成每折每插补的内部候选选择和外层训练。
4. `run_e_selected.py` 使用候选模型及冻结选择输出 CACox 折外预测。
5. `run_rcs_adaptive.py` 调用 `fit_fold_adaptive.R`、`ridge_functions_adaptive.R`，对相同折和插补进行 RCS＋ridge-Cox 拟合与预测。
6. `run_final.py`、`final_metrics.py`、`final_compare.py` 汇总两方法及配对比较。实际阶段、参数和恢复记录以原运行清单与完成标记为准；本页不是新启动命令。

代码目录中还保留与候选实现、调试及历史运行有关的文件。**不能仅凭目录存在就认定该文件参与正式模型**；请使用 `docs/EXECUTED_SOURCE_MAP.csv` 的阶段与哈希确认。

## 编号和边界

`repeat03_fold04` 是第 3 次重复、第 4 个外层折；`imputation_07`、`mi07`、`model_07.rds` 均指该折第 7 份插补。一个 MICE `imputation.rds` 保存该分区的 20 链对象，不是单份插补。详尽的文件级重复／折／插补定位仍在[原工程映射表](C:/Users/NinoNeumann/CodexResults/main_analysis_20261027v2_10x5x20_20261108/docs/FOLD_IMPUTATION_MAP.md)。

本工程未收录模型权重及患者级数据，因此无法单独对新患者产生预测，也不能替代完整模型交付。原工程约 24.4 GB 的数据和结果保持在原位置。本工程只复制代码，复制时每个文件均与来源进行 SHA-256 比对。
