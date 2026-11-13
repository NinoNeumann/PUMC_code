# 20261027v2 新数据：CACox (ES) 与 RCS + ridge-Cox

## 输入与隔离

- 源工作簿：用户提供的 `20261027v2.xlsx`；逐字节副本保存在 `results/cox_es_rcs_newdata_10x5/20261104_v1/input/`。
- 1140 名患者，结局列 `death`、`survialday` 分别对应 `Death`、`Survival_Days`；`Antibody1` 对应原建模字段 `Antibody`。转换后的 `raw_typed.rds` 保留患者原始行序与 ID。
- 与此前输入相比，仅 53 人的抗体值发生变化；因此第一轮五折也重新计算，不复用任何旧数据训练的模型或插补。
- 旧结果目录只读。新数据/CACox 输出在 `results/cox_es_rcs_newdata_10x5/20261104_v1/`，RCS 输出在 `results/rcs_ridge_newdata_10x5/20261104_v1/`。

## 固定流程

1. 10 次重复、每次外层 5 折；每个外层开发集与其五个内部训练子集分别执行 20 份、60 轮 R MICE。预测变量插补不使用结局、患者 ID 或外层验证信息。
2. 每个外层开发集的 20 份插补各自执行 Boruta 非拒绝变量与 Cox-LASSO `lambda.min` 的并集，入选至少 10/20 次才作为该折名单。名单与插补诊断保存在对应 `repeatXX_foldYY/`。
3. 导出逐折原始开发/外层验证集及每份内部、外部插补表到 `bundles/`，并保留全部 `prepared.rds` 和 `imputation.rds`。
4. I20 内部选型在全部 20 份插补与五个内折完成后冻结；CACox (ES) 的 17 个候选及最终选择也只根据内部验证。模型权重、树对象、训练轨迹与外层预测逐单元保存。
5. RCS + ridge-Cox 使用同一折的同一变量名单和插补，内部选择 ridge 惩罚，再对外层五折分别拟合；保存每份外层拟合对象和预测。
6. 两组各有 50 × 20 = 1000 个新的外层模型；最终比较对齐相同患者、折次和插补编号，写入 `comparison/`。

## 运行与监控

- `run_data.py run`、`run_i20.py run` 并行启动；`run_e_selected.py run` 等待 I20 完成，`run.py` 等待全部数据准备完成。`final_compare.py` 等待两组完整结束。
- 数据阶段同时最多 4 个插补 / 6 个筛选 R 作业；I20 最多 3 个 GPU worker；RCS 同时最多 2 折，CACox (ES) 同时最多 2 个 GPU worker。各阶段保留资源门槛、独立状态文件和日志。
- 现有运行的源码、输入和配置哈希写入各阶段 manifest。`COMPLETE.json` 及逐单元校验文件是完成依据；运行状态或外层成绩不能用于改变方案。
