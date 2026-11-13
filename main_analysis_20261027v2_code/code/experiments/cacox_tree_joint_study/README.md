# CACox＋tree：内部联合选择实验

2026-10-30授权。达标条件仅为汇总外层C-index比普通Cox高至少0.02：目标≥0.7590991164；不再要求80/100个体模型分别提升0.02。

同一1140人、29候选变量、Boruta＋LASSO名单15/9/8/9/8和五折×20份既有R MICE。普通Cox0.7390991164、旧等权CACox＋tree0.7416577027只读对照。CACox架构、lr7e-6、dropout.15、WD.01、max3000、patience150、种子保持原值，不搜索神经网络超参数；不使用v2或RCS风险分支。

## 本轮主要改变

原树按单独C-index选树，再固定50%组合。本轮按组合后的内部C-index联合选择树与权重。每折每方案在5个内折×20份插补共100个验证单元上平均，选择一个共同配置，而非每份插补各自调权重。

- J1_probability730：730天生存概率组合，风险=1−组合生存率。
- J2_probability1095：1095天生存概率组合，风险=1−组合生存率。
- J3_standardized_logrisk：两个模型log风险均由当前训练分区均值/标准差标准化，再加权；组合生存曲线仅从训练分区重新估计Breslow基线。

预先固定树深1/2/3，树数75/150/300/600；树learning_rate=.03、subsample=.8、min_leaf=25。CACox权重仅在.1/.25/.5/.75/.9中选择，树权重为1−它。每种模型均保留非零两分支。并列取树少、浅、权重更接近.5的方案。连续变量输入和类别编码与原树一致。

## 实施顺序

1. prepare：哈希固定输入、参考及所有复用Python源；原始CACox完整内部逐轮复现；树max600的前300棵与单独max300一致；自实现Breslow与树官方生存概率一致；旧树首单元内部C一致。
2. 5进程，每任务一个外折×插补；500次原始CACox内部训练，并拟合1500个深度不同的最大600树模型，用前缀获取6000个树配置预测。所有预测、训练曲线和联合候选成绩存档。
3. 全500内部单元完成后冻结三个方案全部五折的配置；之后才读取外层测试预测并训练最终树。
4. 同一100个外层单元，生成三个方案各100模型的预测、树模型和组合元数据；原始CACox外层权重只读复用，不重训或换参数。
5. 输出FINAL_REPORT.md、metrics_pooled.csv（C-index及1/2/3年AUC和95%CI）、paired_differences.csv、model_cindex.csv与cindex_distribution.png。比较新方案与普通Cox、旧等权组合。

输入与源码被manifest锁定。每个内部单元和外层单元都有COMPLETE.json输出哈希；只复用通过校验的完整单元，不覆盖已完成数据。异常保留现场，不手动删锁。RUN_STATUS.json记录真实控制器PID、实际worker PIDs、内部/外层完成数量；各单元RUN_STATUS.json记录训练轮数与树深度。

本轮仅评价以上三个预注册方案。不能根据外层结果改变权重网格、阈值、结构、划分、筛选或插补，也不自动追加新方案。内部筛选名单固定于外层开发集，未完全嵌套；本队列已多轮开发，外层比较仍属开发性证据，100插补模型相关，区间未校正多轮比较的乐观偏差。不保证能够达到0.7591。

来源：scikit-survival GradientBoostingSurvivalAnalysis，https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html 。每阶段拟合Cox部分似然负梯度；组合策略为本项目固定实验方案，不声称为文献保证的改进。

启动：项目根 `.venv-gpu/Scripts/python.exe experiments/cacox_tree_joint_study/runner.py prepare`，通过后同入口`launch`。完整实验目的、时间、参数和结果位置记入输出EXPERIMENTS.md。
