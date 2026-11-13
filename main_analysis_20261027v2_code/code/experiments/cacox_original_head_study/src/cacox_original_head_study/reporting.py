"""Original CACox head-change effects and the fixed matched RCS target."""
import json
import numpy as np
import pandas as pd
from .evaluation import evaluate,individual_cindices,paired_difference,assert_complete,assert_paired
from .probability import probability_checks
from .provenance import verify_inputs
from .runtime import stamp
from .storage import write_json,mark_complete,verify_completed

ROLE="D0_original_logistic_hazard"
LABELS={"R0_rcs_ridge":"RCS＋ridge-Cox基线","O0_original_cox":"原始CACox／原Cox头",ROLE:"原始CACox／新离散头"}


def figures(study,individual,paired,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,"pdf.fonttype":42})
    directory=study.out/"figures";directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5));order=list(LABELS)
    axes[0].boxplot([individual[individual.model_id.eq(m)].C_index for m in order],positions=[1,2,3],showfliers=False)
    for pos,model in enumerate(order,1):
        z=individual[individual.model_id.eq(model)]
        axes[0].scatter(pos+np.random.default_rng(29).uniform(-.12,.12,len(z)),z.C_index,s=12,alpha=.6)
    axes[0].set(xticks=[1,2,3],xticklabels=[LABELS[m].replace("／","\n") for m in order],ylabel="外层测试C-index",title="各组100个模型")
    for reference,offset,color in [("O0_original_cox",-.10,"#2166ac"),("R0_rcs_ridge",.10,"#b2182b")]:
        q=paired[paired.reference_id.eq(reference)]
        for fold in range(1,6):
            z=q[q.fold.eq(fold)]
            axes[1].scatter(fold+offset+np.linspace(-.05,.05,len(z)),z.delta,s=14,alpha=.65,color=color,
                            label=f"新头减{LABELS[reference]}" if fold==1 else None)
    axes[1].axhline(0,color="#777777");axes[1].axhline(.02,color="#777777",ls="--")
    axes[1].set(xticks=range(1,6),xlabel="外层折",ylabel="配对C-index差",title="正值表示新头较高")
    axes[1].legend(fontsize=8)
    for axis in axes:axis.grid(axis="y",alpha=.18)
    fig.tight_layout()
    for suffix in ["png","pdf"]:fig.savefig(directory/f"cindex_distribution.{suffix}",dpi=190,bbox_inches="tight")
    plt.close(fig)


def report(study,frames):
    out=study.out/"comparisons";out.mkdir(exist_ok=True)
    metrics=[];individual=[];folds=[];diffs=[];paired=[]
    for role,frame in frames.items():
        assert_complete(frame);assert_paired(frames["R0_rcs_ridge"],frame)
        per,pool=evaluate(frame);per.insert(0,"model_id",role);per.to_csv(out/f"metrics_each_imputation_{role}.csv",index=False)
        pool.insert(0,"model_id",role);metrics.append(pool)
        each=individual_cindices(frame);each.insert(0,"model_id",role);individual.append(each)
        for fold,z in frame.groupby("fold"):
            _,p=evaluate(z,times=());c=p.iloc[0]
            folds.append(dict(model_id=role,fold=int(fold),C_index=float(c.estimate),lower=float(c.lower),upper=float(c.upper)))
    metrics=pd.concat(metrics,ignore_index=True);individual=pd.concat(individual,ignore_index=True)
    for reference in ["O0_original_cox","R0_rcs_ridge"]:
        per,d=paired_difference(frames[reference],frames[ROLE]);d.insert(0,"reference_id",reference);d.insert(1,"alternative_id",ROLE);diffs.append(d)
        per.to_csv(out/f"paired_each_imputation_vs_{reference}.csv",index=False)
        a=individual[individual.model_id.eq(reference)].drop(columns="model_id")
        b=individual[individual.model_id.eq(ROLE)].drop(columns="model_id")
        q=b.merge(a,on=["fold","imputation"],suffixes=("_new","_reference"),validate="one_to_one")
        q["delta"]=q.C_index_new-q.C_index_reference;q["reference_id"]=reference;paired.append(q)
    diffs=pd.concat(diffs,ignore_index=True);paired=pd.concat(paired,ignore_index=True)
    metrics.to_csv(out/"metrics_pooled.csv",index=False);individual.to_csv(out/"model_cindex.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"fold_cindex.csv",index=False);diffs.to_csv(out/"paired_differences.csv",index=False);paired.to_csv(out/"paired_model_cindex.csv",index=False)
    probability=probability_checks(study,frames);probability.to_csv(out/"probability_errors_each_model.csv",index=False)
    probability.groupby(["model_id","horizon_days"])[["Brier_IPCW","predicted_mean_risk","observed_KM_risk","calibration_mean_gap"]].mean().reset_index().to_csv(out/"probability_errors_summary.csv",index=False)
    original=diffs[diffs.reference_id.eq("O0_original_cox")&diffs.metric.eq("C_index")].iloc[0]
    rcs=diffs[diffs.reference_id.eq("R0_rcs_ridge")&diffs.metric.eq("C_index")].iloc[0]
    z=paired[paired.reference_id.eq("R0_rcs_ridge")];target=study.config["target"]
    hits=int(z.delta.ge(target["paired_gain_min"]).sum())
    summary=dict(status="complete",completed_at=stamp(),architecture="original_CrossAttentionCox",spline_branch=False,
                 search_enabled=False,completed_folds=5,skipped_folds=0,new_inner_fits=500,new_outer_models=100,reused_original_models=100,
                 delta_vs_original=float(original.estimate),delta_vs_original_CI95=[float(original.lower),float(original.upper)],
                 head_improved_point=bool(original.estimate>0),head_improved_statistical=bool(original.lower>0),
                 delta_vs_rcs=float(rcs.estimate),delta_vs_rcs_CI95=[float(rcs.lower),float(rcs.upper)],
                 paired_models_gain_ge_002=hits,paired_models_positive_vs_rcs=int(z.delta.gt(0).sum()),
                 target_met=bool(rcs.estimate>=target["pooled_gain_min"] and hits>=target["paired_models_min"]),models={})
    for role in LABELS:
        row=metrics[metrics.model_id.eq(role)&metrics.metric.eq("C_index")].iloc[0];m=individual[individual.model_id.eq(role)].C_index
        summary["models"][role]=dict(C_index=float(row.estimate),CI95=[float(row.lower),float(row.upper)],
                                     mean_of_100_models=float(m.mean()),range_of_100_models=[float(m.min()),float(m.max())])
    write_json(out/"summary.json",summary);figures(study,individual,paired,summary)
    lines=["# 原始CACox更换生存预测头：固定参数五折实验","",f"完成时间：{stamp()}。","",
        "原始CrossAttentionCox骨干，未添加样条主分支。固定学习率7e-6、Dropout0.15、权重衰减0.01、隐藏96、Transformer两层四头、CrossNet两层，上限3000轮、150轮早停。无超参数搜索。","",
        "原Cox头及RCS基线为通过核验的只读历史模型；新头仅将末层Linear(96,1)换为Linear(96,6)，配套离散删失似然，偏置仅由训练分区初始事件比例设置。其余骨干及初始权重保持一致。","",
        "| 方法 | 五折汇总C-index（95%CI） | 100模型均值（最小～最大） |","|---|---|---|"]
    for role,label in LABELS.items():
        q=summary["models"][role];lines.append(f"| {label} | {q['C_index']:.5f}（{q['CI95'][0]:.5f}～{q['CI95'][1]:.5f}） | {q['mean_of_100_models']:.5f}（{q['range_of_100_models'][0]:.5f}～{q['range_of_100_models'][1]:.5f}） |")
    lines += ["",f"新头相对原始Cox头：{original.estimate:+.5f}（95%CI {original.lower:.5f}～{original.upper:.5f}）。点估计改善：{summary['head_improved_point']}；配对区间下限大于0：{summary['head_improved_statistical']}。","",
              f"新头相对RCS：{rcs.estimate:+.5f}（95%CI {rcs.lower:.5f}～{rcs.upper:.5f}）。提升≥0.02的配对模型{hits}/100；严格双重目标{'达到' if summary['target_met'] else '未达到'}。","",
              "数据为同一1140人、29候选变量、同一5折×20份R MICE插补，Boruta＋LASSO名单为15/9/8/9/8项。筛选名单来自外层开发集，内部未重新筛选。500次内部拟合仅确定新头各份训练轮数；全部冻结后才生成100个新头外层预测。","",
              "原Cox C-index使用log风险，新头使用固定1095日负RMST；AUC均使用1-S(t)。完整AUC及95%CI、IPCW Brier和校准误差见comparisons。原始Cox保留原损失，离散头采用其对应似然，评价的是完整的生存头替换方案。","",
              "当前队列已多轮用于开发，属于继续开发；区间未校正多轮选型乐观偏差。100个插补模型不独立。不得将原始CACox头的结果与废弃v2混称。","",
              "每次目的、时间、有效参数、结果路径和解释见EXPERIMENTS.md。","","![C-index分布](figures/cindex_distribution.png)",""]
    (study.out/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    manifest=json.loads((study.out/"manifest.json").read_text())
    if verify_inputs(study)!=manifest["inputs"] or study.code_hashes()!=manifest["source_sha256"]:raise ValueError("Source/input changed")
    candidate=study.out/"candidates"/ROLE;verified_inner=0;verified_outer=0
    for fold in range(1,6):
        inner=candidate/"inner"/ROLE/f"fold{fold:02d}/full_m20"
        if not verify_completed(inner):raise ValueError("Incomplete inner group")
        verified_inner+=len(pd.read_csv(inner/"metrics.csv"))
        outer=candidate/"outer"/f"fold{fold:02d}"
        if not verify_completed(outer):raise ValueError("Incomplete outer fold")
        for j in range(1,21):
            if not verify_completed(outer/f"imputation{j:02d}"):raise ValueError("Incomplete model")
            verified_outer+=1
    if verified_inner!=500 or verified_outer!=100:raise ValueError("Wrong fit count")
    write_json(study.out/"validation/FINAL_INTEGRITY.json",dict(status="passed",verified_inner_fits=500,verified_new_outer_models=100,verified_reference_original_models=100,source_and_inputs_unchanged=True,at=stamp()))
    study.log_experiment("最终配对汇总",stamp(),"判断原始CACox更换生存头是否有效",study.config["candidate"],out,
                         f"新头减原头={original.estimate:+.5f}，新头减RCS={rcs.estimate:+.5f}，严格目标={summary['target_met']}")
    mark_complete(out,["summary.json","metrics_pooled.csv","model_cindex.csv","fold_cindex.csv","paired_differences.csv","paired_model_cindex.csv","probability_errors_summary.csv"])
    mark_complete(study.out,["comparisons/COMPLETE.json","FINAL_REPORT.md","figures/cindex_distribution.png","validation/FINAL_INTEGRITY.json"])
    return summary
