"""Prepare typed, source-checked tables for the CACox (ES) comparison workbook."""
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103'
FINAL=ROOT/'results/rcs_ridge_10x5/20261102_v1/final'
NAME={'E_selected':'cacox（es）','rcs_ridge_cox':'RCS＋ridge-Cox'}
KEY=['repeat_id','fold','imputation']

def read(name):return pd.read_csv(BASE/name)
def plain(frame):
    return [[None if pd.isna(v) else (v.item() if hasattr(v,'item') else v)
             for v in row] for row in frame.itertuples(index=False,name=None)]

fc=read('fold_pooled_cindex.csv')
fa=read('auc_fold_pooled.csv')
fb=read('brier_fold_pooled.csv')
fi=read('ibs_3point_fold_pooled.csv')
fold=fc[['repeat_id','fold','imputations','C_E_selected','C_rcs_ridge_cox','difference_E_minus_RCS',
    'E_CI95_lower','E_CI95_upper','RCS_CI95_lower','RCS_CI95_upper',
    'difference_CI95_lower','difference_CI95_upper','fold_Rubin_p']].copy()
for t in (365,730,1095):
    z=fa[fa.horizon_days.eq(t)][['repeat_id','fold','AUC_E_selected','AUC_rcs_ridge_cox','difference_E_minus_RCS']]
    z=z.rename(columns={c:f'{c}_{t}d' for c in z.columns if c not in ('repeat_id','fold')})
    fold=fold.merge(z,on=['repeat_id','fold'],validate='one_to_one')
    q=fb[fb.horizon_days.eq(t)].pivot(index=['repeat_id','fold'],columns='model',values='Brier_IPCW').reset_index()
    q=q.rename(columns={m:f'Brier_{m}_{t}d' for m in NAME})
    fold=fold.merge(q,on=['repeat_id','fold'],validate='one_to_one')
z=fi.pivot(index=['repeat_id','fold'],columns='model',values='IBS_365_1095_3point_approx').reset_index()
z=z.rename(columns={m:f'IBS_approx_{m}' for m in NAME})
fold=fold.merge(z,on=['repeat_id','fold'],validate='one_to_one').sort_values(['repeat_id','fold'])
assert len(fold)==50 and fold.imputations.eq(20).all()

mc=pd.read_csv(FINAL/'model_cindex.csv')
mc=mc[mc.model.isin(NAME)].pivot(index=KEY,columns='model',values='C_index').reset_index()
mc=mc.rename(columns={m:f'C_{m}' for m in NAME})
ma=read('auc_fold_imputation.csv').pivot(index=KEY,columns=['model','horizon_days'],values='AUC')
ma.columns=[f'AUC_{m}_{t}d' for m,t in ma.columns]
mb=read('brier_fold_imputation.csv').pivot(index=KEY,columns=['model','horizon_days'],values='Brier_IPCW')
mb.columns=[f'Brier_{m}_{t}d' for m,t in mb.columns]
mi=read('ibs_3point_fold_imputation.csv').pivot(index=KEY,columns='model',values='IBS_365_1095_3point_approx')
mi.columns=[f'IBS_approx_{m}' for m in mi.columns]
unit=mc.merge(ma.reset_index(),on=KEY,validate='one_to_one').merge(mb.reset_index(),on=KEY,validate='one_to_one').merge(mi.reset_index(),on=KEY,validate='one_to_one')
unit=unit.sort_values(KEY)
assert len(unit)==1000 and not unit[KEY].duplicated().any()

by_mi=pd.read_csv(FINAL/'metrics_by_imputation.csv')
by_mi=by_mi[by_mi.model.isin(NAME)].pivot(index='imputation',columns=['model','metric'],values='estimate')
by_mi.columns=[f'{metric}_{model}' for model,metric in by_mi.columns]
by_mi=by_mi.reset_index().sort_values('imputation')
assert len(by_mi)==20

dca=read('dca_overall.csv')
dca=dca.pivot(index=['horizon_days','threshold'],columns='model',values='net_benefit').reset_index()
dca=dca.rename(columns={m:f'net_benefit_{m}' for m in NAME})
assert len(dca)==90
cal=read('survival_calibration.csv')
cal.model=cal.model.map(NAME)
strata=read('risk_strata_summary.csv')
strata.model=strata.model.map(NAME)
roc=read('roc_pooled_points.csv').pivot(index=['horizon_days','FPR'],columns='model',values='TPR').reset_index()
roc=roc.rename(columns={m:f'TPR_{m}' for m in NAME})
assert len(roc)==603

overall_c=read('overall_comparison.csv')
overall_auc=read('auc_overall.csv')
overall_brier=read('brier_overall.csv')
overall_ibs=read('ibs_3point_overall.csv')
summary=[]
for model in NAME:
    z=overall_c[overall_c.model.eq(model)].iloc[0]
    summary.append(['C-index',NAME[model],None,float(z.estimate),float(z.CI95_lower),float(z.CI95_upper)])
for t in (365,730,1095):
    for model in NAME:
        z=overall_auc[overall_auc.horizon_days.eq(t)&overall_auc.model.eq(model)].iloc[0]
        summary.append(['AUC',NAME[model],t,float(z.AUC),float(z.CI95_lower),float(z.CI95_upper)])
        b=overall_brier[overall_brier.horizon_days.eq(t)&overall_brier.model.eq(model)].iloc[0]
        summary.append(['IPCW Brier',NAME[model],t,float(b.Brier_IPCW),None,None])
for model in NAME:
    z=overall_ibs[overall_ibs.model.eq(model)].iloc[0]
    summary.append(['近似 IBS 365–1095 天',NAME[model],None,float(z.IBS_365_1095_3point_approx),None,None])
z=overall_c[overall_c.model.eq('E_minus_RCS')].iloc[0]
comparison=[['C-index 差值',None,float(z.estimate),float(z.CI95_lower),float(z.CI95_upper),float(z.p_Rubin)]]
for t in (365,730,1095):
    q=overall_auc[overall_auc.horizon_days.eq(t)&overall_auc.model.eq('E_selected_minus_RCS')].iloc[0]
    comparison.append(['AUC 差值',t,float(q.AUC),float(q.CI95_lower),float(q.CI95_upper),None])

payload={
 'summary':summary,'comparison':comparison,
 'fold':{'columns':list(fold.columns),'rows':plain(fold)},
 'unit':{'columns':list(unit.columns),'rows':plain(unit)},
 'imputation':{'columns':list(by_mi.columns),'rows':plain(by_mi)},
 'dca':{'columns':list(dca.columns),'rows':plain(dca)},
 'calibration':{'columns':list(cal.columns),'rows':plain(cal)},
 'strata':{'columns':list(strata.columns),'rows':plain(strata)},
 'roc':{'columns':list(roc.columns),'rows':plain(roc)},
 'source':str(BASE),
}
target=BASE/'es_workbook_data.json'
target.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf8')
print(target)
print({k:len(payload[k]['rows']) for k in ('fold','unit','imputation','dca','calibration','strata','roc')})
