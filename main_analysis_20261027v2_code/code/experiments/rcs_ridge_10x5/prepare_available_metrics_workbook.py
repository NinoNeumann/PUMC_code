"""Stage verified, directly available paired metrics for an Excel deliverable."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103'
FINAL=ROOT/'results/rcs_ridge_10x5/20261102_v1/final'
LABEL={'E_selected':'cacox（es）','rcs_ridge_cox':'RCS＋ridge-Cox'}

def matrix(frame):
    frame=frame.copy()
    for col in frame.select_dtypes(include='object').columns:
        frame[col]=frame[col].replace(LABEL)
    return {'columns':list(frame.columns),
            'rows':[[None if pd.isna(v) else (v.item() if hasattr(v,'item') else v)
                     for v in row] for row in frame.itertuples(index=False,name=None)]}

old=json.loads((BASE/'es_workbook_data.json').read_text(encoding='utf8'))
summary=[row for row in old['summary'] if 'IBS' not in row[0]]
comparison=[row.copy() for row in old['comparison']]
paired=pd.read_csv(FINAL/'paired_differences.csv')
for row in comparison:
    if row[0]!='AUC 差值':
        continue
    horizon=int(row[1])
    match=paired.loc[paired.comparison.eq('rcs_ridge_cox_minus_E_selected') &
                     paired.metric.eq(f'AUC_{horizon}d')]
    assert len(match)==1
    q=match.iloc[0]
    assert abs(row[2]+q.estimate)<1e-10
    assert abs(row[3]+q.upper)<1e-10 and abs(row[4]+q.lower)<1e-10
    row[5]=float(2*student_t.sf(abs(q.estimate/q.SE),q.df))
fold=pd.DataFrame(old['fold']['rows'],columns=old['fold']['columns'])
fold=fold.loc[:,~fold.columns.str.startswith('IBS_approx')]
unit=pd.DataFrame(old['unit']['rows'],columns=old['unit']['columns'])
unit=unit.loc[:,~unit.columns.str.startswith('IBS_approx')]
mi=pd.DataFrame(old['imputation']['rows'],columns=old['imputation']['columns'])
roc=pd.read_csv(BASE/'roc_fold_pooled_points.csv')
dca=pd.read_csv(BASE/'dca_fold_pooled.csv')
cal=pd.read_csv(BASE/'survival_calibration.csv')
roc_overall=pd.read_csv(BASE/'roc_pooled_points.csv')
dca_overall=pd.read_csv(BASE/'dca_overall.csv')
auc_detail=pd.read_csv(BASE/'auc_fold_pooled.csv')
assert len(fold)==50 and len(unit)==1000 and len(mi)==20
assert len(roc)==50*2*3*201 and len(dca)==50*2*3*30
assert len(cal)==2*3*5 and len(auc_detail)==50*3
assert not roc[['repeat_id','fold','model','horizon_days','FPR']].duplicated().any()
assert not dca[['repeat_id','fold','model','horizon_days','threshold']].duplicated().any()
assert set(roc.model)==set(dca.model)==set(cal.model)==set(LABEL)
assert set(roc.horizon_days)==set(dca.horizon_days)=={365,730,1095}
payload={
    'summary':summary,'comparison':comparison,
    'fold':matrix(fold),'unit':matrix(unit),'imputation':matrix(mi),
    'auc_detail':matrix(auc_detail),
    'roc_fold':matrix(roc),'dca_fold':matrix(dca),
    'roc_overall':matrix(roc_overall),'dca_overall':matrix(dca_overall),
    'calibration_overall':matrix(cal),
    'source':str(BASE),
    'scope':{'repeats':10,'folds':50,'imputations_per_fold':20,
             'roc_fold_rows':len(roc),'dca_fold_rows':len(dca),
             'calibration_scope':'overall only, not 50 fold-specific curves',
             'ibs_status':'0-1095d IBS not yet validated; excluded'},
}
dest=BASE/'available_metrics_workbook_data.json'
dest.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf8')
print(dest)
print({k:len(v['rows']) for k,v in payload.items() if isinstance(v,dict) and 'rows' in v})
