"""Exercise reporting plumbing using saved internal preflight states;zero fits."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('check_rankobjective_formal_pipeline',HERE/'runner.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
def main():
    s.b.configure();out=s.OUT/'pipeline_check'
    if out.exists():raise RuntimeError('Preserve previous pipeline check')
    out.mkdir();g=s.b.io.load_group(1,1);pred={};rows=[]
    for cid in s.ORDER:
        source=s.b.source_dir(1,1,cid.split('_')[1]) if cid.startswith('control_') else s.OUT/'preflight'/('f1_'+cid)
        assert s.m.verify_completed(source)
        with np.load(source/'predictions.npz') as z:pred[cid]=(z['train_members'].mean(2),z['val_members'].mean(2))
    grid=pd.DataFrame(s.grid(g,pred,1,1));assert len(grid)==240
    t=g['val_time'];e=g['val_event'].astype(bool);i,j=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
    for cid,(a,v) in pred.items():
        for k in range(1,21):
            for kind in ['NN','I20']:
                aa,vv=(a[k-1],v[k-1]) if kind=='NN' else s.blend(g,a[k-1],v[k-1],1,k)[:2]
                c=float(((vv[i]>vv[j])+.5*(vv[i]==vv[j])).mean());row=grid[grid.candidate.eq(cid)&grid.kind.eq(kind)&grid.imputation.eq(k)]
                assert len(row)==1 and abs(c-float(row.C_index.iloc[0]))<1e-12
        base=s.b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],a[0]);q=s.frame(g,1,1,cid+'_NN',v[0],base)
        assert len(q)==len(g['val_ids']) and np.isfinite(q.score).all();np.testing.assert_array_equal(q.score,v[0]);rows.append(dict(candidate=cid,frame_C_error=0.))
    for objective in ['H','P']:
        x=pd.read_csv(s.OUT/f'preflight/f1_{objective}_NLL/training_trace.csv',float_precision='round_trip');y=pd.read_csv(s.OUT/f'preflight/f1_{objective}_C/training_trace.csv',float_precision='round_trip');n=min(len(x),len(y))
        np.testing.assert_array_equal(x.iloc[:n][['training_objective','gradient_norm_before_clip']],y.iloc[:n][['training_objective','gradient_norm_before_clip']])
    # One-group grid tests the real MultiIndex lookup,not a fake fivefold choice.
    means=grid.groupby(['candidate','kind','inner_fold']).C_index.mean().groupby(['candidate','kind']).mean()
    for kind in ['NN','I20']:
        values={cid:float(means.loc[(cid,kind)]) for cid in s.ORDER};top=max(values.values());chosen=next(cid for cid in s.ORDER if values[cid]>=top-1e-12)
        fake={'choices':{'1':{'S_'+kind:dict(candidate=chosen)}}};assert s.role_choice('S_'+kind,1,fake)==(chosen,kind)
    record=dict(at=s.m.stamp(),status='passed',new_fits=0,new_outer_predictions=0,original_MI_views_checked=20,internal_candidate_rows=240,internal_frame_checks=rows,
        role_mapping_and_stored_member_mean_and_C=True,no_formal_choice_frozen=True,source={p.name:s.m.sha(p) for p in [HERE/'runner.py',Path(__file__)]})
    grid.to_csv(out/'internal_grid.csv',index=False);s.m.write_json(out/'CHECK.json',record);s.m.mark_complete(out,['CHECK.json','internal_grid.csv']);print(record)
if __name__=='__main__':main()
