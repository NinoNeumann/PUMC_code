"""Two fixed internal groups,10 source states,40MI views; no new fitting."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('preflight_heterogeneous21_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'preflight';PLAN=b.ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION21_PLAN.md'
def main():
    b.configure();start=time.perf_counter()
    if OUT.exists():raise RuntimeError('Preserve preflight artifacts')
    assert PLAN.exists() and b.m.verify_completed(b.OUT/'math')
    math=b.read(b.OUT/'math/MATH_CHECK.json')
    for name,h in math['source'].items():assert b.m.sha(HERE/name)==h
    for root in [b.source17.OUT,b.source20.OUT]:
        for sub in ['', 'verification','interpretation','visual_qa']:assert b.m.verify_completed(root/sub)
    OUT.mkdir();plan_sha=b.m.sha(PLAN);records=[]
    for f in [1,4]:
        b.rg.ensure_headroom();g=b.io.load_group(f,1);predictions,proof=b.sources(g,f,1,replay=True);result,stats=b.combine(predictions);rows=b.grid(g,result,f,1);assert len(rows)==340
        directory=OUT/f'fold{f:02d}';directory.mkdir();pd.DataFrame(rows).to_csv(directory/'grid.csv',index=False);arrays={}
        for c in b.CONDITIONS:
            cid,name,weight=c['candidate'],c['source'],c['weight17'];a,v=(z.cpu().numpy() for z in result[cid]);arrays[cid+'_train']=a;arrays[cid+'_val']=v
            control_train,control_val=predictions['C_J_SP']
            expected_a,expected_v=(control_train,control_val) if weight==1. else (b.blend.numpy_score(control_train,predictions[name][0],weight,stats[name]),b.blend.numpy_score(control_val,predictions[name][1],weight,stats[name]))
            np.testing.assert_allclose(a,expected_a,atol=2e-12,rtol=0);np.testing.assert_allclose(v,expected_v,atol=2e-12,rtol=0)
            if weight==1.:np.testing.assert_array_equal(a,control_train);np.testing.assert_array_equal(v,control_val)
            for j in range(20):
                saved=rows[b.ORDER.index(cid)*20+j]
                assert b.independent_C(g['train_time'],g['train_event'],a[j])==saved['train_C']
                assert b.independent_C(g['val_time'],g['val_event'],v[j])==saved['C_index']
            if weight in [0.,1.]:
                native=b.native_grid(f,1,name);observed=pd.DataFrame(rows);observed=observed[observed.candidate.eq(cid)].sort_values('imputation')
                np.testing.assert_array_equal(observed[['C_index','train_C']].to_numpy(),native[['C_index','train_C']].to_numpy())
        b.m.write_json(directory/'audit.json',dict(fold=f,inner_fold=1,source=g['metadata'],references=proof,training_score_statistics=stats,new_model_fits=0,new_outer_predictions=0,source_weight_replays=5,independent_grid_rows=340))
        np.savez_compressed(directory/'risks.npz',**arrays);b.m.mark_complete(directory,['grid.csv','audit.json','risks.npz']);b.rg.record(OUT/'monitor',True);record=dict(fold=f,source_states_replayed=5,MI_views=20,grid_rows=340,CPUCUDAC_and_endpoints_error=0.,new_model_fits=0,new_outer_predictions=0);records.append(record);print(json.dumps(record),flush=True)
    assert b.m.sha(PLAN)==plan_sha
    summary=dict(at=b.m.stamp(),status='passed',groups=records,source_states_replayed=10,MI_views=40,independent_grid_rows=680,new_model_fits=0,new_outer_predictions=0,registered_plan_SHA=plan_sha,math_complete_SHA=b.m.sha(b.OUT/'math/COMPLETE.json'),source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',HERE/'blend.py',HERE/'check_math.py',Path(__file__)]},wall_seconds=time.perf_counter()-start)
    b.m.write_json(OUT/'PREFLIGHT.json',summary);b.m.mark_complete(OUT,['PREFLIGHT.json','fold01/COMPLETE.json','fold04/COMPLETE.json']);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
