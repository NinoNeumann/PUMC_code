"""Full fixed5fold20MI inference-only mixture study; sources are immutable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,importlib.util,itertools,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('formal_heterogeneous21_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
ROOT,OUT,m,rg=b.ROOT,b.OUT,b.m,b.rg
sys.path.insert(0,str(ROOT/'experiments/cacox_mi_shared_cox'));import evaluation_adapter as ev
CM={c['candidate']:c for c in b.CONDITIONS};ROLES=list(b.ROLES)
PLAN=dict(study='cacox_heterogeneous_blend_20261031_v2_stream',conditions=b.CONDITIONS,roles=ROLES,tie_order=b.ORDER,
    formula='w17*s17+(1-w17)*((sNAM-trainingmeanNAM)/trainingSDNAM*trainingSD17+trainingmean17);w1 bitwise source17',
    selection='mean20MI C perinnerfold thenmean5fold;1e-12ties;all5choices before newouter',
    data='fixed1140/149/29candidates/BorutaLASSO15,9,8,9,8/RMICE20maxit60/600originalEncoder arrays/allmemberships',
    new_model_fits=0,source_states=150,inner_groups=25,inner_C_rows=8500,outer_groups=5,outer_MI_units=100,outer_outputs=1800,
    physical_limit_GiB=rg.LIMIT/rg.GiB,reserve_GiB=4,owned_RSS_limit_GiB=8,maximum_workers=1,execution_blocks=dict(candidates=1,MI_views=4,pairs=4096),execution_only_repair=True,
    minimum_C=.76,exact_RCS_plus002=.7613465232424127,interval_scope=ev.SCOPE,
    limitations='samecohort development,fixed-list/inheritedI20 dependence,earlystop/weightselection reuse internalvalidation,repeatedouterview,relatedMI;not independentunbiased confirmation')
def read(p):return b.read(p)
def directory(phase,f,k=None):return OUT/phase/f'fold{f:02d}'/(f'inner{k:02d}' if k else 'shared')
def jobs(phase):return list(itertools.product(range(1,6),range(1,6))) if phase=='inner' else [(f,) for f in range(1,6)]
def progress():return dict(inner_groups=len(list(OUT.glob('inner/fold*/inner*/COMPLETE.json'))),outer_groups=len(list(OUT.glob('outer/fold*/shared/COMPLETE.json'))),outer_MI_units=len(list(OUT.glob('outer/fold*/shared/mi*/COMPLETE.json'))))
def status(**kw):m.status(OUT/'RUN_STATUS.json',controller_pid=os.getpid(),**kw)
def event(phase,**kw):
    with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(dict(at=m.stamp(),phase=phase,**kw))+'\n')
def log(title,purpose,params,result):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as s:s.write(f'\n## {title}\n\n时间：{m.stamp()}\n\n目的：{purpose}\n\n参数：`{json.dumps(params,ensure_ascii=False)}`\n\n结果位置：{OUT}\n\n结果说明：{result}\n')
def signature():
    roots=[b.source17.OUT,b.source20.OUT]
    preserved=read(OUT/'previous_attempt/PRESERVATION.json')
    for p,h in read(b.base.OUT/'manifest.json')['source'].items():assert m.sha(ROOT/p)==h
    assert m.sha(b.base.OUT/'manifest.json')==read(OUT/'math/MATH_CHECK.json')['base21_manifest_SHA']
    for name,h in read(b.base.OUT/'manifest.json')['gates'].items():
        assert m.verify_completed(b.base.OUT/name) and m.sha(b.base.OUT/name/'COMPLETE.json')==h
    for name,item in preserved['preserved'].items():
        assert m.sha(Path(item['original']))==item['SHA'] and m.sha(OUT/'previous_attempt'/name)==item['SHA']
    for root in roots:
        for sub in ['', 'verification','interpretation','visual_qa']:assert m.verify_completed(root/sub)
        for p,h in read(root/'manifest.json')['source'].items():assert m.sha(ROOT/p)==h
    assert b.io.parent.signature()==read(b.io.parent.OUT/'manifest.json')
    for sub in ['math','preflight','benchmark']:assert m.verify_completed(OUT/sub)
    pre=read(OUT/'preflight/PREFLIGHT.json');bench=read(OUT/'benchmark/SUMMARY.json');math=read(OUT/'math/MATH_CHECK.json')
    for proof in [pre,bench,math]:
        for name,h in proof.get('source_files',proof.get('source',{})).items():assert m.sha(HERE/name)==h
    plan=ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION21_PLAN.md';assert m.sha(plan)==pre['registered_plan_SHA']==bench['registered_plan_SHA']
    assert pre['source_states_replayed']==10 and pre['independent_grid_rows']==680 and pre['new_model_fits']==bench['new_model_fits']==0
    chosen=max([r for r in bench['timings'] if r['status']=='passed'],key=lambda r:(r['groups_per_second'],-r['workers']))['workers'];assert chosen==bench['selected_workers']==read(OUT/'RUNTIME_PLAN.json')['workers']
    refs={}
    for f in range(1,6):
        for k in [None,*range(1,6)]:
            for name in ['C_J_SP',*b.NAM]:
                d=b.source_dir(f,k,name);assert m.verify_completed(d);refs[str(d.relative_to(ROOT))]=m.sha(d/'COMPLETE.json')
    amend=plan.with_name('ITERATION21_EXECUTION_V2.md');assert amend.exists()
    files=[*HERE.glob('*.py'),HERE/'README.md',OUT/'config.json',OUT/'RUNTIME_PLAN.json',plan,amend,OUT/'previous_attempt/PRESERVATION.json']
    return dict(plan=PLAN,source={str(p.relative_to(ROOT)):m.sha(p) for p in files},references=refs,source_roots={str(p.relative_to(ROOT)):m.sha(p/'COMPLETE.json') for p in roots},gates={s:m.sha(OUT/s/'COMPLETE.json') for s in ['math','preflight','benchmark']})
def arrays(result):return {cid+suffix:result[cid][i].cpu().numpy() for cid in b.ORDER for i,suffix in enumerate(['_train','_val'])}
def diversity(g,pred,f,k):
    from scipy.stats import spearmanr
    i,j=b.pair_indices(g['val_time'],g['val_event']);rows=[];control=pred['C_J_SP'][1]
    for name in b.NAM:
        other=pred[name][1]
        for view in range(20):
            a,v=control[view],other[view];rows.append(dict(fold=f,inner_fold=k,imputation=view+1,source=name,validation_spearman=float(spearmanr(a,v).statistic),validation_pair_disagreement=float(np.mean(np.sign(a[i]-a[j])!=np.sign(v[i]-v[j])))))
    return rows
def worker_status(phase,index,**kw):m.status(OUT/f'workers/{phase}_{index}/RUN_STATUS.json',status='running',worker_pid=os.getpid(),**kw)
def inner(f,k,index):
    d=directory('inner',f,k)
    if m.verify_completed(d):return
    worker_status('inner',index,fold=f,inner_fold=k,step='load_frozen_sources');g=b.io.load_group(f,k);pred,refs=b.sources(g,f,k);result,stats=b.combine(pred);d.mkdir(parents=True,exist_ok=True)
    worker_status('inner',index,fold=f,inner_fold=k,step='CUDA_candidates_and_counts')
    pd.DataFrame(b.grid(g,result,f,k)).to_csv(d/'grid.csv',index=False);np.savez_compressed(d/'risks.npz',**arrays(result))
    pd.DataFrame(diversity(g,pred,f,k)).to_csv(d/'complementarity.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),source=g['metadata'],references=refs,training_score_statistics=stats,CUDA_inference=all(a.is_cuda and v.is_cuda for a,v in result.values()),new_model_fits=0,source_states=5))
    m.mark_complete(d,['grid.csv','risks.npz','audit.json','complementarity.csv'])
def freeze(write=True):
    assert progress()['inner_groups']==25;choices={}
    for f in range(1,6):
        q=pd.concat([pd.read_csv(directory('inner',f,k)/'grid.csv',float_precision='round_trip') for k in range(1,6)]);assert len(q)==1700
        mean=q.groupby(['candidate','inner_fold']).C_index.mean().groupby('candidate').mean();values={cid:float(mean[cid]) for cid in b.ORDER};top=max(values.values());cid=next(cid for cid in b.ORDER if values[cid]>=top-1e-12)
        choices[str(f)]=dict(candidate=cid,condition=CM[cid],inner_C=values[cid],all_candidate_C=values)
    if write:
        assert progress()['outer_groups']==progress()['outer_MI_units']==0;m.write_json(OUT/'FROZEN_CHOICES.json',dict(at=m.stamp(),choices=choices,all_fivefold_before_outer=True,new_model_fits=0));event('all_fivefold_frozen',choices=choices);log('五折选择冻结','只用每折五内折×20MI决定有限组合',choices,'之后才计算本轮新外层混合，不新训练模型。')
    return choices
def choice(role,f,frozen):return frozen['choices'][str(f)]['candidate'] if role=='E_selected' else role
def frame(g,f,j,role,v,baseline):
    q=g['va'].copy();q['repeat_id']=1;q['fold']=f;q['imputation']=j; q['score']=v;sv=b.io.parent.old.gpu.cuda_survival(v,baseline).cpu().numpy()
    for ix,t in enumerate(m.COMMON['times']):q[f'survival_{t}d']=sv[:,ix]
    return m.normalize(q,role,PLAN['study'],'Frozen17/20 fullrisk convex mixture;training-score alignment;no new model fits')
def outer(f,index):
    d=directory('outer',f)
    if m.verify_completed(d):return
    worker_status('outer',index,fold=f,step='load_frozen_sources');g=b.io.load_group(f);pred,refs=b.sources(g,f,None);result,stats=b.combine(pred);frozen=read(OUT/'FROZEN_CHOICES.json');d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'risks.npz',**arrays(result))
    for j in range(1,21):
        target=d/f'mi{j:02d}';target.mkdir(exist_ok=True);wrappers={};general=[]
        for role in ROLES:
            cid=choice(role,f,frozen);c=CM[cid];a,v=(z[j-1].cpu().numpy() for z in result[cid]);baseline=b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],a)
            frame(g,f,j,role,v,baseline).to_csv(target/(role+'.csv'),index=False)
            wrappers[role]=dict(candidate=cid,condition=c,source_references={name:refs[name] for name in ['C_J_SP',*([c['source']] if cid!=b.CONTROL else [])]},training_statistics=None if cid==b.CONTROL else stats[c['source']][j-1],source=g['metadata'],Encoder=g['metadata']['encoders'][j-1],MI_index=j,baseline={key:value.cpu() for key,value in baseline.items()},new_model_fits=0)
            general.append(dict(role=role,fold=f,imputation=j,train_C=m.c_evaluator(g['tr'])(a),test_C=m.c_evaluator(g['va'])(v)))
        torch.save(wrappers,target/'wrappers.pt');pd.DataFrame(general).to_csv(target/'generalization.csv',index=False);m.mark_complete(target,['wrappers.pt','generalization.csv',*[r+'.csv' for r in ROLES]])
        worker_status('outer',index,fold=f,imputation=j,step='save_risk_probability_wrappers')
    m.write_json(d/'audit.json',dict(at=m.stamp(),source=g['metadata'],references=refs,training_score_statistics=stats,selected=frozen['choices'][str(f)],CUDA_inference=True,new_model_fits=0,source_states=5))
    m.mark_complete(d,['risks.npz','audit.json',*[f'mi{j:02d}/COMPLETE.json' for j in range(1,21)]])
def worker(phase,index):
    try:
        for args in jobs(phase)[index::read(OUT/'RUNTIME_PLAN.json')['workers']]:(inner if phase=='inner' else outer)(*args,index)
        m.status(OUT/f'workers/{phase}_{index}/RUN_STATUS.json',status='complete',worker_pid=os.getpid())
    except BaseException as e:m.status(OUT/f'workers/{phase}_{index}/RUN_STATUS.json',status='failed',worker_pid=os.getpid(),error=repr(e),traceback=traceback.format_exc());raise
def schedule(phase):
    children=[];ticks=0;last=None
    try:
        for i in range(read(OUT/'RUNTIME_PLAN.json')['workers']):
            rg.ensure_headroom();d=OUT/f'workers/{phase}_{i}';d.mkdir(parents=True,exist_ok=True)
            with (d/'out.log').open('a') as oo,(d/'err.log').open('a') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--phase',phase,'--index',str(i)],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
        while any(c.poll() is None for c in children):
            resource=rg.record(OUT/'monitor',ticks%3==0);ticks+=1;pr=progress();status(status='running',phase=phase,progress=pr,worker_launcher_pids=[c.pid for c in children],resource=resource)
            if pr!=last:event(phase,progress=pr);last=pr
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Mixture '+phase+' worker failed')
            time.sleep(2)
        assert all(c.returncode==0 for c in children)
    except BaseException:rg.stop_owned(children);raise
def replay():
    rows=[];frozen=read(OUT/'FROZEN_CHOICES.json')
    for phase in ['inner','outer']:
        for args in jobs(phase):
            f=args[0];k=args[1] if phase=='inner' else None;d=directory(phase,f,k);assert m.verify_completed(d);g=b.io.load_group(f,k);pred,refs=b.sources(g,f,k,replay=True);result,stats=b.combine(pred);audit=read(d/'audit.json');assert audit['source']==g['metadata'] and audit['references']==refs and audit['training_score_statistics']==stats
            with np.load(d/'risks.npz') as z:
                expected=arrays(result);assert set(z.files)==set(expected)
                for key,value in expected.items():np.testing.assert_array_equal(z[key],value)
            if phase=='inner':
                q=pd.DataFrame(b.grid(g,result,f,k)).sort_values(['candidate','imputation']);saved=pd.read_csv(d/'grid.csv',float_precision='round_trip').sort_values(['candidate','imputation']);np.testing.assert_array_equal(q[['C_index','train_C']],saved[['C_index','train_C']])
                for cid in [b.CONTROL,*[c['candidate'] for c in b.CONDITIONS if c['weight17']==0.]]:
                    native=b.native_grid(f,k,CM[cid]['source']);actual=q[q.candidate.eq(cid)].sort_values('imputation');np.testing.assert_array_equal(actual[['C_index','train_C']].to_numpy(),native[['C_index','train_C']].to_numpy())
            else:
                assert audit['selected']==frozen['choices'][str(f)]
                for j in range(1,21):
                    target=d/f'mi{j:02d}';assert m.verify_completed(target);wrappers=torch.load(target/'wrappers.pt',map_location='cpu',weights_only=False);assert set(wrappers)==set(ROLES)
                    for role,w in wrappers.items():
                        cid=choice(role,f,frozen);c=CM[cid];a,v=(z[j-1].cpu().numpy() for z in result[cid]);assert w['candidate']==cid and w['condition']==c and w['source']==g['metadata'] and w['Encoder']==g['metadata']['encoders'][j-1]
                        assert w['training_statistics']==(None if cid==b.CONTROL else stats[c['source']][j-1])
                        for name,proof in w['source_references'].items():assert proof==refs[name]
                        q=pd.read_csv(target/(role+'.csv'),float_precision='round_trip').set_index('patient_id').loc[g['val_ids']];np.testing.assert_array_equal(q.score,v);baseline=b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],a)
                        for key,value in baseline.items():np.testing.assert_array_equal(value.cpu(),w['baseline'][key])
                        np.testing.assert_array_equal(b.io.parent.old.gpu.cuda_survival(v,baseline).cpu().numpy(),q[[f'survival_{t}d' for t in m.COMMON['times']]])
                        if role==b.CONTROL:
                            old=b.source17.OUT/f'outer/fold{f:02d}/shared/mi{j:02d}';assert m.verify_completed(old);native=pd.read_csv(old/'C_J_SP_I20.csv',float_precision='round_trip').set_index('patient_id').loc[g['val_ids']];np.testing.assert_array_equal(q[['score',*[f'survival_{t}d' for t in m.COMMON['times']]]],native[['score',*[f'survival_{t}d' for t in m.COMMON['times']]]])
            rows.append(dict(phase=phase,fold=f,inner_fold=k or 0,source_states=5,MI_views=20,source_member_risk_error=0.,mixture_risk_and_probability_error=0.,new_model_fits=0));rg.record(OUT/'monitor',False);status(status='running',phase='saved_model_replay',groups_replayed=len(rows),progress=progress())
    d=OUT/'verification';d.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(d/'replays.csv',index=False);m.mark_complete(d,['replays.csv'])
def prepare():
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen')
    assert progress()==dict(inner_groups=0,outer_groups=0,outer_MI_units=0);m.write_json(OUT/'config.json',PLAN);m.write_json(OUT/'RUNTIME_PLAN.json',dict(workers=read(OUT/'benchmark/SUMMARY.json')['selected_workers'],maximum=1));m.write_json(OUT/'manifest.json',signature());log('正式冻结','只读17/20模型的有限凸组合',PLAN,'原算法及分块数学/两内部组/单路完整重放计时已通过；0新拟合/外层输出。此前资源保护退出另列。');status(status='prepared',phase='prepared',progress=progress())
def run():
    assert signature()==read(OUT/'manifest.json')
    with m.run_lock(OUT):
        try:
            if progress()['inner_groups']<25:schedule('inner')
            if (OUT/'FROZEN_CHOICES.json').exists():assert freeze(False)==read(OUT/'FROZEN_CHOICES.json')['choices']
            else:freeze()
            if progress()['outer_groups']<5:schedule('outer')
            if not m.verify_completed(OUT/'verification'):replay()
            status(status='running',phase='metrics_and_report',progress=progress());spec=importlib.util.spec_from_file_location('formal_heterogeneous21_report',HERE/'report_study.py');report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report);report.report_all(sys.modules[__name__])
        except BaseException as e:status(status='failed',phase='exception',error=repr(e),traceback=traceback.format_exc(),progress=progress());raise
def launch():
    assert signature()==read(OUT/'manifest.json')
    if (OUT/'RUNNING.lock').exists() or m.verify_completed(OUT):raise RuntimeError('Locked or complete')
    rg.ensure_headroom()
    with (OUT/'controller.out.log').open('a') as oo,(OUT/'controller.err.log').open('a') as ee:child=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW)
    m.write_json(OUT/'LAUNCH.json',dict(at=m.stamp(),launcher_pid=child.pid,workers=read(OUT/'RUNTIME_PLAN.json')['workers']));print('HETEROGENEOUS_BLEND_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':
    b.configure();parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','launch','run','worker']);parser.add_argument('--phase',choices=['inner','outer']);parser.add_argument('--index',type=int);a=parser.parse_args()
    if a.command=='worker':worker(a.phase,a.index)
    else:dict(prepare=prepare,launch=launch,run=run)[a.command]()
