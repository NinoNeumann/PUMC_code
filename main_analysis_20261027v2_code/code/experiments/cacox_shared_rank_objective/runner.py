"""Full5fold20MI rank-objective study, immutable inputs and inner-only choices."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,importlib.util,itertools,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('formal_shared_rankobjective_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
ROOT,OUT,m,rg=b.ROOT,b.OUT,b.m,b.rg
sys.path.insert(0,str(ROOT/'experiments/cacox_mi_shared_cox'));import evaluation_adapter as ev
NEW=[c['condition_id'] for c in b.CONDITIONS];ORDER=['control_NLL','control_C',*NEW];CM={c['condition_id']:c for c in b.CONDITIONS}
ROLES=[cid+'_'+kind for cid in NEW for kind in ['NN','I20']]+['S_NN','S_I20']
PLAN=dict(study='cacox_shared_rank_objective_20261031_v1',conditions=b.CONDITIONS,controls=ORDER[:2],tie_order=ORDER,roles=ROLES,common=b.COMMON,config=b.CONFIG,
    seed='same source seed0:1010000+fold1000+(innerfold or9)100',
    train_objective='H=.5mean20MI8memberEfron+.5mean20MImeanComparablePairSoftplus(mean8risk);P=rank only',
    monitor='fixed NLL or actual FP64mean8 C;strict improvement;patience150',
    selection='per innerfold mean20MI C,then mean5fold;6choices;tolerance1e-12 tie-order fixed;no outer scores',
    epoch_rule='percondition/fold ceilmedian5 inner bestepochs;all5fold choices and budgets frozen before outer',NN_alpha=b.ALPHA,
    data='fixed1140/149/29candidate/BorutaLASSO15,9,8,9,8/RMICE20/maxit60/allouter-inner memberships/eachMIEncoder andarrays',
    inner_groups=25,new_inner_NN=100,reused_inner_NN=50,new_outer_NN=20,reused_outer_NN=10,outer_groups=5,outer_MI_units=100,outer_outputs=1000,
    no_new_tree_kernel=True,no_RCS_v2_risk_branch=True,no_sensitivity=True,physical_limit_GiB=rg.LIMIT/rg.GiB,reserve_GiB=4,owned_RSS_limit_GiB=8,maximum_workers=3,
    minimum_C=.76,additional_exact_RCS_plus002=.7613465232424127,interval_scope=ev.SCOPE,
    limitations='samecohort continueddevelopment,fixed-list/inheritedI20 dependence,repeatedouter viewing,correlatedMI;not independent fullynested validation')
def read(p):return b.read(p)
def directory(phase,f,k=None):return OUT/phase/f'fold{f:02d}'/(f'inner{k:02d}' if k is not None else 'shared')
def jobs(phase):return list(itertools.product(range(1,6),range(1,6))) if phase=='inner' else [(f,) for f in range(1,6)]
def progress():return dict(inner_groups=len(list(OUT.glob('inner/fold*/inner*/COMPLETE.json'))),inner_new_NN=len(list(OUT.glob('inner/fold*/inner*/fits/*/COMPLETE.json'))),
    outer_groups=len(list(OUT.glob('outer/fold*/shared/COMPLETE.json'))),outer_new_NN=len(list(OUT.glob('outer/fold*/shared/fits/*/COMPLETE.json'))),outer_MI_units=len(list(OUT.glob('outer/fold*/shared/mi*/COMPLETE.json'))))
def status(**kw):m.status(OUT/'RUN_STATUS.json',controller_pid=os.getpid(),**kw)
def event(phase,**kw):
    with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(at=m.stamp(),phase=phase,**kw))+'\n')
def log(title,purpose,params,result):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as stream:stream.write(f'\n## {title}\n\n时间：{m.stamp()}\n\n目的：{purpose}\n\n参数：`{json.dumps(params,ensure_ascii=False)}`\n\n结果位置：{OUT}\n\n结果说明：{result}\n')
def signature():
    for root in [b.io.OUT,b.previous.OUT]:
        for d in [root,root/'verification',root/'interpretation',root/'visual_qa']:assert m.verify_completed(d)
        for path,h in read(root/'manifest.json')['source'].items():assert m.sha(ROOT/path)==h
    assert b.io.parent.signature()==read(b.io.parent.OUT/'manifest.json')
    for d in [OUT/'preflight',OUT/'benchmark']:assert m.verify_completed(d)
    pre=read(OUT/'preflight/PREFLIGHT.json');bench=read(OUT/'benchmark/SUMMARY.json')
    for proof in [pre,bench]:
        for name,h in proof['source_files'].items():assert m.sha(HERE/name)==h
    plan=ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION15_PLAN.md';prototype=ROOT/'experiments/cacox_rank_objective_prototype/pair_loss.py'
    assert m.sha(plan)==pre['registered_plan_SHA'] and m.sha(prototype)==pre['prototype_pair_SHA'] and bench['balanced_types'] and bench['all_selected_checks_passed']
    refs={}
    for f in range(1,6):
        for k in [None,*range(1,6)]:
            for monitor in ['NLL','C']:
                d=b.source_dir(f,k,monitor);assert m.verify_completed(d);refs[str(d.relative_to(ROOT))]=m.sha(d/'COMPLETE.json')
    files=[*HERE.glob('*.py'),HERE/'README.md',OUT/'config.json',OUT/'RUNTIME_PLAN.json',plan,prototype]
    return dict(plan=PLAN,source={str(p.relative_to(ROOT)):m.sha(p) for p in files},references=refs,source12_complete=m.sha(b.io.OUT/'COMPLETE.json'),source14_complete=m.sha(b.previous.OUT/'COMPLETE.json'),
        preflight=m.sha(OUT/'preflight/COMPLETE.json'),benchmark=m.sha(OUT/'benchmark/COMPLETE.json'))
def train(g,c,f,k,d,index,epochs=None):
    if m.verify_completed(d):
        net,obj=b.load(d/'neural.pt');assert obj['metadata']==g['metadata'] and obj['condition']==c and obj['seed']==b.seed(f,k);return net
    d.mkdir(parents=True,exist_ok=True)
    def callback(v):rg.ensure_headroom();m.status(OUT/f'workers/{"inner" if k is not None else "outer"}_{index}/RUN_STATUS.json',status='running',worker_pid=os.getpid(),fold=f,inner_fold=k,condition=c,**v)
    net,trace,audit=b.fit(g,c,f,k,fixed_epochs=epochs,progress=callback);a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx'])
    b.save(net,d/'neural.pt',g,c,f,k);np.savez_compressed(d/'predictions.npz',train_members=a,val_members=v,train_ids=g['train_ids'],val_ids=g['val_ids'])
    audit.update(fold=f,inner_fold=k,metadata=g['metadata']);m.write_json(d/'audit.json',audit);pd.DataFrame(trace).to_csv(d/'training_trace.csv',index=False);m.mark_complete(d,['neural.pt','predictions.npz','audit.json','training_trace.csv']);return net
def predictions(g,f,k,d):
    pred={};refs={}
    for cid in ORDER:
        source=b.source_dir(f,k,cid.split('_')[1]) if cid.startswith('control_') else d/'fits'/cid;assert m.verify_completed(source)
        obj=torch.load(source/'neural.pt',map_location='cpu',weights_only=False);assert obj['metadata']==g['metadata'] and obj['seed']==b.seed(f,k) and obj['config']==b.CONFIG and obj['common']==b.COMMON
        with np.load(source/'predictions.npz') as z:
            np.testing.assert_array_equal(z['train_ids'],g['train_ids']);np.testing.assert_array_equal(z['val_ids'],g['val_ids']);pred[cid]=(z['train_members'].mean(2),z['val_members'].mean(2))
        refs[cid]=dict(directory=str(source.relative_to(ROOT)),COMPLETE_SHA=m.sha(source/'COMPLETE.json'),neural_SHA=m.sha(source/'neural.pt'))
    return pred,refs
def blend(g,a,v,f,j):return b.io.source.r.mix(g['I20_train'][j-1],g['I20_val'][j-1],a,v,1-b.ALPHA[f-1])
def grid(g,pred,f,k):
    C,TC=m.c_evaluator(g['va']),m.c_evaluator(g['tr']);rows=[]
    for cid,(a,v) in pred.items():
        for j in range(1,21):
            for kind in ['NN','I20']:
                aa,vv=(a[j-1],v[j-1]) if kind=='NN' else blend(g,a[j-1],v[j-1],f,j)[:2]
                rows.append(dict(candidate=cid,kind=kind,fold=f,inner_fold=k,imputation=j,C_index=C(vv),train_C=TC(aa)))
    assert len(rows)==240;return rows
def prefixes(d,write=True):
    rows=[];initial=[]
    for name in ['H','P']:
        a=read(d/f'fits/{name}_NLL/audit.json');c=read(d/f'fits/{name}_C/audit.json');assert a['initial_state_SHA']==c['initial_state_SHA'];initial.append(a['initial_state_SHA'])
        x=pd.read_csv(d/f'fits/{name}_NLL/training_trace.csv',float_precision='round_trip');y=pd.read_csv(d/f'fits/{name}_C/training_trace.csv',float_precision='round_trip');n=min(len(x),len(y))
        fields=['training_mean_view_member_Efron','training_mean_view_ensemble_pair_loss','training_objective','gradient_norm_before_clip']
        if a['inner_fold'] is not None:fields+=['validation_mean_view_ensemble_Efron','validation_twice_wins']
        np.testing.assert_array_equal(x.iloc[:n][fields].to_numpy(),y.iloc[:n][fields].to_numpy());rows.append(dict(objective=name,common_epochs=n,training_monitor_prefix_error=0.))
    assert len(set(initial))==1
    if write:pd.DataFrame(rows).to_csv(d/'monitor_prefix_checks.csv',index=False)
    else:assert pd.read_csv(d/'monitor_prefix_checks.csv').to_dict('records')==rows
    return rows
def inner(f,k,index):
    d=directory('inner',f,k)
    if m.verify_completed(d):return
    g=b.io.load_group(f,k);d.mkdir(parents=True,exist_ok=True)
    for c in b.CONDITIONS:net=train(g,c,f,k,d/'fits'/c['condition_id'],index);del net
    prefixes(d);pred,refs=predictions(g,f,k,d);pd.DataFrame(grid(g,pred,f,k)).to_csv(d/'grid.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),source=g['metadata'],references=refs,new_sharedNN=4,reused_controls=2,outer_evaluated=False));m.mark_complete(d,['audit.json','grid.csv','monitor_prefix_checks.csv',*[f'fits/{cid}/COMPLETE.json' for cid in NEW]])
def freeze(write=True):
    assert progress()['inner_groups']==25 and progress()['inner_new_NN']==100;epochs={};choices={}
    for f in range(1,6):
        epochs[str(f)]={};choices[str(f)]={};q=pd.concat([pd.read_csv(directory('inner',f,k)/'grid.csv',float_precision='round_trip') for k in range(1,6)],ignore_index=True)
        assert len(q)==1200;means=q.groupby(['candidate','kind','inner_fold']).C_index.mean().groupby(['candidate','kind']).mean()
        for cid in ORDER:
            source=lambda k:b.source_dir(f,k,cid.split('_')[1]) if cid.startswith('control_') else directory('inner',f,k)/'fits'/cid
            best=[read(source(k)/'audit.json')['best_epoch'] for k in range(1,6)];budget=int(np.ceil(np.median(best)));epochs[str(f)][cid]=dict(best_inner_epochs=best,epochs=budget,reused=cid.startswith('control_'))
            if cid.startswith('control_'):assert read(b.source_dir(f,None,cid.split('_')[1])/'audit.json')['best_epoch']==budget
        for kind in ['NN','I20']:
            values={cid:float(means.loc[cid,kind]) for cid in ORDER};top=max(values.values());cid=next(cid for cid in ORDER if values[cid]>=top-1e-12)
            choices[str(f)]['S_'+kind]=dict(candidate=cid,kind=kind,inner_C=values[cid],all_candidate_C=values)
    if write:
        assert progress()['outer_new_NN']==progress()['outer_MI_units']==0;m.write_json(OUT/'FROZEN_CHOICES.json',dict(at=m.stamp(),epochs=epochs,choices=choices,all_fivefold_before_outer=True,new_inner_NN=100))
        event('all_fivefold_frozen',epochs=epochs,choices=choices);log('全部选型与轮数冻结','只用5内折×20MI确定预算和两个有限回退选择',dict(epochs=epochs,choices=choices),'之后才新外层训练，无外层选型。')
    return dict(epochs=epochs,choices=choices)

def role_choice(role,f,frozen):
    if role.startswith('S_'):return frozen['choices'][str(f)][role]['candidate'],role[2:]
    return role.rsplit('_',1)
def frame(g,f,j,role,v,baseline):
    q=g['va'].copy();q['repeat_id']=1;q['fold']=f;q['imputation']=j;q['score']=v;sv=b.io.parent.old.gpu.cuda_survival(v,baseline).cpu().numpy()
    for idx,t in enumerate(m.COMMON['times']):q[f'survival_{t}d']=sv[:,idx]
    return m.normalize(q,role,PLAN['study'],'SharedMI TabM final meanrisk survival-pair training objective;all fixed inputs')
def outer(f,index):
    d=directory('outer',f)
    if m.verify_completed(d):return
    g=b.io.load_group(f);d.mkdir(parents=True,exist_ok=True);frozen=read(OUT/'FROZEN_CHOICES.json');epochs=frozen['epochs'][str(f)]
    for c in b.CONDITIONS:net=train(g,c,f,None,d/'fits'/c['condition_id'],index,epochs[c['condition_id']]['epochs']);del net
    prefixes(d);pred,refs=predictions(g,f,None,d)
    for j in range(1,21):
        target=d/f'mi{j:02d}';target.mkdir(exist_ok=True);wrappers={};general=[]
        for role in ROLES:
            cid,kind=role_choice(role,f,frozen);a,v=pred[cid][0][j-1],pred[cid][1][j-1];stats=None
            if kind=='I20':a,v,stats=blend(g,a,v,f,j)
            baseline=b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],a);frame(g,f,j,role,v,baseline).to_csv(target/(role+'.csv'),index=False)
            wrappers[role]=dict(candidate=cid,kind=kind,neural_source=refs[cid],epochs=epochs[cid],MI_index=j,Encoder=g['metadata']['encoders'][j-1],source=g['metadata'],statistics=stats,
                NN_alpha=b.ALPHA[f-1] if kind=='I20' else 1.,I20_root=g['metadata']['original_I20_root'] if kind=='I20' else None,baseline={key:value.cpu() for key,value in baseline.items()})
            general.append(dict(role=role,fold=f,imputation=j,train_C=m.c_evaluator(g['tr'])(a),test_C=m.c_evaluator(g['va'])(v)))
        torch.save(wrappers,target/'wrappers.pt');pd.DataFrame(general).to_csv(target/'generalization.csv',index=False);m.mark_complete(target,['wrappers.pt','generalization.csv',*[r+'.csv' for r in ROLES]])
    m.write_json(d/'audit.json',dict(at=m.stamp(),source=g['metadata'],references=refs,new_sharedNN=4,reused_controls=2,MI_outputs=200));m.mark_complete(d,['audit.json','monitor_prefix_checks.csv',*[f'fits/{cid}/COMPLETE.json' for cid in NEW],*[f'mi{j:02d}/COMPLETE.json' for j in range(1,21)]])
def worker(phase,index):
    path=OUT/f'workers/{phase}_{index}/RUN_STATUS.json'
    try:
        for args in jobs(phase)[index::read(OUT/'RUNTIME_PLAN.json')['workers']]:(inner if phase=='inner' else outer)(*args,index)
        m.status(path,status='complete',worker_pid=os.getpid())
    except BaseException as e:m.status(path,status='failed',worker_pid=os.getpid(),error=repr(e),traceback=traceback.format_exc());raise
def schedule(phase):
    children=[];count=read(OUT/'RUNTIME_PLAN.json')['workers'];ticks=0;last=None
    try:
        for i in range(count):
            rg.ensure_headroom();d=OUT/f'workers/{phase}_{i}';d.mkdir(parents=True,exist_ok=True)
            with (d/'out.log').open('a') as oo,(d/'err.log').open('a') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--phase',phase,'--index',str(i)],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
        while any(c.poll() is None for c in children):
            resource=rg.record(OUT/'monitor',ticks%3==0);ticks+=1;pr=progress();status(status='running',phase=phase,progress=pr,worker_launcher_pids=[c.pid for c in children],resource=resource)
            if pr!=last:event(phase,progress=pr);last=pr
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Rank objective '+phase+' worker failure')
            time.sleep(2)
        assert all(c.returncode==0 for c in children)
    except BaseException:rg.stop_owned(children);raise
def prepare():
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen')
    assert progress()==dict(inner_groups=0,inner_new_NN=0,outer_groups=0,outer_new_NN=0,outer_MI_units=0)
    m.write_json(OUT/'config.json',PLAN);m.write_json(OUT/'RUNTIME_PLAN.json',dict(workers=read(OUT/'benchmark/SUMMARY.json')['selected_workers'],maximum=3))
    m.write_json(OUT/'manifest.json',signature());log('正式冻结','固定共享NN最终风险排序目标，内部监控匹配',PLAN,'真实完整20MI预检和四类型均衡并发通过；尚无15新外层。');status(status='prepared',phase='prepared',progress=progress())
def replay():
    rows=[];frozen=read(OUT/'FROZEN_CHOICES.json')
    for phase in ['inner','outer']:
        for args in jobs(phase):
            f=args[0];k=args[1] if phase=='inner' else None;d=directory(phase,f,k);assert m.verify_completed(d);g=b.io.load_group(f,k);assert g['metadata']==read(d/'audit.json')['source']
            for cid in ORDER:
                fit=b.source_dir(f,k,cid.split('_')[1]) if cid.startswith('control_') else d/'fits'/cid;assert m.verify_completed(fit);net,obj=b.load(fit/'neural.pt');assert obj['metadata']==g['metadata'] and obj['seed']==b.seed(f,k)
                if cid in NEW:assert obj['condition']==CM[cid]
                a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx']);del net
                with np.load(fit/'predictions.npz') as z:np.testing.assert_array_equal(a,z['train_members']);np.testing.assert_array_equal(v,z['val_members'])
            prefixes(d,False);pred,refs=predictions(g,f,k,d)
            if phase=='inner':
                x=pd.DataFrame(grid(g,pred,f,k)).sort_values(['candidate','kind','imputation']);y=pd.read_csv(d/'grid.csv',float_precision='round_trip').sort_values(['candidate','kind','imputation']);np.testing.assert_array_equal(x[['C_index','train_C']],y[['C_index','train_C']])
            else:
                for j in range(1,21):
                    target=d/f'mi{j:02d}';assert m.verify_completed(target);wrappers=torch.load(target/'wrappers.pt',map_location='cpu',weights_only=False);assert set(wrappers)==set(ROLES)
                    for role,w in wrappers.items():
                        cid,kind=role_choice(role,f,frozen);a,v=pred[cid][0][j-1],pred[cid][1][j-1];stats=None
                        if kind=='I20':a,v,stats=blend(g,a,v,f,j)
                        assert w['candidate']==cid and w['kind']==kind and w['statistics']==stats and w['neural_source']==refs[cid] and w['epochs']==frozen['epochs'][str(f)][cid] and w['source']==g['metadata'] and w['Encoder']==g['metadata']['encoders'][j-1]
                        q=pd.read_csv(target/(role+'.csv'),float_precision='round_trip').set_index('patient_id').loc[g['val_ids']];np.testing.assert_array_equal(q.score,v)
                        baseline=b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],a)
                        for key,value in baseline.items():np.testing.assert_array_equal(value.cpu(),w['baseline'][key])
                        np.testing.assert_array_equal(b.io.parent.old.gpu.cuda_survival(v,baseline).cpu().numpy(),q[[f'survival_{t}d' for t in m.COMMON['times']]])
            rows.append(dict(phase=phase,fold=f,inner_fold=k or 0,MI_views=20,new_NN=4,reused_NN=2,member_and_combined_risk_error=0.,probability_error=0.,matched_monitor_prefixes=2));rg.record(OUT/'monitor',False);status(status='running',phase='saved_model_replay',groups_replayed=len(rows),progress=progress())
    d=OUT/'verification';d.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(d/'replays.csv',index=False);m.mark_complete(d,['replays.csv'])
def run():
    assert signature()==read(OUT/'manifest.json')
    with m.run_lock(OUT):
        try:
            if progress()['inner_groups']<25:schedule('inner')
            if (OUT/'FROZEN_CHOICES.json').exists():assert freeze(False)=={key:read(OUT/'FROZEN_CHOICES.json')[key] for key in ['epochs','choices']}
            else:freeze()
            if progress()['outer_groups']<5:schedule('outer')
            if not m.verify_completed(OUT/'verification'):replay()
            status(status='running',phase='metrics_and_report',progress=progress());spec=importlib.util.spec_from_file_location('shared_rankobjective_report',HERE/'report_study.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.report_all(sys.modules[__name__])
        except BaseException as e:status(status='failed',phase='exception',error=repr(e),traceback=traceback.format_exc(),progress=progress());raise
def launch():
    assert signature()==read(OUT/'manifest.json')
    if (OUT/'RUNNING.lock').exists() or m.verify_completed(OUT):raise RuntimeError('Locked or complete')
    rg.ensure_headroom()
    with (OUT/'controller.out.log').open('a',encoding='utf-8') as oo,(OUT/'controller.err.log').open('a',encoding='utf-8') as ee:child=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW)
    m.write_json(OUT/'LAUNCH.json',dict(at=m.stamp(),launcher_pid=child.pid,workers=read(OUT/'RUNTIME_PLAN.json')['workers']));print('SHARED_RANK_OBJECTIVE_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');b.configure();parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','launch','run','worker']);parser.add_argument('--phase',choices=['inner','outer']);parser.add_argument('--index',type=int);a=parser.parse_args()
    if a.command=='worker':worker(a.phase,a.index)
    else:dict(prepare=prepare,launch=launch,run=run)[a.command]()
