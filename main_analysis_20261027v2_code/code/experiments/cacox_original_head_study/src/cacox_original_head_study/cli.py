import argparse
import json
from pathlib import Path
from .runtime import Study,configure,stamp
from .provenance import validate
from .experiment import smoke,run


def main():
    parser=argparse.ArgumentParser(description="Fixed original CACox vs original CACox with discrete survival head")
    parser.add_argument("command",choices=["validate","smoke","run"])
    parser.add_argument("--project",type=Path,default=Path.cwd())
    parser.add_argument("--config",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();config=args.config or args.project/"experiments/cacox_original_head_study/configs/study_v1.json"
    study=Study(args.project,config);study.out.mkdir(parents=True,exist_ok=True);configure();start=stamp()
    if not (study.out/"EXPERIMENTS.md").exists():
        (study.out/"EXPERIMENTS.md").write_text("# 原始CACox更换生存预测头\n\n固定所有训练超参数，不使用v2或样条分支。记录时间、目的、参数、结果位置与解释。\n",encoding="utf-8")
    if args.command=="validate":
        result=validate(study)
        study.log_experiment("原始模型与数学核验",start,"确保新工程复现原始100模型且未混入v2",study.config["common"],study.out/"validation","100个检查点、共同骨干初始化、删失似然及生存/RMST解析核验通过")
    elif args.command=="smoke":
        result=smoke(study)
        study.log_experiment("真实数据小试跑",start,"固定参数检查原始骨干两种头的训练与保存读取",study.config["common"],study.out/"validation/smoke","内部开发/验证分区试跑，未评价新模型外层结果")
    else:
        if (study.out/"manifest.json").exists() and not args.resume:raise FileExistsError("Existing run requires explicit --resume")
        result=run(study)
    print(json.dumps(dict(command=args.command,status=result.get("status"),output=str(study.out)),ensure_ascii=False),flush=True)
