"""配置报告修正后的定位对照；新运行目录与历史结果隔离。"""
import argparse
from datetime import datetime
from pathlib import Path
import uuid

from ..config import ROOT
from ..io import read,atomic_json,digest,validate_run_name
from .catalog import jobs as choose_jobs,METHODS,UPSTREAM


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',default='all',choices=['all','bank','email','elevator','examdb','gpl','zipme'])
    parser.add_argument('--stratum',default='1',choices=['1','2','3','all'])
    parser.add_argument('--version')
    parser.add_argument('--run-name')
    parser.add_argument('--detection-run',help='识别人工运行名；不提供时明确引用已归档的上一轮预测')
    parser.add_argument('--predictions-dir',type=Path,help='比较方法折外预测目录；不提供时引用已归档预测')
    parser.add_argument('--stage',default='all',choices=['all','rank','evaluate','verify','summary'])
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args(argv)
    name=args.run_name or datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    validate_run_name(name)
    if args.detection_run: validate_run_name(args.detection_run)
    selected=choose_jobs(args.system,args.stratum,args.version)
    if args.list:
        for job in selected: print(job['system'],job['stratum'],job['version'])
        print('案例数',len(selected),'；未运行定位。');return 0
    from ..localization import engine
    output=ROOT/'experimental_results/localization_runs'/name
    for job in selected:
        job.update(run_name=name,detection_run=args.detection_run,
                   predictions_dir=str(args.predictions_dir.resolve()) if args.predictions_dir else None)
    protocol=dict(jobs=selected,methods=METHODS,upstream=UPSTREAM,protocol=engine.PROTOCOL,
                  full_scope=args.system=='all' and args.stratum=='all' and not args.version,
                  joint_prediction_source='指定人工识别' if args.detection_run else '已归档的原识别预测',
                  baseline_prediction_source='指定折外预测' if args.predictions_dir else '已归档的折外预测',
                  detection_run=args.detection_run,predictions_dir=str(args.predictions_dir) if args.predictions_dir else None)
    target=output/'manifest.json'
    if target.exists() and read(target)!=protocol: raise ValueError('同名实验的选择或预测来源不同，请更换运行名')
    output.mkdir(parents=True,exist_ok=True);atomic_json(target,protocol)
    if args.stage in {'all','rank'}:
        from ..localization.lookup import indexed_lookup
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        for job in selected:
            ranking,_,_=engine.backend.upstream()
            with indexed_lookup(ranking): result=engine.rank_case(job)
            print(job['system'],job['stratum'],job['version'],result,flush=True)
    if args.stage in {'all','evaluate'}:
        from ..evaluation.localization import evaluate
        evaluate(selected,output)
    if args.stage in {'all','verify'}:
        from ..evaluation.verify import verify
        verify(output)
    if args.stage in {'all','summary'}:
        from .summary import summarize
        print('实验报告',summarize(output),flush=True)
    return 0
