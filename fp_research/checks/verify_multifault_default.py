"""独立核验PULP源码默认5%抽样补充对照，并合并到统一表。"""
import math
import pandas as pd

from fp_research.inventory import atomic_json
from fp_research.multifault_study import ROOT,OUTPUT,KEYS,read,digest,write_csv
from fp_research.extension_evaluate import eval_metrics
from fp_research.result_registry import check
from fp_research.multifault_presentation import applicability


def main():
    proof=read(OUTPUT/'verified.json')
    if digest(OUTPUT/'predictions.csv')!=proof['predictions_sha256']:raise ValueError('主结果核验凭据不符')
    main=pd.read_csv(OUTPUT/'predictions.csv')
    truth=main[main.method=='joint'][KEYS+['stratum','label','executed_label','base','any_base']]
    data=pd.read_csv(OUTPUT/'PULP-default.csv').merge(truth,on=KEYS,how='left',validate='one_to_one')
    if len(data)!=len(truth) or data.label.isna().any() or set(data[KEYS].itertuples(index=False,name=None))!=set(truth[KEYS].itertuples(index=False,name=None)):
        raise ValueError('默认抽样对照的候选分母不符')
    primary=main[main.method=='PULP'].set_index(KEYS)
    for r in data.to_dict('records'):
        if primary.loc[tuple(r[k] for k in KEYS),'split']!=r['split']:raise ValueError('验证折不符')
        if r['prediction']!=int(r['score']>.5):raise ValueError('默认概率阈值不符')
    original={r['split']:r for r in read(OUTPUT/'PULP-training.json')}
    receipts=read(OUTPUT/'PULP-default-training.json')
    if len(receipts)!=30:raise ValueError('默认抽样折记录不完整')
    prereg=read(OUTPUT/'default-sampling-preregistration.json')
    if digest(OUTPUT/'fold_predictions'/f"PULP-default-{prereg['smoke_fold']}.csv")!=prereg['smoke_sha256']:
        raise ValueError('默认抽样与事先无真值试运行的预测不同')
    for r in receipts:
        p=original[r['split']]
        if r['input_files']!=p['input_files'] or r['testing']!=p['testing']:raise ValueError('两种抽样设置的样本不同')
        n=r['rows']-r['known_failure_positives']
        if r['sampling_ratio']!=.05 or r['sampled_unlabeled_per_estimator']!=int(n*.05) or r['estimators']!=30:
            raise ValueError('不是源码默认抽样设置')
        target=OUTPUT/'fold_predictions'/f"PULP-default-{r['split']}.csv"
        if digest(target)!=r['output_sha256']:raise ValueError('默认抽样预测改变')
    metrics=[]
    for (s,k),part in data.groupby(['system','stratum']):
        groups=[('all',part,'label'),('executed',part,'executed_label'),('all_base',part[part.base==1],'label'),
                ('other',part[part.base==0],'label'),('any_base',part[part.any_base==1],'label'),('no_base',part[part.any_base==0],'label')]
        for subset,sub,label in groups:
            if not len(sub):continue
            row=dict(system=s,stratum=int(k),method='PULP-default',subset=subset,**eval_metrics(sub,label))
            check(sub,row);metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUTPUT/'metrics-PULP-default.csv',index=False)
    data.to_csv(OUTPUT/'predictions-PULP-default.csv',index=False)
    table=pd.read_csv(OUTPUT/'三系统多缺陷实验比较.csv',encoding='utf-8-sig')
    from fp_research.result_registry import NAMES,METRICS
    subset_names={'all':'全部','executed':'缺陷行实际执行标签','all_base':'全部故障均在基础特征','other':'至少一个故障在其他特征',
                  'any_base':'至少一个故障在基础特征','no_base':'没有基础特征故障'}
    rows=[]
    for m in metrics:
        template=table[(table['系统']==NAMES[m['system']])&(table['缺陷数量层']==m['stratum'])&
                       table['方法'].str.contains('PULP')&(table['评价子集']==subset_names[m['subset']])].iloc[0].to_dict()
        template.update(方法='正例与未标注学习（PULP，源码默认5%抽样）',结果来源=str(OUTPUT/'metrics-PULP-default.csv'),
                        预测明细来源=str(OUTPUT/'predictions-PULP-default.csv'),来源文件摘要=digest(OUTPUT/'predictions-PULP-default.csv'),
                        说明='额外固定的源码默认5%抽样对照；相同五折、相同特征、30个分类器、随机种子42；不根据评价结果选择抽样设置。')
        template.update({zh:m[k] for k,zh in METRICS.items()});rows.append(template)
    default_table=applicability(pd.DataFrame(rows))
    default_table.to_csv(OUTPUT/'PULP默认抽样补充比较.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(ROOT/'experimental_results/六系统单缺陷实验比较.csv',encoding='utf-8-sig');old.insert(1,'缺陷数量层',1)
    pd.concat([old,table,default_table],ignore_index=True).to_csv(ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv',index=False,encoding='utf-8-sig')
    report=OUTPUT/'实验报告.md';body=report.read_text(encoding='utf8').split('\n## 默认抽样补充对照')[0]
    lines=['','## 默认抽样补充对照','',
           '主表中的PULP沿用此前等量抽样设置，并非源码构造函数的默认5%。两种设置在查看评价分数前固定运行，以下同时报告，避免将参数不同的结果混为一谈。','',
           '|系统|缺陷数|沿用等量抽样的调和平均值|默认5%抽样的调和平均值|','|---|---:|---:|---:|']
    main_metrics=pd.read_csv(OUTPUT/'metrics.csv')
    for r in metrics:
        if r['subset']!='all':continue
        previous=main_metrics[(main_metrics.system==r['system'])&(main_metrics.stratum==r['stratum'])&(main_metrics.method=='PULP')&(main_metrics.subset=='all')].iloc[0]
        values=(f'{previous.f1*100:.2f}%',f"{r['f1']*100:.2f}%") if r['tp']+r['fn'] else ('—','—')
        lines.append(f"|{NAMES[r['system']]}|{r['stratum']}|{values[0]}|{values[1]}|")
    lines.extend(['','两种抽样设置的输入特征摘要、配置顺序及测试版本已逐折核对一致，变化仅为抽样比例。详细结果见[PULP默认抽样补充比较](PULP默认抽样补充比较.csv)。','',
        '## 结果解读与重复核验','',
        '在存在基准阳性的四个系统与缺陷数组合中，联合贝叶斯方法在三个组合的调和平均值最高；电梯三缺陷中，有监督分类与默认5%抽样的正例学习均为96.71%，高于联合方法的95.07%。该组合中联合方法误报4个、漏报24个，前两者均误报20个、漏报0个，说明减少误报伴随检出率下降，不能声称所有场景均领先。',
        '考试数据库968个候选均无基准阳性，联合方法误报0个；等量抽样正例学习误报13个，默认5%抽样误报188个。有监督分类因训练折只有一个类别而无法评分。这里仅能比较可评分方法的误报，不能评价阳性检出能力。','',
        '三种基线设置在每个系统、每个缺陷层各重训第一个验证折，共18次；预测与首次运行逐值一致，凭据见各方法的重复核验文件（*-repeat-check.json）。这些是确定性复现检查，不是额外独立样本或显著性检验。',''])
    report.write_text(body+'\n'.join(lines),encoding='utf8')
    atomic_json(OUTPUT/'default-verified.json',dict(candidates=len(data),folds=len(receipts),same_inputs=True,
        sampling_ratio=.05,metrics_sha256=digest(OUTPUT/'metrics-PULP-default.csv'),predictions_sha256=digest(OUTPUT/'predictions-PULP-default.csv')))
    completion=read(OUTPUT/'completion.json')
    completion['default_sampling_verified']=digest(OUTPUT/'default-verified.json')
    for p in (report,ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv'):
        completion['files'][p.name]=digest(p)
    atomic_json(OUTPUT/'completion.json',completion)
    print(pd.DataFrame(metrics).query("subset=='all'").to_string(index=False),flush=True)


if __name__=='__main__':main()
