"""将无阳性、无评分的指标明确标为不适用，不改变任何预测或原始统计。"""
import pandas as pd
from .multifault_study import ROOT,OUTPUT,read,digest
from .inventory import atomic_json


def applicability(table):
    table=table.copy()
    table['基准阳性数']=table['检出数']+table['漏报数']
    table['基准阴性数']=table['误报数']+table['正确排除数']
    empty=table['成功评分数']==0;no_positive=table['基准阳性数']==0
    table.loc[empty,['精确率','召回率','调和平均值','准确率','平均精确率']]=float('nan')
    table.loc[no_positive,['召回率','调和平均值','平均精确率']]=float('nan')
    table.loc[(table['检出数']+table['误报数'])==0,'精确率']=float('nan')
    table['统计适用说明']=[('全部无法评分；混淆矩阵为按未报出处理的占位计数，不能视作模型准确率。' if e else '')+
                         ('无基准阳性，不能评价假通过检出能力；仅可考察误报。' if p else '')
                         for e,p in zip(empty,no_positive)]
    return table


def publish():
    proof=read(OUTPUT/'verified.json')
    if digest(OUTPUT/'predictions.csv')!=proof['predictions_sha256']:raise ValueError('已核验明细改变')
    table=applicability(pd.read_csv(OUTPUT/'三系统多缺陷实验比较.csv',encoding='utf-8-sig'))
    table.to_csv(OUTPUT/'三系统多缺陷实验比较.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(ROOT/'experimental_results/六系统单缺陷实验比较.csv',encoding='utf-8-sig');old.insert(1,'缺陷数量层',1)
    pd.concat([old,table],ignore_index=True).to_csv(ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv',index=False,encoding='utf-8-sig')
    report=OUTPUT/'实验报告.md';body=report.read_text(encoding='utf8')
    start=body.index('|系统|缺陷数|方法|候选数|');end=body.index('\n\n',start)
    lines=['|系统|缺陷数|方法|候选数|成功评分|精确率|召回率|调和平均值|准确率|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in table[table['评价子集']=='全部'].to_dict('records'):
        values=[r['系统'],str(int(r['缺陷数量层'])),r['方法'],str(int(r['候选配置数'])),str(int(r['成功评分数']))]
        values += ['—' if pd.isna(r[k]) else f'{r[k]*100:.2f}%' for k in ('精确率','召回率','调和平均值','准确率')]
        lines.append('|'+ '|'.join(values)+'|')
    body=body[:start]+'\n'.join(lines)+body[end:]
    marker='\n## 指标适用性说明'
    body=body.split(marker)[0]
    body+=marker+'\n\n“—”表示指标不适用或模型未能评分。考试数据库双、三缺陷的基准阳性均为0；有监督分类的训练折均为单一类别，968个候选全部无法评分。该系统不能用于证明假通过检出能力，只能考察有评分方法的误报。\n\n机器可读的原始统计保留此前的零分母记零、缺分按未报出计数约定，便于核对共同分母；中文表明确留空不适用的性能项，不能把未评分配置的占位准确率当成分类器成绩。\n'
    report.write_text(body,encoding='utf8')
    if (OUTPUT/'completion.json').exists():
        complete=read(OUTPUT/'completion.json')
        for p in (report,ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv'):complete['files'][p.name]=digest(p)
        atomic_json(OUTPUT/'completion.json',complete)


if __name__=='__main__':publish()
