"""实验原型：观测等价机制的稀疏联合后验与配置风险识别区间。

不接收缺陷数量、变异日志或配置真值。触发率用贝塔(1/2,1/2)的
切比雪夫求积离散近似，退火粒子采样保留机制之间的后验依赖。
区间是等价位置分配下的风险界，不是蒙特卡洛或频率学置信区间。
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.special import logsumexp, softmax


@dataclass
class JointEvidence:
    execution: np.ndarray       # distinct test signatures x equivalence classes
    passed: np.ndarray          # number of passing executions per signature
    failed: np.ndarray
    membership: np.ndarray      # configurations x classes, prior allocation
    lower_membership: np.ndarray
    upper_membership: np.ndarray
    class_mass: np.ndarray      # aggregated feature-balanced prior mass


def levels(points=9):
    if points < 2:
        raise ValueError('触发率求积至少两个节点')
    q = (1 + np.cos((2*np.arange(1, points+1)-1)*np.pi/(2*points)))/2
    return np.r_[0., np.sort(q)]


def validate(e):
    a=np.asarray(e.execution)
    if a.ndim!=2 or a.shape[1]==0 or not np.isin(a,[0,1]).all():
        raise ValueError('需要非空二值执行矩阵')
    if any(np.shape(x)!=(len(a),) for x in (e.passed,e.failed)):
        raise ValueError('用例计数维度不符')
    if np.any(e.passed<0) or np.any(e.failed<0) or np.sum(e.failed)==0:
        raise ValueError('必须有失败证据且计数非负')
    for x in (e.membership,e.lower_membership,e.upper_membership):
        if x.ndim!=2 or x.shape[1]!=a.shape[1] or np.any(x<0) or np.any(x>1):
            raise ValueError('配置成员关系无效')
    if np.any(e.lower_membership>e.membership+1e-12) or np.any(e.membership>e.upper_membership+1e-12):
        raise ValueError('成员关系界不一致')
    if len(e.class_mass)!=a.shape[1] or np.any(e.class_mass<=0) or not np.isclose(e.class_mass.sum(),1):
        raise ValueError('等价类先验质量必须归一化')


def prior_table(e, points, sparsity):
    if sparsity<=0:
        raise ValueError('稀疏先验参数必须为正')
    activation=-np.expm1(-sparsity*e.class_mass)
    return np.column_stack((np.log1p(-activation),
                            np.repeat(np.log(activation[:,None]/points),points,axis=1)))


def log_likelihood(survival,e):
    return survival@e.passed + np.log(-np.expm1(survival))@e.failed


def conditional_likelihood(base,execution,passed,failed,logs,passing_count=None):
    """仅计算依赖当前变量的项；通过证据线性汇总，失败证据保留逐行计算。"""
    affected=(execution>0)&(failed>0)
    failures=failed[affected]
    local=base[:,affected]
    passing=float(execution@passed) if passing_count is None else float(passing_count)
    return np.column_stack([x*passing+np.log(-np.expm1(local+x))@failures for x in logs])


def risks(states,e,weights=None):
    active=states>0
    values=[]
    for membership in (e.membership,e.lower_membership,e.upper_membership):
        by_particle=np.ones((len(states),len(membership)))
        for g in range(active.shape[1]):
            by_particle*=1-active[:,g,None]*membership[None,:,g]
        values.append(np.average(1-by_particle,axis=0,weights=weights))
    if not (np.all(values[1]<=values[0]+1e-10) and np.all(values[0]<=values[2]+1e-10)):
        raise ValueError('风险上下界错误')
    return dict(score=values[0],lower=values[1],upper=values[2],
                inclusion=np.average(active,axis=0,weights=weights),
                expected_mechanisms=float(np.average(active.sum(axis=1),weights=weights)))


def exact(e,points=5,sparsity=1.5):
    """仅供小图核验：枚举完整联合状态，不用独立边缘概率近似风险。"""
    validate(e)
    g=e.execution.shape[1]
    if (points+1)**g>2000000:
        raise ValueError('精确枚举只用于小图')
    states=np.array(list(product(range(points+1),repeat=g)),dtype=np.int16)
    prior=prior_table(e,points,sparsity)
    epsilon=1/(float(e.passed.sum()+e.failed.sum())+1)
    survival=np.log1p(-epsilon)+np.log1p(-levels(points)[states])@e.execution.T
    logw=log_likelihood(survival,e)+prior[np.arange(g),states].sum(axis=1)
    return dict(**risks(states,e,softmax(logw)),log_evidence=float(logsumexp(logw)))


def infer_joint(e,particles=256,points=9,sparsity=1.5,seed=42,sweeps=2):
    validate(e)
    if particles<32 or sweeps<1:
        raise ValueError('至少32粒子和一次完整更新')
    rng=np.random.default_rng(seed)
    g=e.execution.shape[1]
    prior=prior_table(e,points,sparsity)
    q=levels(points)
    logs=np.log1p(-q)
    epsilon=1/(float(e.passed.sum()+e.failed.sum())+1)
    states=np.zeros((particles,g),dtype=np.int16)
    for j in range(g):
        states[:,j]=rng.choice(points+1,size=particles,p=np.exp(prior[j]))
    # 通过项在对数域为线性项，只保留每个候选的通过覆盖次数。
    # 非线性的失败项仍保留全部原始执行模式及其次数。
    failing=e.failed>0
    a=e.execution[failing].astype(float)
    failed=e.failed[failing]
    passed_hits=e.execution.T@e.passed
    passed_log=np.log1p(-epsilon)*e.passed.sum()+logs[states]@passed_hits
    survival=np.log1p(-epsilon)+logs[states]@a.T
    ll=passed_log+np.log(-np.expm1(survival))@failed
    beta=0.;logz=0.;trace=[]
    while beta<1-1e-12:
        if len(trace)>=256:
            raise RuntimeError('退火步数超限，不能把未收敛输出作为完成结果')
        def ess(delta):
            w=softmax(delta*ll)
            return 1/np.sum(w*w)
        hi=1-beta
        if ess(hi)>=0.8*particles:
            delta=hi
        else:
            lo=0.
            for _ in range(35):
                mid=(lo+hi)/2
                if ess(mid)<0.8*particles:hi=mid
                else:lo=mid
            delta=lo
        if delta<=1e-12:
            raise RuntimeError('退火无法前进')
        w=softmax(delta*ll)
        logz+=float(logsumexp(delta*ll)-np.log(particles))
        ids=np.searchsorted(np.cumsum(w),(rng.random()+np.arange(particles))/particles)
        ids=np.minimum(ids,particles-1)
        states=states[ids].copy();survival=survival[ids].copy();ll=ll[ids].copy();passed_log=passed_log[ids].copy()
        beta=min(1.,beta+delta)
        swaps=0
        for _ in range(sweeps):
            for j in rng.permutation(g):
                base=survival-logs[states[:,j,None]]*a[None,:,j]
                conditional=conditional_likelihood(base,a[:,j],None,failed,logs,passing_count=passed_hits[j])
                proposed=ll[:,None]+conditional-conditional[np.arange(particles),states[:,j],None]
                probabilities=softmax(beta*conditional+prior[j],axis=1)
                draw=(rng.random((particles,1))>np.cumsum(probabilities,axis=1)).sum(axis=1)
                draw=np.minimum(draw,points)
                passed_log+=(logs[draw]-logs[states[:,j]])*passed_hits[j]
                states[:,j]=draw
                survival=base+logs[draw,None]*a[None,:,j]
                ll=proposed[np.arange(particles),draw]
            if g>1:
                # 对称的两机制交换提议，帮助在互相解释的后验模态之间移动。
                j=rng.integers(g,size=particles)
                k=(j+rng.integers(1,g,size=particles))%g
                row=np.arange(particles)
                sj=states[row,j].copy();sk=states[row,k].copy()
                prop=survival+(logs[sk]-logs[sj])[:,None]*(a[:,j]-a[:,k]).T
                ppassed=passed_log+(logs[sk]-logs[sj])*(passed_hits[j]-passed_hits[k])
                pll=ppassed+np.log(-np.expm1(prop))@failed
                ratio=beta*(pll-ll)+prior[j,sk]+prior[k,sj]-prior[j,sj]-prior[k,sk]
                accepted=np.log(rng.random(particles))<ratio
                r=row[accepted]
                states[r,j[accepted]]=sk[accepted];states[r,k[accepted]]=sj[accepted]
                survival[accepted]=prop[accepted];ll[accepted]=pll[accepted];passed_log[accepted]=ppassed[accepted]
                swaps+=int(accepted.sum())
        trace.append(dict(beta=beta,ess=float(1/np.sum(w*w)),unique=int(len(np.unique(states,axis=0))),swaps=swaps))
    return dict(**risks(states,e),log_evidence=logz,trace=trace,
                states=states,epsilon=epsilon,seed=seed,particles=particles,
                points=points,sparsity=sparsity)
