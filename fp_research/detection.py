"""条件集合注意力网络：预测已知某类失效存在时，一组测试能否暴露它。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

FIELDS = ("operations","assertions","inputs")


def descriptor(raw):
    # 只允许模型看操作、输入和断言，标识、目录及测试结果均不编码。
    return tuple(frozenset(raw[field]) for field in FIELDS)


def encode(reference, tests):
    rows=[]
    for case in tests:
        row=[]
        for query,observed in zip(reference,case):
            common=len(query & observed)
            row.extend((common / max(1,len(query)), common / max(1,len(observed)),
                        len(observed) / (1 + len(observed))))
        rows.append(row)
    return np.array(rows,dtype="float32")


class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance=nn.Sequential(nn.Linear(9,24),nn.ReLU(),nn.Linear(24,16),nn.ReLU())
        self.attention=nn.Linear(16,1)
        self.readout=nn.Sequential(nn.Linear(33,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,values,mask):
        features=self.instance(values)
        weights=self.attention(features).squeeze(-1).masked_fill(~mask,-torch.inf).softmax(dim=1)
        pooled=(features*weights.unsqueeze(-1)).sum(dim=1)
        maximum=features.masked_fill(~mask.unsqueeze(-1),-torch.inf).amax(dim=1)
        count=mask.sum(dim=1,keepdim=True).float().log1p()
        return self.readout(torch.cat((pooled,maximum,count),dim=1)).squeeze(-1)


def tensors(bags, omit_assertions=False):
    values=np.zeros((len(bags),max(map(len,bags)),9),dtype=np.float32)
    mask=np.zeros(values.shape[:2],dtype=bool)
    for i,bag in enumerate(bags):
        values[i,:len(bag)]=bag
        mask[i,:len(bag)]=True
    if omit_assertions:
        values[:,:,3:6]=0
    return torch.from_numpy(values),torch.from_numpy(mask)


def samples(version_dir):
    sem=json.loads((version_dir/"semantics.json").read_text(encoding="utf8"))
    cases={k:descriptor(d) for k,d in sem["cases"].items()}
    by_config={}
    for k in cases:
        by_config.setdefault(k.split("/",1)[0],[]).append(k)
    seed=int.from_bytes(hashlib.sha256((version_dir.parent.name+version_dir.name).encode()).digest()[:8],"big")
    rng=np.random.default_rng(seed)
    bags,labels=[],[]
    for members in sem["groups"].values():
        positives=set(members)
        for config in sorted({k.split("/",1)[0] for k in members}):
            others=[k for k in members if k.split("/",1)[0]!=config]
            if not others:
                continue
            reference=cases[others[0]]
            ids=by_config[config]
            features=encode(reference,[cases[k] for k in ids])
            # 均匀抽取独立子集，标签只由被抽中用例的历史失效记录决定。
            for size in sorted({1,min(4,len(ids)),max(1,len(ids)//2),len(ids)}):
                for _ in range(1 if size==len(ids) else 3):
                    indices=rng.choice(len(ids),size=size,replace=False)
                    bags.append(features[indices])
                    labels.append(int(any(ids[i] in positives for i in indices)))
    return bags,np.asarray(labels,dtype="float32")


def train_model(bags,labels,omit_assertions=False,seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng=np.random.default_rng(seed)
    model=DetectionNet()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002)
    for epoch in range(12):
        indices=rng.permutation(len(bags))
        model.train()
        for start in range(0,len(indices),128):
            ids=indices[start:start+128]
            logits=model(*tensors([bags[i] for i in ids],omit_assertions))
            loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(labels[ids]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model,bags,omit_assertions=False):
    return model(*tensors(bags,omit_assertions)).sigmoid().numpy()


def frequency_features(bags):
    # 按数量和与该失效的操作/断言/输入相似度匹配历史子集。
    return np.stack([np.r_[np.log1p(len(b))/5,b[:,[0,3,6]].mean(axis=0),b[:,[0,3,6]].max(axis=0)] for b in bags])
