"""目录适配不能改变训练标签、留出身份、分类器及预测值。"""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fp_research import compare


class Partitions(unittest.TestCase):
    def test_explicit_identity_matches_original_interface(self):
        root=Path(__file__).resolve().parents[2]
        train=Path("sample/train")
        test=Path("sample/test")
        splits=[("sample-fold0",[train],[test])]
        truth={("sample","train",str(i)):i%2 for i in range(4)}
        table=pd.DataFrame([[i%2]*6 for i in range(4)],columns=list("abcdef"))
        def features(*_):return list(map(str,range(4))),["TP"]*4,table.copy()
        with tempfile.TemporaryDirectory(dir=root/"result") as tmp:
            old,new=Path(tmp)/"old.csv",Path(tmp)/"new.csv"
            with patch.object(compare,"splits",return_value=splits),patch.object(compare,"load_features",side_effect=features):
                compare.run(root,root.parent/"CLAP","CLAP",truth,old)
                compare.run(None,root.parent/"CLAP","CLAP",truth,new,partitions=splits,
                            identities={train:("sample","train"),test:("sample","test")})
            self.assertEqual(old.read_bytes(),new.read_bytes())
            with new.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([int(r["prediction"]) for r in rows],[0,1,0,1])


if __name__=="__main__":unittest.main()
