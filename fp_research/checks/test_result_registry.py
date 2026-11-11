import unittest
import pandas as pd
from fp_research.result_registry import check


class RegistryStatistics(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame(dict(system=["sample"]*3,version=["v"]*3,configuration=["a","b","c"],
                                     label=[1,0,1],prediction=[1,1,0],score=[0.8,0.7,None]))
        self.table=dict(n=3,scored=2,insufficient=1,tp=1,fp=1,tn=0,fn=1,
                        precision=.5,recall=.5,f1=.5,accuracy=1/3,hits=1,budget=1,average_precision=5/6)

    def test_missing_score_stays_in_denominator(self):
        check(self.frame,self.table)

    def test_incorrect_metric_rejected(self):
        with self.assertRaises(ValueError):check(self.frame,dict(self.table,f1=1.0))
        with self.assertRaises(ValueError):check(self.frame,dict(self.table,average_precision=1.0))

    def test_duplicate_configuration_rejected(self):
        with self.assertRaises(ValueError):check(pd.concat([self.frame,self.frame.iloc[:1]]),self.table)


if __name__=="__main__":unittest.main()
