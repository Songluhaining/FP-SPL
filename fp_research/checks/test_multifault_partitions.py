"""核验多缺陷比较中的系统、缺陷层与版本隔离。"""
import unittest
from fp_research.multifault_study import partitions


class PartitionsTests(unittest.TestCase):
    def test_every_version_held_out_once_within_its_own_stratum(self):
        jobs=[dict(system=s,stratum=k,version=f'NOB_{k}.ID_{i}',path=f'F:/datasets/{s}/{k}/NOB_{k}.ID_{i}')
              for s in ('elevator','email','examdb') for k in (2,3) for i in range(11)]
        splits,identities=partitions(jobs)
        self.assertEqual(len(splits),30)
        seen=[]
        for name,train,test in splits:
            self.assertFalse(set(train)&set(test))
            self.assertEqual(len(train)+len(test),11)
            system,stratum,_=name.split('-')
            for path in train+test:
                self.assertEqual(identities[path][0],system)
                self.assertIn('NOB_'+stratum[0],identities[path][1])
            seen.extend(test)
        self.assertEqual(len(seen),len(jobs))
        self.assertEqual(len(set(seen)),len(jobs))
        self.assertEqual(splits,partitions(list(reversed(jobs)))[0])


if __name__=='__main__': unittest.main()
