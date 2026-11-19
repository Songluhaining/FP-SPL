"""版本五折的分组、留出完整性和单版本边界。"""
from pathlib import Path
import unittest
from fp_research.evaluation.partitions import partitions


class PartitionChecks(unittest.TestCase):
    def test_every_version_held_out_once_within_system_and_fault_layer(self):
        jobs=[dict(system=s,stratum=n,version=f'_MultipleBugs_.NOB_{n}.ID_{i}')
              for s in ['bank','email'] for n in [1,2,3] for i in range(7)]
        folders={(j['system'],j['version']):Path(j['system'])/j['version'] for j in jobs}
        folds,identities=partitions(jobs,folders)
        self.assertEqual(len(folds),30)
        seen=[]
        for name,training,testing in folds:
            self.assertFalse(set(training)&set(testing))
            system,tail=name.split('-',1)
            fault_layer=int(tail.split('fault',1)[0])
            for path in training+testing:
                self.assertEqual(identities[path][0],system)
                self.assertIn(f'NOB_{fault_layer}.',identities[path][1])
            seen.extend(testing)
        self.assertEqual(sorted(map(str,seen)),sorted(map(str,folders.values())))

    def test_cannot_train_a_classifier_on_the_only_evaluated_version(self):
        job=dict(system='bank',stratum=1,version='one')
        with self.assertRaises(ValueError): partitions([job],{('bank','one'):Path('one')})


if __name__=='__main__':unittest.main()
