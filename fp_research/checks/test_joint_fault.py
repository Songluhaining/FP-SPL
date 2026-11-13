import unittest
import numpy as np

from fp_research.joint_fault import JointEvidence,exact,infer_joint,risks


class JointFaultTests(unittest.TestCase):
    def evidence(self):
        return JointEvidence(np.array([[1,0],[0,1],[0,0]],dtype=float),
            np.array([1.,1.,20.]),np.array([8.,8.,0.]),
            np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),
            np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),
            np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),np.array([.5,.5]))

    def test_two_disjoint_faults_can_coexist(self):
        r=exact(self.evidence())
        self.assertTrue(np.all(r['inclusion']>.99))
        self.assertGreater(r['expected_mechanisms'],1.98)
        self.assertEqual(r['score'][3],0)

    def test_joint_risk_not_product_of_marginals(self):
        e=self.evidence()
        states=np.array([[1,0],[0,1]])
        r=risks(states,e)
        self.assertEqual(r['score'][2],1)
        self.assertNotEqual(r['score'][2],1-np.prod(1-r['inclusion']))

    def test_identification_interval_contains_allocations(self):
        e=JointEvidence(np.array([[1.]]),np.array([1.]),np.array([5.]),
             np.array([[.3],[1.],[0.]]),np.array([[0.],[1.],[0.]]),
             np.array([[1.],[1.],[0.]]),np.array([1.]))
        r=exact(e)
        self.assertEqual(r['lower'][0],0)
        self.assertAlmostEqual(r['upper'][0],r['inclusion'][0])
        self.assertAlmostEqual(r['score'][0],.3*r['inclusion'][0])

    def test_particles_match_exact_and_preserve_order(self):
        e=self.evidence()
        # 一个失败覆盖两候选：保留竞争解释，而不是两个独立边缘。
        e.execution=np.array([[1.,1.],[0.,0.]])
        e.passed=np.array([1.,10.]);e.failed=np.array([8.,0.])
        reference=exact(e,points=5)
        for seed in (42,2026):
            r=infer_joint(e,particles=2048,points=5,seed=seed,sweeps=3)
            np.testing.assert_allclose(r['score'],reference['score'],atol=.045,rtol=0)
            self.assertAlmostEqual(r['trace'][-1]['beta'],1)
        order=[1,0]
        e.execution=e.execution[order];e.passed=e.passed[order];e.failed=e.failed[order]
        np.testing.assert_allclose(exact(e,points=5)['score'],reference['score'],atol=1e-12)


if __name__=='__main__':unittest.main()
