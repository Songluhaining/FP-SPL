"""核验人工入口选例边界、独立目录及报告修正，不启动真实案例推断。"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fp_research.manual_joint import corrected_report, scenario, select_jobs, run_job


class ManualJointChecks(unittest.TestCase):
    def test_wrapper_uses_existing_fit_and_restores_case_output_namespace(self):
        from fp_research import multifault_study as core
        previous = core.SCENARIO
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            (case/'config.report.csv.done').write_text('Product,Base,__TEST_OUTPUT__\na,T,__PASSED__\nb,T,__FAILED__\n')
            job = dict(system='bank', stratum=2, version=case.name, path=str(case), configurations=2, candidates=1)
            def fake_fit(received):
                self.assertEqual(received, job)
                self.assertEqual(core.SCENARIO, 'manual_joint_detection/check-one')
                output = core.folder(received)
                output.mkdir(parents=True)
                (output/'joint.csv').write_text('configuration,prediction\na,1\n')
                (output/'joint.json').write_text(json.dumps(dict(stable=True)))
                return 'test wrapper only'
            with patch.object(core, 'fit', side_effect=fake_fit) as called:
                self.assertTrue(run_job(job, 'check-one'))
                called.assert_called_once_with(job)
            self.assertEqual(core.SCENARIO, previous)
            output = case/'.fp_research/manual_joint_detection/check-one'
            self.assertTrue((output/'manual-run.json').is_file())
            self.assertIn('a,T,__FAILED__', (output/'config.report.joint.csv').read_text())
            self.assertIn('a,T,__PASSED__', (case/'config.report.csv.done').read_text())
            with patch.object(core, 'fit', side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError): run_job(job, 'check-failure')
            self.assertEqual(core.SCENARIO, previous)

    def test_all_six_systems_and_deferred_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locations = dict(systems={s:str(root/s) for s in ['bank','email','elevator','examdb','gpl','zipme']},
                             deferred_multifault=['gpl','zipme'])
            for system, base in locations['systems'].items():
                for n in (1, 2, 3):
                    case = Path(base)/f'4wise-{system}-{n}BUG-Full'/f'_MultipleBugs_.NOB_{n}.ID_1'
                    case.mkdir(parents=True)
                    (case/'config.report.csv.done').write_text('Product,Base,__TEST_OUTPUT__\na,T,__PASSED__\nb,T,__FAILED__\n')
            jobs = select_jobs(locations, 'all', 'all')
            self.assertEqual(len(jobs), 14)
            self.assertEqual({j['system'] for j in select_jobs(locations, 'all', '1')}, set(locations['systems']))
            self.assertEqual(len(select_jobs(locations, 'bank', '2', '_MultipleBugs_.NOB_2.ID_1')), 1)
            direct = select_jobs(locations, case=jobs[0]['path'])
            self.assertEqual(direct[0], jobs[0])
            with self.assertRaises(ValueError): select_jobs(locations, 'gpl', '2')
            with self.assertRaises(ValueError): select_jobs(locations, 'bank', '1', 'missing')
            self.assertFalse(any(root.rglob('.fp_research')))

    def test_reports_use_only_predictions_and_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, predictions, target = root/'original.csv', root/'joint.csv', root/'result/config.report.joint.csv'
            raw = b'Product,Base,__TEST_OUTPUT__\r\na,T,__PASSED__\r\nb,T,__FAILED__\r\nc,T,__PASSED__\r\n'
            original.write_bytes(raw)
            predictions.write_text('configuration,prediction\na,1\nc,0\n')
            self.assertEqual(corrected_report(original, predictions, target), 1)
            self.assertEqual(original.read_bytes(), raw)
            self.assertEqual(target.read_bytes(), raw.replace(b'a,T,__PASSED__', b'a,T,__FAILED__'))
            self.assertEqual(corrected_report(original, predictions, target), 1)
            predictions.write_text('configuration,prediction\na,1\n')
            with self.assertRaises(ValueError): corrected_report(original, predictions, target)
            predictions.write_text('configuration,prediction\na,0\nc,0\n')
            with self.assertRaises(ValueError): corrected_report(original, predictions, target)

    def test_run_name_cannot_escape_case_derived_directory(self):
        self.assertEqual(scenario('manual-20261119_1'), 'manual_joint_detection/manual-20261119_1')
        for invalid in ['../original', 'a/b', 'a\\b', 'C:\\case', '', '.', '..']:
            with self.assertRaises(ValueError): scenario(invalid)


if __name__ == '__main__':
    unittest.main()
