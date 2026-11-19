"""核验报告修正边界、读取路径及没有实际失败轨迹时的行为。"""
from pathlib import Path
import tempfile
import unittest

from fp_research.varcop_report_relabel import write_report, load_report, use_report, write_spcs, observed_spcs


class ReportRelabelChecks(unittest.TestCase):
    def test_exact_last_field_replacement_and_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "original.csv", root / "joint/config.report.csv"
            raw = b'variant,Base,Optional,result\r\na,T,F,__PASSED__\r\nb,T,T,__FAILED__\r\nc,T,T,__PASSED__\r\n'
            source.write_bytes(raw)
            write_report(source, target, {"a"})
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(target.read_bytes(), raw.replace(b'a,T,F,__PASSED__', b'a,T,F,__FAILED__'))
            self.assertEqual(len(load_report(target)[1]), 3)
            for invalid in ({"absent"}, {"b"}):
                with self.assertRaises(ValueError):
                    write_report(source, target, invalid)
            with self.assertRaises(ValueError):
                write_report(source, source, {"a"})

    def test_all_bound_report_readers_and_failure_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "config.report.csv"
            original.write_text('variant,Base,Optional,result\na,T,F,__PASSED__\nb,T,T,__FAILED__\n')
            target = root / "derived/config.report.csv"
            write_report(original, target, {"a"})
            from fp_research.varcop_localization import upstream
            _, _, spc = upstream()
            from util import FileManager
            from ranking import VarBugManager
            modules = [FileManager, spc, VarBugManager]
            previous = [m.get_model_configs_report_path for m in modules]
            self.assertEqual(VarBugManager.is_var_bug_by_config(str(root), ['Base']), 1)
            with use_report(root, target):
                self.assertTrue(all(Path(m.get_model_configs_report_path(str(root))) == target for m in modules))
                self.assertEqual(VarBugManager.is_var_bug_by_config(str(root), ['Base']), 0)
                with self.assertRaises(ValueError):
                    spc.get_model_configs_report_path(str(root / 'another-case'))
            self.assertEqual([m.get_model_configs_report_path for m in modules], previous)
            info = write_spcs(root, target, root / 'spc.log')
            self.assertEqual((info['passed'], info['failed'], info['configurations']), (0, 2, 2))
            self.assertEqual((root / 'spc.log').read_text(), '')

    def test_keep_report_failure_without_fabricating_failed_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coverage = root / 'variants/observed/coverage'
            coverage.mkdir(parents=True)
            (coverage / 'spectrum_failed_coverage.xml').write_text('<coverage/>')
            log, target = root / 'spc.log', root / 'spc-observed.log'
            observed = f"Optional; {(root / 'variants/observed/src').as_posix()}\n"
            inferred = f"Optional; {(root / 'variants/inferred/src').as_posix()}\n"
            log.write_text(observed+inferred)
            self.assertEqual(observed_spcs(root, log, target), ['inferred'])
            self.assertEqual(target.read_text(), observed)
            self.assertEqual(log.read_text(), observed+inferred)


if __name__ == '__main__':
    unittest.main()
