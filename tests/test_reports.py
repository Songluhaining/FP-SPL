"""核验报告修正边界、读取路径及没有实际失败轨迹时的行为。"""
from pathlib import Path
import tempfile
import unittest

from fp_research.localization.engine import write_report, load_report, use_report, write_spcs, observed_spcs
from fp_research.localization.lookup import indexed_lookup


class ReportRelabelChecks(unittest.TestCase):
    def test_index_preserves_first_match_missing_and_restoration(self):
        from types import SimpleNamespace
        def original(name, sequence):
            for row in sequence:
                if row[0] == name:
                    return row[1], row[2]
            return -1000, 0
        ranking = SimpleNamespace(get_local_score=original, STM_NOT_FOUND=-1000)
        sequence = [('a', .5, 1), ('b', .3, 2), ('a', .1, 3)]
        with indexed_lookup(ranking):
            for name in ['a', 'b', 'missing']:
                self.assertEqual(ranking.get_local_score(name, sequence), original(name, sequence))
            self.assertEqual(ranking.get_local_score('a', sequence), (.5, 1))
        self.assertIs(ranking.get_local_score, original)

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
            from fp_research.localization.common import upstream
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

    def test_changed_status_actually_changes_feature_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, corrected = root / 'config.report.csv', root / 'new/config.report.csv'
            original.write_text('variant,Base,Optional,result\na,T,F,__PASSED__\nb,T,T,__FAILED__\nc,F,T,__PASSED__\n')
            write_report(original, corrected, {'a'})
            before, after = root / 'before.log', root / 'after.log'
            write_spcs(root, original, before)
            write_spcs(root, corrected, after)
            combinations = lambda p: {frozenset(line.split(';', 1)[0].split(',')) for line in p.read_text().splitlines()}
            self.assertEqual(combinations(before), {frozenset({'Base', 'Optional'})})
            self.assertIn(frozenset({'Base'}), combinations(after))
            self.assertNotIn(frozenset({'Base', 'Optional'}), combinations(after))
            self.assertEqual(len(load_report(corrected)[1]), 3)


if __name__ == '__main__':
    unittest.main()
