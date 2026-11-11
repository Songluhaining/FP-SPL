import tempfile
from pathlib import Path
import unittest
from fp_research.case_overlap import conflicts,completed_aliases


class CacheIsolation(unittest.TestCase):
    def test_only_unfinished_shared_keys_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            self.assertEqual(conflicts(['a','b','b'],{'b','c'},cache),['b'])
            (cache/'b').mkdir()
            (cache/'b/partial.log').write_text('not completed')
            self.assertEqual(conflicts(['b'],{'b'},cache),['b'])
            (cache/'b/complete.json').write_text('{}')
            self.assertEqual(conflicts(['a','b'],{'b'},cache),[])
            self.assertEqual(conflicts(['a'],set(),cache),[])

    def test_refresh_only_accepts_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            index={'entries':[{'raw':'a','canonical':'same'},{'raw':'b','canonical':'other'}]}
            self.assertEqual(completed_aliases(index,cache),{})
            (cache/'a').mkdir()
            (cache/'a/complete.json').write_text('{}')
            self.assertEqual(completed_aliases(index,cache),{'same':'a'})
            (cache/'b').mkdir()
            (cache/'b/partial.log').write_text('unfinished')
            self.assertEqual(completed_aliases(index,cache),{'same':'a'})


if __name__=='__main__':unittest.main()
