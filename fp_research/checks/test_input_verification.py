import json
from pathlib import Path
import tempfile
import unittest

from fp_research.case_study import ROOT,raw_digest,results
from fp_research.input_verification import verify_inputs


class InputVerificationTests(unittest.TestCase):
    def test_parallel_matches_serial_and_rejects_modified_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'result') as tmp:
            folder=Path(tmp);jobs=[];serial={}
            for i in range(3):
                version=folder/str(i);source=version/'variants/cfg/src/Main.java'
                source.parent.mkdir(parents=True);source.write_text('class Main { int x='+str(i)+'; }')
                stamp=results(version,'main');stamp.mkdir(parents=True)
                digest=raw_digest(version)
                (stamp/'complete.json').write_text(json.dumps(dict(raw_digest=digest)))
                jobs.append(('example',version));serial[('example',str(i))]=digest
            self.assertEqual(verify_inputs(jobs,'main',folder/'parallel.json',3),serial)
            (results(jobs[0][1],'main')/'extra.log').write_text('派生结果不属于原始输入')
            self.assertEqual(verify_inputs(list(reversed(jobs)),'main',folder/'reordered.json',2),serial)
            (jobs[0][1]/'variants/cfg/src/Main.java').write_text('class Main { int x=99; }')
            with self.assertRaisesRegex(ValueError,'原始输入改变'):
                verify_inputs(jobs,'main',folder/'changed.json',3)


if __name__=='__main__':unittest.main()
