import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch,Mock

from fp_research import case_finish as finish


class SchedulingTests(unittest.TestCase):
    def test_main_evaluation_precedes_external_masked_completion(self):
        events=[]
        with tempfile.TemporaryDirectory(dir=finish.ROOT/'result') as tmp:
            root=Path(tmp)
            versions={s:[root/s/'case'] for s in finish.SYSTEMS}
            for paths in versions.values():
                for scenario in ('main','masked'):
                    output=finish.results(paths[0],scenario);output.mkdir(parents=True)
                    (output/'complete.json').write_text('{}')
            for scenario in ('main','masked'):
                output=root/'experimental_results/gpl_zipme'/scenario;output.mkdir(parents=True)
                for name in ('metrics.csv','predictions.csv'):(output/name).write_text('test')
            process=Mock();process.create_time.return_value=1
            process.is_running.side_effect=[True,True,True,True,False,False,False,False]
            def run(args,**kwargs):events.append(('command',args[2],args[3:]))
            def sleep(seconds):events.append(('wait',seconds))
            with patch.object(finish,'ROOT',root),patch.object(finish,'case_versions',side_effect=lambda d,s:versions[s]), \
                 patch.object(finish.psutil,'Process',return_value=process),patch.object(finish.subprocess,'Popen') as spawn, \
                 patch.object(finish.subprocess,'run',side_effect=run),patch.object(finish.time,'sleep',side_effect=sleep), \
                 patch.object(finish,'make'),patch.object(finish,'verify'), \
                 patch('sys.argv',['case_finish','--include-masked','--wait-masked-pid','zipme:123']):
                finish.main()
            spawn.assert_not_called()
            first_wait=next(i for i,e in enumerate(events) if e[0]=='wait')
            main_fit=next(i for i,e in enumerate(events) if e[0]=='command' and e[2][:1]==['fit'])
            self.assertLess(main_fit,first_wait)
            self.assertEqual(sum(e[0]=='command' and e[2][:1]==['fit'] for e in events),4)
            self.assertEqual(events[-1][1],'fp_research.case_report')


if __name__=='__main__':unittest.main()
