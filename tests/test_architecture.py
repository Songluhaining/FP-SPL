"""防止主方法重新依赖评价真值、旧方案及不存在的模块。"""
import ast
from pathlib import Path
import unittest
from fp_research.config import ROOT


class ArchitectureChecks(unittest.TestCase):
    def test_detection_does_not_import_baselines_localization_or_evaluation(self):
        forbidden={'evaluation','experiments','baselines','localization','truth'}
        for path in (ROOT/'fp_research/detection').glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module:
                    self.assertFalse(set(node.module.split('.'))&forbidden,str(path))

    def test_legacy_engines_and_single_fault_inference_are_gone(self):
        for name in ['main.py','pom.xml','pyutils','pylib','src','lib','fp_research/latent_fault.py',
                     'fp_research/joint_pilot.py','fp_research/study.py','fp_research/varcop_localization.py']:
            self.assertFalse((ROOT/name).exists(),name)
        tree=ast.parse((ROOT/'fp_research/data/coverage.py').read_text(encoding='utf-8-sig'))
        self.assertNotIn('infer',{n.name for n in tree.body if isinstance(n,ast.FunctionDef)})

    def test_every_relative_import_resolves_in_current_package(self):
        for path in (ROOT/'fp_research').rglob('*.py'):
            if 'vendor' in path.parts:continue
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if not isinstance(node,ast.ImportFrom) or not node.level:continue
                base=path.parent
                for _ in range(node.level-1):base=base.parent
                if node.module:
                    target=base.joinpath(*node.module.split('.'))
                    self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),f'{path}: {node.module}')
                else:
                    for alias in node.names:
                        target=base/alias.name
                        self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),f'{path}: {alias.name}')


if __name__=='__main__':unittest.main()
