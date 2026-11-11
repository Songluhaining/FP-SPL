import unittest
import json
from pathlib import Path
from fp_research.test_environment import mapped_scaffolding


class EnvironmentTests(unittest.TestCase):
    def test_only_environment_assignments(self):
        body='''System.setProperty("java.io.tmpdir", "/tmp");
System.setProperty("user.dir", "/home/source");
System.setProperty("user.home", "/home/owner");
System.setProperty("user.language", "en");
assertEquals("/tmp", testedPath);
'''
        mapped,changes=mapped_scaffolding(body,Path('D:/isolated'))
        self.assertEqual(len(changes),3)
        restored=mapped
        for row in changes:
            restored=restored.replace(json.dumps(str(Path('D:/isolated'))),json.dumps(row['original']),1)
        self.assertEqual(restored,body)
        self.assertIn('assertEquals("/tmp", testedPath);',mapped)

    def test_no_environment_no_change(self):
        body='assertEquals(1, product.run());'
        self.assertEqual(mapped_scaffolding(body,Path('D:/isolated')),(body,[]))


if __name__=='__main__':unittest.main()
