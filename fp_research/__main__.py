"""统一入口：识别、比较方法、定位及历史归档核验。"""
import argparse
import importlib
import sys
COMMANDS={'detect':'detection.cli','baseline':'baselines.cli','localize':'experiments.cli','archive-check':'evaluation.archive'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=COMMANDS)
    parsed=parser.parse_args(sys.argv[1:2])
    return importlib.import_module('fp_research.'+COMMANDS[parsed.command]).main(sys.argv[2:])
if __name__=='__main__': raise SystemExit(main())
