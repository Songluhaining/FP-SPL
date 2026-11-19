"""保留上一版人工命令；实际实现位于detection.cli。"""
from .detection.cli import main
if __name__=='__main__': raise SystemExit(main())
