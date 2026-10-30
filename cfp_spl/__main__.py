"""包执行入口：命令 python -m cfp_spl 从这里转入命令行参数解析。"""

from .cli import main

raise SystemExit(main())
