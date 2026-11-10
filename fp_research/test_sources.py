"""定位生成的单元测试方法，保留注解和方法中的字符串、注释与嵌套块。"""
import re

METHOD=re.compile(r"@Test(?:\s*\([^)]*\))?\s+(?:public\s+)?(?:final\s+)?void\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^\{]+)?\{")
LEX=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[{}]',re.S)
PACKAGE=re.compile(r"\bpackage\s+([\w.]+)\s*;")


def methods(source):
    for match in METHOD.finditer(source):
        depth=1
        for token in LEX.finditer(source,match.end()):
            if token[0]=="{":
                depth+=1
            elif token[0]=="}":
                depth-=1
            if depth==0:
                yield match[1],match.start(),token.end()
                break
        else:
            raise ValueError("测试方法未闭合："+match[1])


def filtered(source,keep):
    spans=[(start,end) for name,start,end in methods(source) if not keep(name)]
    for start,end in reversed(spans):
        source=source[:start]+source[end:]
    return source
