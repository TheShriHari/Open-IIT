"""Only evaluate.py may refer to surveyed_addresses.csv in code (docstrings and comments are fine)."""
import ast

from conftest import ROOT

KEY = "surveyed_addresses"


def code_strings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_only_evaluate_reads_answer_key():
    files = [p for p in ROOT.rglob("*.py") if "tests" not in p.parts and ".git" not in p.parts]
    offenders = [p.relative_to(ROOT).as_posix() for p in files
                 if any(KEY in s for s in code_strings(p))]
    assert offenders == ["evaluate.py"]
