import ast
import builtins
import sys

def check(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.add(alias.name)

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    print("Imports:", imports)
    print("Unused:", imports - used)

check('scripts/aozora_dataset_builder.py')
