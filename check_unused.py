import ast
import sys

def check(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.name)

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    print("Imports:", imports)
    print("Unused:", imports - names)

check("scripts/aozora_dataset_builder.py")
