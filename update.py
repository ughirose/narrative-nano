with open("scripts/aozora_dataset_builder.py", "r") as f:
    lines = f.readlines()
with open("scripts/aozora_dataset_builder.py", "w") as f:
    for line in lines:
        if "import urllib.error" not in line:
            f.write(line)
