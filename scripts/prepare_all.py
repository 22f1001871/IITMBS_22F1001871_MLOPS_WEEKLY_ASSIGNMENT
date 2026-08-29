import subprocess
import sys

def run(*args):
    subprocess.run([sys.executable, *args], check=True)

def main():
    run("scripts/split_data.py")
    for version in ("v1", "v2"):
        for split in ("train", "validation"):
            run(
                "scripts/prepare_data.py",
                "--input", f"data/{split}.csv",
                "--output", f"data/iris_{version}_{split}.jsonl",
                "--version", version,
            )

if __name__ == "__main__":
    main()
