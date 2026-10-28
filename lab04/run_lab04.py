
from pathlib import Path
import subprocess
import sys


LAB_DIR = Path(__file__).resolve().parent
SRC_DIR = LAB_DIR / "src"


def run_step(script_name):
    script_path = SRC_DIR / script_name

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=LAB_DIR.parent,
        check=True,
    )


def main():
    run_step("main.py")
    run_step("label_quality.py")
    run_step("finalize.py")

    print("\nLab 4 pipeline completed successfully.")


if __name__ == "__main__":
    main()
