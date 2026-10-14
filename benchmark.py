from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], label: str) -> None:
    """Run one test command and stop if it fails."""
    print(f"Running {label}...")

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
        )
    except FileNotFoundError as error:
        executable = command[0]
        raise RuntimeError(
            f"Could not find the program required for {label}: {executable}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} failed with exit code {error.returncode}"
        ) from error

    if result.returncode == 0:
        print(f"{label}: OK")


def find_c_executable() -> Path:
    """Find the compiled C executable."""
    candidates = [
        ROOT / "matrix_multiply.exe",
        ROOT / "build" / "matrix_multiply.exe",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find matrix_multiply.exe. "
        "Compile matrix_multiply.c before running this benchmark."
    )


def main() -> None:
    """Run the self-tests for Python, Java and C."""
    if shutil.which("javac") is None:
        raise RuntimeError(
            "javac was not found. Open a terminal with a JDK available."
        )

    java_class = ROOT / "MatrixMultiply.class"

    if not java_class.is_file():
        print("Compiling Java...")
        subprocess.run(
            ["javac", "MatrixMultiply.java"],
            cwd=ROOT,
            check=True,
        )

    c_executable = find_c_executable()

    run_command(
        [sys.executable, "matrix_multiply.py", "--self-test"],
        "Python self-test",
    )

    run_command(
        ["java", "MatrixMultiply", "--self-test"],
        "Java self-test",
    )

    run_command(
        [str(c_executable), "--self-test"],
        "C self-test",
    )

    print("All tests passed correctly.")


if __name__ == "__main__":
    main()