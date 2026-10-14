"""Run correctness tests and compare matrix results in Python, Java and C."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12


def run_command(command: list[str], label: str) -> None:
    """Run one command and stop if it fails."""
    print(f"Running {label}...")

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not find the program required for {label}: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} failed with exit code {error.returncode}"
        ) from error

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
        "Compile matrix_multiply.c before running the benchmark."
    )


def compile_java_if_needed() -> None:
    """Compile the Java implementation if its class file is missing."""
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


def run_self_tests(c_executable: Path) -> None:
    """Run the built-in correctness tests in all three implementations."""
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


def write_dataset(path: Path) -> None:
    """Write one small deterministic dataset for the comparison test."""
    path.write_text(
        "\n".join(
            [
                "2",
                "1 2",
                "3 4",
                "5 6",
                "7 8",
                "",
            ]
        ),
        encoding="ascii",
    )


def read_matrix(path: Path, n: int) -> list[list[float]]:
    """Read an n-by-n result matrix from a text file."""
    values = [
        float(token)
        for token in path.read_text(encoding="ascii").split()
    ]

    expected_value_count = n * n
    if len(values) != expected_value_count:
        raise ValueError(
            f"{path}: expected {expected_value_count} values, "
            f"found {len(values)}"
        )

    return [
        values[row * n : (row + 1) * n]
        for row in range(n)
    ]


def compare_matrices(
    actual: list[list[float]],
    expected: list[list[float]],
    label: str,
) -> None:
    """Compare two matrices using absolute and relative tolerances."""
    if len(actual) != len(expected):
        raise AssertionError(
            f"{label}: different number of rows"
        )

    for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if len(actual_row) != len(expected_row):
            raise AssertionError(
                f"{label}: different number of columns in row {i}"
            )

        for j, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            if not math.isfinite(actual_value):
                raise AssertionError(
                    f"{label}: value at [{i}][{j}] is not finite"
                )

            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=RELATIVE_TOLERANCE,
                abs_tol=ABSOLUTE_TOLERANCE,
            ):
                raise AssertionError(
                    f"{label}: mismatch at [{i}][{j}]. "
                    f"Expected {expected_value}, received {actual_value}"
                )


def compare_program_outputs(c_executable: Path) -> None:
    """Run all implementations with the same input and compare outputs."""
    n = 2
    expected = [
        [19.0, 22.0],
        [43.0, 50.0],
    ]

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary_directory = Path(temporary_name)
        input_path = temporary_directory / "input.txt"

        python_output = temporary_directory / "python_output.txt"
        java_output = temporary_directory / "java_output.txt"
        c_output = temporary_directory / "c_output.txt"

        write_dataset(input_path)

        common_arguments = [
            "--input",
            str(input_path),
            "--output",
            "",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--batch",
            "1",
        ]

        python_arguments = common_arguments.copy()
        python_arguments[3] = str(python_output)
        run_command(
            [
                sys.executable,
                "matrix_multiply.py",
                *python_arguments,
            ],
            "Python comparison run",
        )

        java_arguments = common_arguments.copy()
        java_arguments[3] = str(java_output)
        run_command(
            [
                "java",
                "MatrixMultiply",
                *java_arguments,
            ],
            "Java comparison run",
        )

        c_arguments = common_arguments.copy()
        c_arguments[3] = str(c_output)
        run_command(
            [
                str(c_executable),
                *c_arguments,
            ],
            "C comparison run",
        )

        python_result = read_matrix(python_output, n)
        java_result = read_matrix(java_output, n)
        c_result = read_matrix(c_output, n)

        compare_matrices(
            python_result,
            expected,
            "Python result",
        )
        compare_matrices(
            java_result,
            expected,
            "Java result",
        )
        compare_matrices(
            c_result,
            expected,
            "C result",
        )

        compare_matrices(
            java_result,
            python_result,
            "Java versus Python",
        )
        compare_matrices(
            c_result,
            python_result,
            "C versus Python",
        )

    print("All implementations produced the expected matrix.")
    print("Python, Java and C results match.")


def main() -> None:
    """Run self-tests and compare the three implementations."""
    compile_java_if_needed()
    c_executable = find_c_executable()

    run_self_tests(c_executable)
    compare_program_outputs(c_executable)

    print("All correctness checks passed.")


if __name__ == "__main__":
    main()