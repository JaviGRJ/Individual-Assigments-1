from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_SIZES = [32, 64, 96, 128, 192, 256]
DEFAULT_SEED = 20261014

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
    """Run built-in correctness tests in all implementations."""
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
    """Write one small deterministic dataset for comparison."""
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
    """Read an n-by-n matrix from a text file."""
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

    for i, (actual_row, expected_row) in enumerate(
        zip(actual, expected)
    ):
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
                    f"Expected {expected_value}, "
                    f"received {actual_value}"
                )


def compare_program_outputs(c_executable: Path) -> None:
    """Run all implementations with the same small input."""
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

        python_result = read_matrix(python_output, 2)
        java_result = read_matrix(java_output, 2)
        c_result = read_matrix(c_output, 2)

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


def create_matrix(
    n: int,
    generator: random.Random,
) -> list[list[float]]:
    """Create an n-by-n matrix with values between -8 and 8."""
    return [
        [
            float(generator.randint(-8, 8))
            for _ in range(n)
        ]
        for _ in range(n)
    ]


def write_matrix(
    output,
    matrix: list[list[float]],
) -> None:
    """Write one matrix as whitespace-separated rows."""
    for row in matrix:
        output.write(
            " ".join(str(int(value)) for value in row)
            + "\n"
        )


def generate_dataset(
    output_path: Path,
    n: int,
    seed: int,
) -> None:
    """Generate and save two deterministic matrices."""
    generator = random.Random(seed)

    matrix_a = create_matrix(n, generator)
    matrix_b = create_matrix(n, generator)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as output:
        output.write(f"{n}\n")
        write_matrix(output, matrix_a)
        write_matrix(output, matrix_b)

    print(f"Generated dataset: {output_path}")


def generate_datasets(
    sizes: list[int],
    seed: int,
) -> list[Path]:
    """Generate one shared dataset for every matrix size."""
    data_directory = ROOT / "data"
    dataset_paths = []

    for n in sizes:
        dataset_path = data_directory / f"n_{n}.txt"

        generate_dataset(
            dataset_path,
            n,
            seed,
        )

        dataset_paths.append(dataset_path)

    return dataset_paths


def read_dataset(
    path: Path,
) -> tuple[int, list[list[float]], list[list[float]]]:
    """Read the size and two matrices from a dataset file."""
    values = [
        float(token)
        for token in path.read_text(
            encoding="ascii"
        ).split()
    ]

    if not values:
        raise ValueError(
            f"{path}: dataset is empty"
        )

    n = int(values[0])
    expected_value_count = 1 + 2 * n * n

    if n < 1:
        raise ValueError(
            f"{path}: matrix size must be positive"
        )

    if len(values) != expected_value_count:
        raise ValueError(
            f"{path}: expected {expected_value_count} values, "
            f"found {len(values)}"
        )

    matrix_a_values = values[1 : 1 + n * n]
    matrix_b_values = values[1 + n * n :]

    matrix_a = [
        matrix_a_values[row * n : (row + 1) * n]
        for row in range(n)
    ]

    matrix_b = [
        matrix_b_values[row * n : (row + 1) * n]
        for row in range(n)
    ]

    return n, matrix_a, matrix_b


def reference_multiply(
    matrix_a: list[list[float]],
    matrix_b: list[list[float]],
) -> list[list[float]]:
    """Calculate the reference matrix product in Python."""
    n = len(matrix_a)

    result = [
        [0.0 for _ in range(n)]
        for _ in range(n)
    ]

    for i in range(n):
        for j in range(n):
            total = 0.0

            for k in range(n):
                total += matrix_a[i][k] * matrix_b[k][j]

            result[i][j] = total

    return result


def validate_result_file(
    path: Path,
    expected: list[list[float]],
    label: str,
) -> None:
    """Validate every value written by one implementation."""
    n = len(expected)
    actual = read_matrix(path, n)

    compare_matrices(
        actual,
        expected,
        label,
    )


def run_dataset_program(
    language: str,
    input_path: Path,
    output_path: Path,
    c_executable: Path,
) -> None:
    """Run one implementation with a generated dataset."""
    arguments = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--warmups",
        "0",
        "--repetitions",
        "1",
        "--batch",
        "1",
    ]

    if language == "Python":
        command = [
            sys.executable,
            "matrix_multiply.py",
            *arguments,
        ]
    elif language == "Java":
        command = [
            "java",
            "MatrixMultiply",
            *arguments,
        ]
    elif language == "C":
        command = [
            str(c_executable),
            *arguments,
        ]
    else:
        raise ValueError(
            f"unknown language: {language}"
        )

    run_command(
        command,
        f"{language} validation run",
    )


def compare_generated_datasets(
    sizes: list[int],
    c_executable: Path,
) -> None:
    """Validate every generated dataset."""
    data_directory = ROOT / "data"

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary_directory = Path(temporary_name)

        for n in sizes:
            input_path = data_directory / f"n_{n}.txt"

            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Dataset does not exist: {input_path}"
                )

            dataset_n, matrix_a, matrix_b = read_dataset(
                input_path
            )

            if dataset_n != n:
                raise ValueError(
                    f"{input_path}: expected size {n}, "
                    f"found {dataset_n}"
                )

            print(f"Validating dataset n={n}...")

            expected = reference_multiply(
                matrix_a,
                matrix_b,
            )

            for language in ("Python", "Java", "C"):
                output_path = (
                    temporary_directory
                    / f"{language.lower()}_{n}.txt"
                )

                run_dataset_program(
                    language,
                    input_path,
                    output_path,
                    c_executable,
                )

                validate_result_file(
                    output_path,
                    expected,
                    f"{language} result for n={n}",
                )

            print(
                f"Dataset n={n}: "
                "all implementations match"
            )


def main() -> None:
    """Generate datasets and validate all implementations."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used to generate shared datasets.",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Matrix sizes to generate and validate.",
    )

    args = parser.parse_args()

    if any(size < 2 for size in args.sizes):
        parser.error(
            "all matrix sizes must be at least 2"
        )

    compile_java_if_needed()

    c_executable = find_c_executable()

    generate_datasets(
        args.sizes,
        args.seed,
    )

    run_self_tests(c_executable)

    compare_program_outputs(c_executable)

    compare_generated_datasets(
        args.sizes,
        c_executable,
    )

    print("All correctness checks passed.")
    print(
        "All generated datasets passed "
        "cross-language validation."
    )


if __name__ == "__main__":
    main()