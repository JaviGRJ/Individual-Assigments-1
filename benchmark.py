from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent

DEFAULT_SIZES = [32, 64, 96, 128, 192, 256]
DEFAULT_SEED = 20261014
DEFAULT_REPETITIONS = 5

BATCHES = {
    32: 16,
    64: 4,
    96: 2,
}

WARMUPS = {
    "Python": 1,
    "Java": 20,
    "C": 1,
}

LANGUAGES = ("Python", "Java", "C")

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
    """Compile Java if its class file is missing."""
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
    """Run the built-in tests in Python, Java and C."""
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
    """Write the small known-answer comparison dataset."""
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
    """Run Python, Java and C with the same small input."""
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
    """Create an n-by-n matrix with values from -8 to 8."""
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
    """Generate one shared dataset for each matrix size."""
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
    """Read the size and two matrices from a dataset."""
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
    """Calculate a reference matrix product in Python."""
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
    """Validate every value written by an implementation."""
    n = len(expected)
    actual = read_matrix(path, n)

    compare_matrices(
        actual,
        expected,
        label,
    )


def build_program_command(
    language: str,
    input_path: Path,
    output_path: Path,
    c_executable: Path,
    warmups: int,
    repetitions: int,
    batch: int,
) -> list[str]:
    """Build the command used to run one implementation."""
    arguments = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--warmups",
        str(warmups),
        "--repetitions",
        str(repetitions),
        "--batch",
        str(batch),
    ]

    if language == "Python":
        return [
            sys.executable,
            "matrix_multiply.py",
            *arguments,
        ]

    if language == "Java":
        return [
            "java",
            "MatrixMultiply",
            *arguments,
        ]

    if language == "C":
        return [
            str(c_executable),
            *arguments,
        ]

    raise ValueError(
        f"unknown language: {language}"
    )


def run_dataset_program(
    language: str,
    input_path: Path,
    output_path: Path,
    c_executable: Path,
) -> None:
    """Run one implementation for correctness validation."""
    command = build_program_command(
        language=language,
        input_path=input_path,
        output_path=output_path,
        c_executable=c_executable,
        warmups=0,
        repetitions=1,
        batch=1,
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

            for language in LANGUAGES:
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


def run_timed_program(
    language: str,
    input_path: Path,
    output_path: Path,
    c_executable: Path,
    repetitions: int,
) -> dict[str, object]:
    """Run one implementation and return its timing information."""
    n = int(input_path.stem.split("_")[1])
    warmups = WARMUPS[language]
    batch = BATCHES.get(n, 1)

    command = build_program_command(
        language=language,
        input_path=input_path,
        output_path=output_path,
        c_executable=c_executable,
        warmups=warmups,
        repetitions=repetitions,
        batch=batch,
    )

    print(
        f"Benchmarking {language}, "
        f"n={n}, "
        f"batch={batch}..."
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = completed.stdout.strip().splitlines()

    if not output_lines:
        raise RuntimeError(
            f"{language}: program produced no JSON output"
        )

    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{language}: could not parse timing output:\n"
            f"{completed.stdout}"
        ) from error

    times = [
        float(value)
        for value in payload["times_seconds"]
    ]

    if len(times) != repetitions:
        raise ValueError(
            f"{language}: expected {repetitions} timing samples, "
            f"received {len(times)}"
        )

    if any(value <= 0 for value in times):
        raise ValueError(
            f"{language}: timing samples must be positive"
        )

    return {
        "times_seconds": times,
        "checksum": float(payload["checksum"]),
        "batch": int(payload["batch"]),
    }


def write_results_csv(
    rows: list[dict[str, str]],
) -> Path:
    """Write raw timing samples to results.csv."""
    output_path = ROOT / "results.csv"

    fieldnames = [
        "language",
        "n",
        "sample_index",
        "batch_products",
        "time_seconds",
        "result_checksum",
        "status",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def benchmark_all_datasets(
    sizes: list[int],
    c_executable: Path,
    repetitions: int,
) -> list[dict[str, str]]:
    """Benchmark every language and size."""
    data_directory = ROOT / "data"
    rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary_directory = Path(temporary_name)

        for n in sizes:
            input_path = data_directory / f"n_{n}.txt"

            dataset_n, matrix_a, matrix_b = read_dataset(
                input_path
            )

            if dataset_n != n:
                raise ValueError(
                    f"{input_path}: expected size {n}, "
                    f"found {dataset_n}"
                )

            expected = reference_multiply(
                matrix_a,
                matrix_b,
            )

            for language in LANGUAGES:
                output_path = (
                    temporary_directory
                    / f"{language.lower()}_timed_{n}.txt"
                )

                payload = run_timed_program(
                    language=language,
                    input_path=input_path,
                    output_path=output_path,
                    c_executable=c_executable,
                    repetitions=repetitions,
                )

                validate_result_file(
                    output_path,
                    expected,
                    f"{language} timed result for n={n}",
                )

                times = payload["times_seconds"]
                checksum = float(payload["checksum"])
                batch = int(payload["batch"])

                for index, value in enumerate(times, start=1):
                    rows.append(
                        {
                            "language": language,
                            "n": str(n),
                            "sample_index": str(index),
                            "batch_products": str(batch),
                            "time_seconds": format(
                                float(value),
                                ".17g",
                            ),
                            "result_checksum": format(
                                checksum,
                                ".17g",
                            ),
                            "status": "ok",
                        }
                    )

    rows.sort(
        key=lambda row: (
            int(row["n"]),
            LANGUAGES.index(row["language"]),
            int(row["sample_index"]),
        )
    )

    results_path = write_results_csv(rows)

    print(
        f"Raw timing results written to: "
        f"{results_path}"
    )
    print(
        f"Recorded {len(rows)} timing samples."
    )

    return rows


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    """Calculate a linearly interpolated percentile."""
    if not values:
        raise ValueError(
            "cannot calculate a percentile from no values"
        )

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "percentile fraction must be between 0 and 1"
        )

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]

    weight = position - lower_index

    return lower_value + (
        upper_value - lower_value
    ) * weight


def calculate_iqr(
    values: list[float],
) -> float:
    """Calculate the interquartile range."""
    first_quartile = percentile(
        values,
        0.25,
    )

    third_quartile = percentile(
        values,
        0.75,
    )

    return third_quartile - first_quartile


def write_summary_csv(
    rows: list[dict[str, str]],
) -> Path:
    """Write median and IQR statistics to summary.csv."""
    grouped: dict[tuple[str, int], list[float]] = {}

    for row in rows:
        if row["status"] != "ok":
            continue

        key = (
            row["language"],
            int(row["n"]),
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            float(row["time_seconds"])
        )

    sizes = sorted(
        {
            int(row["n"])
            for row in rows
        }
    )

    summary_rows: list[dict[str, str]] = []

    for n in sizes:
        for language in LANGUAGES:
            values = grouped.get(
                (language, n),
                [],
            )

            if not values:
                summary_rows.append(
                    {
                        "language": language,
                        "n": str(n),
                        "samples": "0",
                        "median_seconds": "",
                        "median_milliseconds": "",
                        "iqr_seconds": "",
                        "iqr_milliseconds": "",
                        "status": "no successful samples",
                    }
                )
                continue

            median_seconds = median(values)
            iqr_seconds = calculate_iqr(values)

            summary_rows.append(
                {
                    "language": language,
                    "n": str(n),
                    "samples": str(len(values)),
                    "median_seconds": format(
                        median_seconds,
                        ".17g",
                    ),
                    "median_milliseconds": format(
                        median_seconds * 1000.0,
                        ".17g",
                    ),
                    "iqr_seconds": format(
                        iqr_seconds,
                        ".17g",
                    ),
                    "iqr_milliseconds": format(
                        iqr_seconds * 1000.0,
                        ".17g",
                    ),
                    "status": "ok",
                }
            )

    summary_path = ROOT / "summary.csv"

    fieldnames = [
        "language",
        "n",
        "samples",
        "median_seconds",
        "median_milliseconds",
        "iqr_seconds",
        "iqr_milliseconds",
        "status",
    ]

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path


def print_summary(
    summary_path: Path,
) -> None:
    """Print summary statistics to the console."""
    print(
        f"Summary written to: "
        f"{summary_path}"
    )
    print()
    print(
        "Language  n      Median (ms)      IQR (ms)      Samples"
    )
    print(
        "--------  -----  ---------------  ------------  -------"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            if row["status"] != "ok":
                print(
                    f"{row['language']:8} "
                    f"{row['n']:5} "
                    f"{'no data':15} "
                    f"{'no data':12} "
                    f"{row['samples']:7}"
                )
                continue

            median_text = (
                f"{float(row['median_milliseconds']):.4f}"
            )

            iqr_text = (
                f"{float(row['iqr_milliseconds']):.4f}"
            )

            print(
                f"{row['language']:8} "
                f"{row['n']:5} "
                f"{median_text:15} "
                f"{iqr_text:12} "
                f"{row['samples']:7}"
            )


def main() -> None:
    """Generate datasets, validate and benchmark."""
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
        help="Matrix sizes to generate and benchmark.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Measured repetitions per language and size.",
    )

    args = parser.parse_args()

    if any(size < 2 for size in args.sizes):
        parser.error(
            "all matrix sizes must be at least 2"
        )

    if args.repetitions < 5:
        parser.error(
            "at least five repetitions are required"
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

    rows = benchmark_all_datasets(
        sizes=args.sizes,
        c_executable=c_executable,
        repetitions=args.repetitions,
    )

    summary_path = write_summary_csv(rows)
    print_summary(summary_path)

    print("All correctness checks passed.")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()