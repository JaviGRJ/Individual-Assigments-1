from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import csv
import datetime as datetime_module
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent

DEFAULT_SIZES = [32, 64, 96, 128, 192, 256]
DEFAULT_SEED = 20261014
DEFAULT_REPETITIONS = 5
DEFAULT_TIME_BUDGET_SECONDS = 60.0
DEFAULT_MEMORY_BUDGET_MIB = 2048

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
MEMORY_METRIC = "peak_process_working_set_bytes"


class ProcessMemory:
    """Read the peak working set of a Windows process."""

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("page_fault_count", ctypes.wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
            ("quota_nonpaged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError(
                "Peak working-set measurement requires Windows."
            )

        self.kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        self.psapi = ctypes.WinDLL(
            "psapi",
            use_last_error=True,
        )

        self.kernel32.OpenProcess.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        self.kernel32.OpenProcess.restype = (
            ctypes.wintypes.HANDLE
        )

        self.kernel32.CloseHandle.argtypes = [
            ctypes.wintypes.HANDLE,
        ]
        self.kernel32.CloseHandle.restype = (
            ctypes.wintypes.BOOL
        )

        self.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(self.Counters),
            ctypes.wintypes.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = (
            ctypes.wintypes.BOOL
        )

    def open(self, process_id: int) -> int:
        """Open a process handle for memory inspection."""
        process_query_information = 0x0400
        process_vm_read = 0x0010
        access = process_query_information | process_vm_read

        handle = self.kernel32.OpenProcess(
            access,
            False,
            process_id,
        )

        if not handle:
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return handle

    def peak(self, handle: int) -> int:
        """Return peak working-set memory in bytes."""
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)

        success = self.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )

        if not success:
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return int(counters.peak_working_set_size)

    def close(self, handle: int) -> None:
        """Close a process memory handle."""
        if not self.kernel32.CloseHandle(handle):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )


def run_command(
    command: list[str],
    label: str,
) -> None:
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
            f"Could not find the program required for "
            f"{label}: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{error.returncode}"
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


def run_self_tests(
    c_executable: Path,
) -> None:
    """Run built-in tests in Python, Java and C."""
    run_command(
        [
            sys.executable,
            "matrix_multiply.py",
            "--self-test",
        ],
        "Python self-test",
    )

    run_command(
        [
            "java",
            "MatrixMultiply",
            "--self-test",
        ],
        "Java self-test",
    )

    run_command(
        [
            str(c_executable),
            "--self-test",
        ],
        "C self-test",
    )


def write_dataset(path: Path) -> None:
    """Write a small known-answer comparison dataset."""
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


def read_matrix(
    path: Path,
    n: int,
) -> list[list[float]]:
    """Read an n-by-n matrix from a text file."""
    values = [
        float(token)
        for token in path.read_text(
            encoding="ascii"
        ).split()
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
    """Compare two matrices using numeric tolerances."""
    if len(actual) != len(expected):
        raise AssertionError(
            f"{label}: different number of rows"
        )

    for i, (actual_row, expected_row) in enumerate(
        zip(actual, expected)
    ):
        if len(actual_row) != len(expected_row):
            raise AssertionError(
                f"{label}: different number of columns "
                f"in row {i}"
            )

        for j, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            if not math.isfinite(actual_value):
                raise AssertionError(
                    f"{label}: value at [{i}][{j}] "
                    "is not finite"
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


def compare_program_outputs(
    c_executable: Path,
) -> None:
    """Run Python, Java and C with the same small input."""
    expected = [
        [19.0, 22.0],
        [43.0, 50.0],
    ]

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary_directory = Path(temporary_name)

        input_path = temporary_directory / "input.txt"
        python_output = (
            temporary_directory / "python_output.txt"
        )
        java_output = (
            temporary_directory / "java_output.txt"
        )
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

        python_result = read_matrix(
            python_output,
            2,
        )
        java_result = read_matrix(
            java_output,
            2,
        )
        c_result = read_matrix(
            c_output,
            2,
        )

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

    print(
        "All implementations produced "
        "the expected matrix."
    )
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
) -> tuple[
    int,
    list[list[float]],
    list[list[float]],
]:
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
    actual = read_matrix(
        path,
        len(expected),
    )

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
    """Run one implementation for validation."""
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
    memory: ProcessMemory,
    time_budget_seconds: float,
    memory_budget_mib: int,
) -> dict[str, object]:
    """Run one implementation while measuring time and memory."""
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

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    handle = memory.open(process.pid)
    peak_working_set = 0
    start_time = time.monotonic()
    memory_limit_bytes = (
        memory_budget_mib * 1024 * 1024
    )
    limit_error: str | None = None
    stdout = ""
    stderr = ""

    try:
        while process.poll() is None:
            peak_working_set = max(
                peak_working_set,
                memory.peak(handle),
            )

            elapsed = time.monotonic() - start_time

            if peak_working_set > memory_limit_bytes:
                limit_error = (
                    f"{language} n={n} exceeded the "
                    f"{memory_budget_mib} MiB memory limit"
                )
                process.terminate()
                break

            if elapsed > time_budget_seconds:
                limit_error = (
                    f"{language} n={n} exceeded the "
                    f"{time_budget_seconds} second time limit"
                )
                process.terminate()
                break

            time.sleep(0.005)

        try:
            stdout, stderr = process.communicate(
                timeout=5
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            limit_error = (
                f"{language} n={n} did not terminate "
                "after the limit"
            )

        peak_working_set = max(
            peak_working_set,
            memory.peak(handle),
        )

    finally:
        memory.close(handle)

    if limit_error is not None:
        raise RuntimeError(limit_error)

    if process.returncode != 0:
        raise RuntimeError(
            f"{language} n={n} failed with exit code "
            f"{process.returncode}.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    output_lines = stdout.strip().splitlines()

    if not output_lines:
        raise RuntimeError(
            f"{language}: program produced no JSON output"
        )

    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{language}: could not parse timing output:\n"
            f"{stdout}"
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
        "peak_working_set_bytes": peak_working_set,
    }


def write_results_csv(
    rows: list[dict[str, str]],
) -> Path:
    """Write raw timing and memory samples to results.csv."""
    output_path = ROOT / "results.csv"

    fieldnames = [
        "language",
        "n",
        "sample_index",
        "batch_products",
        "time_seconds",
        "peak_working_set_bytes",
        "memory_metric",
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
    memory: ProcessMemory,
    time_budget_seconds: float,
    memory_budget_mib: int,
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
                    memory=memory,
                    time_budget_seconds=time_budget_seconds,
                    memory_budget_mib=memory_budget_mib,
                )

                validate_result_file(
                    output_path,
                    expected,
                    f"{language} timed result for n={n}",
                )

                times = payload["times_seconds"]
                checksum = float(payload["checksum"])
                batch = int(payload["batch"])
                peak_working_set_bytes = int(
                    payload["peak_working_set_bytes"]
                )

                for index, value in enumerate(
                    times,
                    start=1,
                ):
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
                            "peak_working_set_bytes": str(
                                peak_working_set_bytes
                            ),
                            "memory_metric": MEMORY_METRIC,
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
    """Write timing and memory statistics to summary.csv."""
    grouped_times: dict[
        tuple[str, int],
        list[float],
    ] = {}

    grouped_memory: dict[
        tuple[str, int],
        list[int],
    ] = {}

    for row in rows:
        if row["status"] != "ok":
            continue

        key = (
            row["language"],
            int(row["n"]),
        )

        grouped_times.setdefault(
            key,
            [],
        ).append(
            float(row["time_seconds"])
        )

        grouped_memory.setdefault(
            key,
            [],
        ).append(
            int(row["peak_working_set_bytes"])
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
            key = (language, n)
            times = grouped_times.get(key, [])
            memory_values = grouped_memory.get(key, [])

            if not times:
                summary_rows.append(
                    {
                        "language": language,
                        "n": str(n),
                        "samples": "0",
                        "median_seconds": "",
                        "median_milliseconds": "",
                        "iqr_seconds": "",
                        "iqr_milliseconds": "",
                        "peak_working_set_bytes": "",
                        "peak_working_set_mib": "",
                        "memory_metric": MEMORY_METRIC,
                        "status": "no successful samples",
                    }
                )
                continue

            median_seconds = median(times)
            iqr_seconds = calculate_iqr(times)
            peak_working_set_bytes = max(memory_values)
            peak_working_set_mib = (
                peak_working_set_bytes
                / (1024 * 1024)
            )

            summary_rows.append(
                {
                    "language": language,
                    "n": str(n),
                    "samples": str(len(times)),
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
                    "peak_working_set_bytes": str(
                        peak_working_set_bytes
                    ),
                    "peak_working_set_mib": format(
                        peak_working_set_mib,
                        ".6f",
                    ),
                    "memory_metric": MEMORY_METRIC,
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
        "peak_working_set_bytes",
        "peak_working_set_mib",
        "memory_metric",
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
    """Print timing and memory statistics."""
    print(
        f"Summary written to: "
        f"{summary_path}"
    )
    print()
    print(
        "Language  n      Median (ms)   "
        "IQR (ms)    Peak (MiB)   Samples"
    )
    print(
        "--------  -----  ------------  "
        "----------  -----------  -------"
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
                    f"{'no data':12} "
                    f"{'no data':10} "
                    f"{'no data':11} "
                    f"{row['samples']:7}"
                )
                continue

            median_text = (
                f"{float(row['median_milliseconds']):.4f}"
            )
            iqr_text = (
                f"{float(row['iqr_milliseconds']):.4f}"
            )
            memory_text = (
                f"{float(row['peak_working_set_mib']):.2f}"
            )

            print(
                f"{row['language']:8} "
                f"{row['n']:5} "
                f"{median_text:12} "
                f"{iqr_text:10} "
                f"{memory_text:11} "
                f"{row['samples']:7}"
            )


def get_java_version() -> str:
    """Return the Java runtime version when available."""
    java_path = shutil.which("java")

    if java_path is None:
        return "not found"

    result = subprocess.run(
        [java_path, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    version_output = (
        result.stdout + result.stderr
    ).strip()

    return version_output or "unknown"


def get_javac_version() -> str:
    """Return the Java compiler version when available."""
    javac_path = shutil.which("javac")

    if javac_path is None:
        return "not found"

    result = subprocess.run(
        [javac_path, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    version_output = (
        result.stdout + result.stderr
    ).strip()

    return version_output or "unknown"


def get_total_memory_gib() -> float | None:
    """Return total physical memory on Windows in GiB."""
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.wintypes.DWORD),
            ("memory_load", ctypes.wintypes.DWORD),
            ("total_physical", ctypes.wintypes.ULARGE_INTEGER),
            ("available_physical", ctypes.wintypes.ULARGE_INTEGER),
            ("total_page_file", ctypes.wintypes.ULARGE_INTEGER),
            ("available_page_file", ctypes.wintypes.ULARGE_INTEGER),
            ("total_virtual", ctypes.wintypes.ULARGE_INTEGER),
            ("available_virtual", ctypes.wintypes.ULARGE_INTEGER),
            ("available_extended_virtual", ctypes.wintypes.ULARGE_INTEGER),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)

    kernel32 = ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    )

    if not kernel32.GlobalMemoryStatusEx(
        ctypes.byref(status)
    ):
        return None

    return status.total_physical / (1024 ** 3)


def write_environment(
    sizes: list[int],
    seed: int,
    repetitions: int,
    time_budget_seconds: float,
    memory_budget_mib: int,
    c_executable: Path,
) -> Path:
    """Write the experimental configuration to environment.json."""
    environment = {
        "run_timestamp": (
            datetime_module.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "operating_system": (
            f"{platform.system()} "
            f"{platform.release()} "
            f"{platform.version()} "
            f"{platform.machine()}"
        ),
        "python_version": sys.version.replace(
            "\n",
            " ",
        ),
        "java_runtime_version": get_java_version(),
        "java_compiler_version": get_javac_version(),
        "c_executable": str(c_executable),
        "c_compiler_flags": [
            "/nologo",
            "/Od",
            "/W4",
            "/std:c11",
        ],
        "seed": seed,
        "sizes": sizes,
        "repetitions": repetitions,
        "warmups": WARMUPS,
        "batch_products_per_sample": {
            str(n): BATCHES.get(n, 1)
            for n in sizes
        },
        "time_budget_seconds": time_budget_seconds,
        "memory_budget_mib": memory_budget_mib,
        "memory_metric": MEMORY_METRIC,
        "timing_boundary": (
            "multiplication loops only; "
            "loading, allocation, output and validation "
            "are outside the timed region"
        ),
        "threads": {
            "application_multiplication_threads": 1,
            "numerical_library_threads": 0,
            "affinity": (
                "OS scheduler default; "
                "no affinity pinning"
            ),
        },
        "hardware": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_processors": os.cpu_count(),
            "total_memory_gib": get_total_memory_gib(),
        },
    }

    environment_path = ROOT / "environment.json"

    environment_path.write_text(
        json.dumps(
            environment,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return environment_path


def main() -> None:
    """Generate datasets, validate, benchmark and summarize."""
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

    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
        help="Maximum time allowed per configuration.",
    )

    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=DEFAULT_MEMORY_BUDGET_MIB,
        help="Maximum peak working set per configuration.",
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

    if args.time_budget_seconds <= 0:
        parser.error(
            "the time budget must be positive"
        )

    if args.memory_budget_mib <= 0:
        parser.error(
            "the memory budget must be positive"
        )

    if os.name != "nt":
        parser.error(
            "memory measurement requires Windows"
        )

    compile_java_if_needed()

    c_executable = find_c_executable()
    memory = ProcessMemory()

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
        memory=memory,
        time_budget_seconds=args.time_budget_seconds,
        memory_budget_mib=args.memory_budget_mib,
    )

    summary_path = write_summary_csv(rows)
    print_summary(summary_path)

    environment_path = write_environment(
        sizes=args.sizes,
        seed=args.seed,
        repetitions=args.repetitions,
        time_budget_seconds=args.time_budget_seconds,
        memory_budget_mib=args.memory_budget_mib,
        c_executable=c_executable,
    )

    print(
        f"Environment written to: "
        f"{environment_path}"
    )
    print("All correctness checks passed.")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()