from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def multiply(a: list[list[float]], b: list[list[float]], c: list[list[float]]) -> None:
    n = len(a)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total


def read_dataset(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    values = [float(token) for token in path.read_text(encoding="ascii").split()]
    n = int(values[0])
    if len(values) != 1 + 2 * n * n:
        raise ValueError(f"{path}: expected {1 + 2 * n * n} values, found {len(values)}")
    split = 1 + n * n
    a_values = values[1:split]
    b_values = values[split:]
    a = [a_values[row * n : (row + 1) * n] for row in range(n)]
    b = [b_values[row * n : (row + 1) * n] for row in range(n)]
    return a, b


def write_result(path: Path, c: list[list[float]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as output:
        for row in c:
            output.write(" ".join(format(value, ".17g") for value in row) + "\n")


def self_test() -> None:
    cases = [
        (
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[19.0, 22.0], [43.0, 50.0]],
        ),
        ([[1.0, 0.0], [0.0, 1.0]], [[2.0, -3.0], [4.0, 5.0]], [[2.0, -3.0], [4.0, 5.0]]),
        ([[0.0, 0.0], [0.0, 0.0]], [[2.0, -3.0], [4.0, 5.0]], [[0.0, 0.0], [0.0, 0.0]]),
    ]
    for a, b, expected in cases:
        c = [[0.0] * len(a) for _ in a]
        multiply(a, b, c)
        if c != expected:
            raise AssertionError(f"self-test failed: expected {expected}, received {c}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")

    a, b = read_dataset(args.input)
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    for _ in range(args.warmups):
        multiply(a, b, c)

    samples = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        for _ in range(args.batch):
            multiply(a, b, c)
        samples.append((time.perf_counter() - start) / args.batch)

    write_result(args.output, c)
    checksum = sum(sum(row) for row in c)
    print(json.dumps({"times_seconds": samples, "checksum": checksum, "batch": args.batch}))


if __name__ == "__main__":
    main()