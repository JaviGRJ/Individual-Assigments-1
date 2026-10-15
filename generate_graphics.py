"""Generate runtime and memory figures from summary.csv."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.csv"
RUNTIME_PATH = ROOT / "runtime.svg"
MEMORY_PATH = ROOT / "memory.svg"

LANGUAGES = ("Python", "Java", "C")

COLORS = {
    "Python": "#2b6cb0",
    "Java": "#c05621",
    "C": "#2f855a",
}


def read_summary(
    path: Path,
) -> dict[str, dict[int, dict[str, float]]]:
    """Read successful timing and memory summaries."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find summary file: {path}"
        )

    data: dict[str, dict[int, dict[str, float]]] = {
        language: {}
        for language in LANGUAGES
    }

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "language",
            "n",
            "median_seconds",
            "peak_working_set_mib",
            "status",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "summary.csv is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            language = row["language"]
            n = int(row["n"])

            if language not in LANGUAGES:
                raise ValueError(
                    f"Unknown language in summary.csv: {language}"
                )

            if row["status"] != "ok":
                continue

            runtime_seconds = float(
                row["median_seconds"]
            )

            memory_mib = float(
                row["peak_working_set_mib"]
            )

            if runtime_seconds <= 0:
                raise ValueError(
                    f"Invalid runtime for {language}, n={n}"
                )

            if memory_mib <= 0:
                raise ValueError(
                    f"Invalid memory value for {language}, n={n}"
                )

            data[language][n] = {
                "runtime_seconds": runtime_seconds,
                "memory_mib": memory_mib,
            }

    if not any(data[language] for language in LANGUAGES):
        raise ValueError(
            "summary.csv has no successful measurements"
        )

    return data


def x_coordinate(
    n: int,
    minimum_n: int,
    maximum_n: int,
    left: float,
    chart_width: float,
) -> float:
    """Convert a matrix size to an SVG x coordinate."""
    denominator = max(
        1,
        maximum_n - minimum_n,
    )

    return left + chart_width * (
        (n - minimum_n) / denominator
    )


def y_coordinate(
    value: float,
    minimum_value: float,
    maximum_value: float,
    top: float,
    chart_height: float,
    logarithmic: bool,
) -> float:
    """Convert a value to an SVG y coordinate."""
    if logarithmic:
        transformed_value = math.log10(value)
        transformed_minimum = math.log10(
            minimum_value
        )
        transformed_maximum = math.log10(
            maximum_value
        )
    else:
        transformed_value = value
        transformed_minimum = minimum_value
        transformed_maximum = maximum_value

    denominator = (
        transformed_maximum
        - transformed_minimum
    )

    if denominator == 0:
        denominator = 1.0

    return top + chart_height * (
        (transformed_maximum - transformed_value)
        / denominator
    )


def create_chart(
    output_path: Path,
    title: str,
    y_label: str,
    data_key: str,
    y_suffix: str,
    logarithmic: bool,
) -> None:
    """Create one SVG chart from summary.csv."""
    data = read_summary(SUMMARY_PATH)

    width = 900
    height = 560

    left = 94
    right = 34
    top = 82
    bottom = 82

    chart_width = width - left - right
    chart_height = height - top - bottom

    points = [
        (
            n,
            values[data_key],
        )
        for language in LANGUAGES
        for n, values in data[language].items()
        if values[data_key] > 0
    ]

    if not points:
        raise ValueError(
            f"No positive values available for {data_key}"
        )

    minimum_n = min(
        n
        for n, _ in points
    )

    maximum_n = max(
        n
        for n, _ in points
    )

    raw_values = [
        value
        for _, value in points
    ]

    minimum_value = min(raw_values)
    maximum_value = max(raw_values)

    if minimum_value == maximum_value:
        minimum_value *= 0.5
        maximum_value *= 1.5

    axis_path = (
        f"M {left} {top} "
        f"V {top + chart_height} "
        f"H {left + chart_width}"
    )

    lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="34" '
            'text-anchor="middle" '
            'font-family="Arial, sans-serif" '
            'font-size="20">'
            f"{escape(title)}</text>"
        ),
        (
            f'<path d="{axis_path}" '
            'fill="none" stroke="#333333"/>'
        ),
        (
            f'<text x="{width / 2}" y="{height - 22}" '
            'text-anchor="middle" '
            'font-family="Arial, sans-serif" '
            'font-size="14">'
            "Matrix side length n</text>"
        ),
        (
            f'<text x="24" y="{height / 2}" '
            'text-anchor="middle" '
            f'transform="rotate(-90 24 {height / 2})" '
            'font-family="Arial, sans-serif" '
            'font-size="14">'
            f"{escape(y_label)}</text>"
        ),
    ]

    for tick in range(5):
        ratio = tick / 4
        y = top + chart_height * ratio

        if logarithmic:
            minimum_log = math.log10(minimum_value)
            maximum_log = math.log10(maximum_value)

            transformed = maximum_log - (
                maximum_log - minimum_log
            ) * ratio

            label_value = 10 ** transformed
        else:
            label_value = maximum_value - (
                maximum_value - minimum_value
            ) * ratio

        label = f"{label_value:.3g}{y_suffix}"

        grid_path = (
            f"M {left - 4} {y:.1f} "
            f"H {left + chart_width}"
        )

        lines.append(
            (
                f'<path d="{grid_path}" '
                'stroke="#dddddd"/>'
            )
        )

        lines.append(
            (
                f'<text x="{left - 10}" '
                f'y="{y + 4:.1f}" '
                'text-anchor="end" '
                'font-family="Arial, sans-serif" '
                'font-size="11">'
                f"{escape(label)}</text>"
            )
        )

    available_sizes = sorted(
        {
            n
            for language in LANGUAGES
            for n in data[language]
        }
    )

    for n in available_sizes:
        x = x_coordinate(
            n,
            minimum_n,
            maximum_n,
            left,
            chart_width,
        )

        tick_path = (
            f"M {x:.1f} "
            f"{top + chart_height} v 4"
        )

        lines.append(
            (
                f'<path d="{tick_path}" '
                'stroke="#333333"/>'
            )
        )

        lines.append(
            (
                f'<text x="{x:.1f}" '
                f'y="{top + chart_height + 21}" '
                'text-anchor="middle" '
                'font-family="Arial, sans-serif" '
                'font-size="11">'
                f"{n}</text>"
            )
        )

    for index, language in enumerate(LANGUAGES):
        series = sorted(
            data[language].items()
        )

        coordinates = []

        for n, values in series:
            x = x_coordinate(
                n,
                minimum_n,
                maximum_n,
                left,
                chart_width,
            )

            y = y_coordinate(
                values[data_key],
                minimum_value,
                maximum_value,
                top,
                chart_height,
                logarithmic,
            )

            coordinates.append(
                f"{x:.1f},{y:.1f}"
            )

        color = COLORS[language]

        if coordinates:
            polyline_points = " ".join(
                coordinates
            )

            lines.append(
                (
                    f'<polyline points="{polyline_points}" '
                    'fill="none" '
                    f'stroke="{color}" '
                    'stroke-width="2.5"/>'
                )
            )

            for n, values in series:
                x = x_coordinate(
                    n,
                    minimum_n,
                    maximum_n,
                    left,
                    chart_width,
                )

                y = y_coordinate(
                    values[data_key],
                    minimum_value,
                    maximum_value,
                    top,
                    chart_height,
                    logarithmic,
                )

                lines.append(
                    (
                        f'<circle cx="{x:.1f}" '
                        f'cy="{y:.1f}" r="4" '
                        f'fill="{color}"/>'
                    )
                )

        legend_x = left + 170 * index

        legend_path = (
            f"M {legend_x} 59 h 25"
        )

        lines.append(
            (
                f'<path d="{legend_path}" '
                f'stroke="{color}" '
                'stroke-width="3"/>'
            )
        )

        lines.append(
            (
                f'<text x="{legend_x + 32}" y="64" '
                'font-family="Arial, sans-serif" '
                'font-size="13">'
                f"{language}</text>"
            )
        )

    lines.append("</svg>")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Generated figure: {output_path}")


def main() -> None:
    """Generate runtime and memory figures."""
    create_chart(
        output_path=RUNTIME_PATH,
        title="Basic matrix multiplication runtime",
        y_label="Median time in seconds",
        data_key="runtime_seconds",
        y_suffix=" s",
        logarithmic=True,
    )

    create_chart(
        output_path=MEMORY_PATH,
        title="Peak process working set",
        y_label="Peak working set in MiB",
        data_key="memory_mib",
        y_suffix=" MiB",
        logarithmic=False,
    )

    print("Both figures generated successfully.")


if __name__ == "__main__":
    main()