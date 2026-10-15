## Contents

This directory contains the reproducibility materials for Assignment 1:

- `matrix_multiply.c`: straightforward dense matrix multiplication in C.
- `MatrixMultiply.java`: straightforward dense matrix multiplication in Java.
- `matrix_multiply.py`: straightforward dense matrix multiplication in Python.
- `benchmark.py`: input generation, correctness checks, benchmarking and raw-result
  collection.
- `generate_graphics.py`: analysis script that generates the figures from
  `summary.csv`.
- `data/`: shared input matrices used by the three implementations.
- `results.csv`: raw benchmark measurements.
- `summary.csv`: median and variability summaries.
- `environment.json`: dependency, hardware and experimental configuration.
- `runtime.svg` and `memory.svg`: figures generated from the benchmark summary.

The implementations use the required straightforward triple-loop algorithm and
are compared using the same logical input matrices.

## Dependencies and environment

- 64-bit Windows.
- Python 3.12 or later.
- JDK 21 or later, providing `java` and `javac`.
- Microsoft Visual Studio C/C++ build tools, providing `vcvars64.bat` and
  `cl.exe`.
- No third-party Python packages.

The exact processor, RAM, operating system, Python version, Java version, C
compiler version, compiler flags and runtime settings used for the recorded
measurements are stored in `environment.json`.

## Build and run order

Run all commands from:

```text
C:\Users\Javier Guerra\Individual-Assigments-1
```

### 1. Compile C first

Open PowerShell in the assignment directory and run:

```powershell
$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" `
  -Filter vcvars64.bat `
  -Recurse `
  -ErrorAction SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName

cmd.exe /d /c "call `"$vcvars`" && cd /d `"C:\Users\Javier Guerra\Individual-Assigments-1`" && cl /nologo /Od /W4 /std:c11 matrix_multiply.c"
```

This creates `matrix_multiply.exe`. The `/Od` flag disables C optimisation,
`/W4` enables the warning level used for the study and `/std:c11` selects C11
support.

If the project is in another directory, replace the path after `cd /d`.

### 2. Compile Java

From the same directory, run:

```powershell
javac MatrixMultiply.java
```

Python does not require a separate compilation step.

### 3. Run correctness tests

Run the self-tests before the benchmark:

```powershell
python matrix_multiply.py --self-test
java MatrixMultiply --self-test
.\matrix_multiply.exe --self-test
```

The tests include known-answer, identity and zero-matrix cases.

### 4. Generate inputs and run the benchmark

Run:

```powershell
python benchmark.py
```

This generates or regenerates the shared input files in `data/`, validates the
three implementations against a trusted reference, measures time and memory,
and writes:

- `results.csv` with every raw measurement;
- `summary.csv` with the median and variability for each configuration;
- `environment.json` with the environment and experiment settings.

### 5. Generate the figures

After the benchmark finishes, run:

```powershell
python generate_graphics.py
```

This reads `summary.csv` and writes:

- `runtime.svg`: runtime by matrix size, with labelled units;
- `memory.svg`: peak process memory by matrix size, with labelled units.

## Reproducing the recorded inputs

The benchmark generates two shared matrices for each tested matrix size using
the recorded seed and value range in `environment.json`. The generated files
are stored in `data/` and are read by the Python, Java and C implementations.
The same logical inputs are therefore used for every language comparison.

## Reused materials and versions

The study reuses no external source code, datasets or figures. The only
external reference is the course specification:
