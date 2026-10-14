import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class MatrixMultiply {
    private static void multiply(double[][] a, double[][] b, double[][] c) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double total = 0.0;
                for (int k = 0; k < n; k++) {
                    total += a[i][k] * b[k][j];
                }
                c[i][j] = total;
            }
        }
    }

    private static boolean equal(double actual, double expected) {
        double absoluteTolerance = 1e-12;
        double relativeTolerance = 1e-12;
        return Math.abs(actual - expected)
                <= absoluteTolerance + relativeTolerance * Math.abs(expected);
    }

    private static void selfTest() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] identity = {{1, 0}, {0, 1}};
        double[][] zero = {{0, 0}, {0, 0}};
        double[][] values = {{2, -3}, {4, 5}};
        double[][] result = new double[2][2];
        double[][] expected = {{19, 22}, {43, 50}};
        multiply(a, b, result);
        assertMatrix(result, expected, "known-answer");
        multiply(identity, values, result);
        assertMatrix(result, values, "identity");
        multiply(zero, values, result);
        assertMatrix(result, zero, "zero-matrix");
    }

    private static void assertMatrix(double[][] actual, double[][] expected, String label) {
        for (int i = 0; i < actual.length; i++) {
            for (int j = 0; j < actual.length; j++) {
                if (!equal(actual[i][j], expected[i][j])) {
                    throw new AssertionError(label + " self-test failed at [" + i + "][" + j + "]");
                }
            }
        }
    }

    private static double[][][] readDataset(Path path) throws IOException {
        List<Double> values = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.US_ASCII)) {
            String line;
            while ((line = reader.readLine()) != null) {
                for (String token : line.trim().split("\\s+")) {
                    if (!token.isEmpty()) {
                        values.add(Double.parseDouble(token));
                    }
                }
            }
        }
        int n = values.get(0).intValue();
        if (n < 1 || values.size() != 1 + 2 * n * n) {
            throw new IOException("invalid dataset dimensions or value count: " + path);
        }
        double[][] a = new double[n][n];
        double[][] b = new double[n][n];
        int index = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = values.get(index++);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = values.get(index++);
            }
        }
        return new double[][][] {a, b};
    }

    private static void writeResult(Path path, double[][] c) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.US_ASCII)) {
            for (double[] row : c) {
                for (int j = 0; j < row.length; j++) {
                    if (j > 0) {
                        writer.write(' ');
                    }
                    writer.write(Double.toString(row[j]));
                }
                writer.newLine();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("--self-test")) {
            selfTest();
            return;
        }
        if (args.length != 10
                || !args[0].equals("--input")
                || !args[2].equals("--output")
                || !args[4].equals("--warmups")
                || !args[6].equals("--repetitions")
                || !args[8].equals("--batch")) {
            throw new IllegalArgumentException(
                    "usage: MatrixMultiply --input FILE --output FILE --warmups N --repetitions N --batch N");
        }
        Path input = Path.of(args[1]);
        Path output = Path.of(args[3]);
        int warmups = Integer.parseInt(args[5]);
        int repetitions = Integer.parseInt(args[7]);
        int batch = Integer.parseInt(args[9]);
        if (warmups < 0 || repetitions < 1 || batch < 1) {
            throw new IllegalArgumentException("warmups must be nonnegative; repetitions and batch positive");
        }
        selfTest();
        double[][][] matrices = readDataset(input);
        double[][] a = matrices[0];
        double[][] b = matrices[1];
        int n = a.length;
        double[][] c = new double[n][n];

        for (int i = 0; i < warmups; i++) {
            multiply(a, b, c);
        }
        double[] samples = new double[repetitions];
        for (int i = 0; i < repetitions; i++) {
            long start = System.nanoTime();
            for (int j = 0; j < batch; j++) {
                multiply(a, b, c);
            }
            samples[i] = ((System.nanoTime() - start) / 1_000_000_000.0) / batch;
        }

        writeResult(output, c);
        double checksum = 0.0;
        for (double[] row : c) {
            for (double value : row) {
                checksum += value;
            }
        }
        StringBuilder json = new StringBuilder("{\"times_seconds\":[");
        for (int i = 0; i < samples.length; i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append(String.format(Locale.ROOT, "%.17g", samples[i]));
        }
        json.append("],\"checksum\":")
                .append(String.format(Locale.ROOT, "%.17g", checksum))
                .append(",\"batch\":")
                .append(batch)
                .append('}');
        System.out.println(json);
    }
}