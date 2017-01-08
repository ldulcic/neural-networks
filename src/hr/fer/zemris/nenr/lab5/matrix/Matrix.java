package hr.fer.zemris.nenr.lab5.matrix;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Represents two-dimensional array (matrix) which contains double values as basic elements.
 * <p>
 * Created by luka on 10/21/16.
 */
public class Matrix implements Serializable {

    public enum MatrixAxis {HORIZONTAL, VERTICAL}

    private double[][] elements;
    private int width;
    private int height;

    /**
     * Initializes matrix of dimension [width, height] with all elements set to 0.
     */
    public Matrix(int width, int height) {
        this.elements = new double[height][width];
        this.width = width;
        this.height = height;
    }

    public Matrix(double[][] elements) {
        if (elements.length == 0) {
            throw new IllegalArgumentException("Matrix must contain at least one row!");
        }
        this.elements = elements;
        this.width = elements[0].length;
        this.height = elements.length;
    }

    public boolean isVector() {
        return width == 1 || height == 1;
    }

    public boolean isScalar() {
        return width == 1 && height == 1;
    }

    /**
     * @param row Index of row where element is.
     * @param column Index of column where element is.
     * @return element Element on index [row, column]
     * @throws IndexOutOfBoundsException row >= matrix.height or column >= matrix.width.
     */
    public double getElement(int row, int column) throws IndexOutOfBoundsException {
        checkIfIndicesValid(row, column);
        return elements[row][column];
    }

    public double[] getRow(int index) {
        checkIfRowValid(index);
        return elements[index];
    }

    /**
     * Sets value of matrix element at index [row, column].
     *
     * @param row Index of row.
     * @param column Index of column.
     * @param element Element to be set at [row, column].
     * @throws IndexOutOfBoundsException row >= matrix.height or column >= matrix.width.
     */
    public void setElement(int row, int column, double element) throws IndexOutOfBoundsException {
        checkIfIndicesValid(row, column);
        elements[row][column] = element;
    }

    /**
     * Performs multiplication (dot product) of this matrix with <code>matrix</code>.
     * This matrix is left multiplier and <code>matrix</code> is right multiplier.
     *
     * @param matrix Matrix with which we multiply this matrix.
     * @return Matrix result of multiplication.
     */
    public Matrix dot(Matrix matrix) {
        return MatrixOperations.multiply(this, matrix);
    }

    /**
     * Performs element-wise multiplication of matrix values (convolution). WARNING this method changes matrix.
     *
     * @param matrix Matrix with which we convolve this matrix.
     * @return Matrix result of convolution (this matrix).
     */
    public Matrix convolve(Matrix matrix) {
        return MatrixOperations.convolveInPlace(this, matrix);
    }

    /**
     * Adds this matrix with <code>matrix</code>. Results is store in this matrix. WARNING this method changes matrix.
     *
     * @param matrix Matrix with which we add.
     * @return this matrix for convenience.
     */
    public Matrix add(Matrix matrix) {
        MatrixOperations.addMatricesInPlace(this, matrix);
        return this;
    }

    /**
     * Subtracts this matrix with <code>matrix</code>. Results is store in this matrix. WARNING this method changes
     * matrix.
     *
     * @param matrix Matrix with which we subtract.
     * @return this matrix for convenience.
     */
    public Matrix subtract(Matrix matrix) {
        MatrixOperations.subtractMatricesInPlace(this, matrix);
        return this;
    }

    /**
     * Multiplies this matrix with scalar in-place. WARNING this method changes matrix.
     *
     * @param scalar Scalar with which we multiply matrix.
     * @return this matrix for convenience.
     */
    public Matrix multiply(double scalar) {
        MatrixOperations.multiplyWithScalar(this, scalar);
        return this;
    }

    /**
     * Return maximum value of this matrix.
     */
    public double max() {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (elements[i][j] > max) {
                    max = elements[i][j];
                }
            }
        }
        return max;
    }

    /**
     * Sums all elements in matrix.
     *
     * @return Sum of all elements in matrix.
     */
    public double sum() {
        double sum = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                sum += elements[i][j];
            }
        }
        return sum;
    }

    /**
     * Sums elements of matrix based on <code>axis</code> parameter.
     * If <code>axis</code> is <code>HORIZONTAL</code> this method returns vectors who's elements contain sum of every
     * row in this matrix.
     * If <code>axis</code> is <code>VERTICAL</code> this method returns vectors who's elements contain sum of every
     * column in this matrix.
     *
     * @param axis Parameter which defines how to perform summation of matrix elements. Possible values are HORIZONTAL
     *             and VERTICAL.
     * @return <code>Matrix</code> which is result of summation of matrix element. In case of horizontal summation
     *          result is column-vector, and in case of vertical summation result is row-vector.
     */
    public Matrix sum(MatrixAxis axis) {
        Objects.requireNonNull(axis);
        return axis == MatrixAxis.HORIZONTAL ? sumHorizontal() : sumVertical();
    }

    /**
     * @return Vector who's elements contain sums of every row in this matrix.
     */
    @SuppressWarnings("Duplicates")
    private Matrix sumHorizontal() {
        Matrix horizontalSum = new Matrix(1, height);
        double rowSum;
        for (int i = 0; i < height; i++) {
            rowSum = 0;
            for (int j = 0; j < width; j++) {
                rowSum += elements[i][j];
            }
            horizontalSum.setElement(i, 0, rowSum);
        }
        return horizontalSum;
    }

    /**
     * @return Vector who's elements contain sums of every column in this matrix.
     */
    @SuppressWarnings("Duplicates")
    private Matrix sumVertical() {
        Matrix verticalSum = new Matrix(width, 1);
        double columnSum;
        for (int i = 0; i < width; i++) {
            columnSum = 0;
            for (int j = 0; j < height; j++) {
                columnSum += elements[j][i];
            }
            verticalSum.setElement(0, i, columnSum);
        }
        return verticalSum;
    }

    /**
     * Swaps <code>row1</code> and <code>row2</code>
     *
     * @param row1 Index of row1.
     * @param row2 Index of row2.
     * @throws IndexOutOfBoundsException row1 >= matrix.height or row2 >= matrix.height.
     */
    public void swapRows(int row1, int row2) throws IndexOutOfBoundsException {
        checkIfRowValid(row1);
        checkIfRowValid(row2);
        if (row1 == row2) {
            return;
        }
        double[] tmp = elements[row1];
        elements[row1] = elements[row2];
        elements[row2] = tmp;
    }

    /**
     * Transposes matrix in-place. WARNING this method changes matrix.
     */
    public void transpose() {
        double[][] transposed = new double[width][height];
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                transposed[i][j] = elements[j][i];
            }
        }
        int tmp = width;
        width = height;
        height = tmp;
        elements = transposed;
    }

    /**
     * Creates new matrix which transpose of this matrix. This method doesn't change the original matrix.
     *
     * @return Matrix which is transpose of this matrix.
     */
    public Matrix getTransposed() {
        double[][] transposed = new double[width][height];
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                transposed[i][j] = elements[j][i];
            }
        }
        return new Matrix(transposed);
    }

    /**
     * Writes matrix into file specified by path. Matrix rows are separated by new line character and column elements
     * are separated by space character.
     * Example 3x3 matrix in file:
     *      1 2 3
     *      4 5 6
     *      7 8 9
     *
     * @param path Path to file where matrix should be written, if file doesn't exist it will be created.
     * @throws IOException If file doesn't exist or some other IO error occur while writing to file.
     */
    public void writeToFile(String path) throws IOException {
        FileWriter fw = new FileWriter(new File(path));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sb.append(elements[i][j]).append(' ');
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append(System.lineSeparator());
        }
        fw.write(sb.toString());
        fw.flush();
        fw.close();
    }

    private void checkIfIndicesValid(int row, int column) {
        checkIfRowValid(row);
        checkIfColumnValid(column);
    }

    private void checkIfRowValid(int row) {
        if (row >= height) {
            throw new IndexOutOfBoundsException(String.format("Row %d >= height(%d)", row, height));
        }
    }

    private void checkIfColumnValid(int column) {
        if (column >= width) {
            throw new IndexOutOfBoundsException(String.format("Column %d >= width(%d)", column, width));
        }
    }

    /**
     * @return Number of columns in matrix.
     */
    public int getWidth() {
        return width;
    }

    /**
     * @return Number of rows in matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * @return New matrix with exact same elements as this one.
     */
    public Matrix clone() {
        double[][] elements = new double[height][width];
        for (int i = 0; i < height; ++i) {
            elements[i] = Arrays.copyOf(this.elements[i], this.elements[i].length);
        }
        return new Matrix(elements);
    }

    /**
     * Reads matrix from file. This method expects that each row is written in single line where values are separated
     * by space character. Rows are separated by new line character.
     * Example 3x3 matrix in file:
     *      1 2 3
     *      4 5 6
     *      7 8 9
     *
     * @param path Path to file where matrix is.
     * @return <code>Matrix</code> object which contains matrix from file in <code>path</code>.
     * @throws IOException If file doesn't exist or some other IO error occur.
     */
    public static Matrix fromFile(String path) throws IOException {
        List<String> lines = Files.readAllLines(new File(path).toPath());
        if (lines == null || lines.isEmpty()) {
            throw new RuntimeException("Matrix must contain at least one row.");
        }
        int height = lines.size();
        int width = lines.get(0).trim().split(" ").length;
        double[][] elements = new double[height][width];
        String[] parts;
        for (int i = 0; i < height; ++i) {
            String line = lines.get(i);
            parts = line.trim().split(" ");
            for (int j = 0; j < width; ++j) {
                elements[i][j] = Double.parseDouble(parts[j]);
            }
        }
        return new Matrix(elements);
    }

    /**
     * @param size Size of identity matrix.
     * @return Identity matrix of dimensions [size, size] which has ones on diagonal and zeros everywhere else.
     */
    public static Matrix identityMatrix(int size) throws IllegalArgumentException {
        if (size <= 0) {
            throw new IllegalArgumentException("size must be greater than 0.");
        }
        Matrix identity = new Matrix(size, size);
        for (int i = 0; i < size; ++i) {
            identity.setElement(i, i, 1);
        }
        return identity;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof Matrix) {
            Matrix matrix = (Matrix) obj;
            if (matrix.width == this.width && matrix.height == this.height) {
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        if (!MathUtils.equals(matrix.getElement(i, j), elements[i][j])) {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sb.append(String.format("%-20f", elements[i][j]));
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }
}
