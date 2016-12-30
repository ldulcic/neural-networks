package hr.fer.zemris.nenr.lab5.matrix;

import java.util.Random;

/**
 * Util class for handling <code>Matrix</code>.
 *
 * Created by luka on 10/22/16.
 */
public class MatrixUtil {

    private static Random random = new Random();

    /**
     * Checks if matrix is square.
     *
     * @param matrix Matrix which will be checked.
     * @throws MatrixNotSquareException If matrix is not square.
     */
    public static void checkIsMatrixSquare(Matrix matrix) throws MatrixNotSquareException {
        if (matrix.getHeight() != matrix.getWidth()) {
            throw new MatrixNotSquareException();
        }
    }

    /**
     * Checks whether matrices <code>first</code> and <code>second</code> have same dimensions.
     *
     * @throws IncompatibleMatricesException If matrices don't have same dimensions.
     */
    public static void checkIfMatricesHaveSameDimensions(Matrix first, Matrix second) throws IncompatibleMatricesException {
        if (first.getHeight() != second.getHeight() || first.getWidth() != second.getWidth()) {
            throw new IncompatibleMatricesException("Matrices must have same dimensions.");
        }
    }

    /**
     * Generates matrix of dimensions [<code>height</code>, <code>width</code>] with elements initialised from
     * normal distribution with zero mean and unit variance.
     *
     * @return New <code>{@link Matrix}</code> initialized from normal distribution.
     */
    public static Matrix randomNormalMatrix(int height, int width) {
        Matrix matrix = new Matrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix.setElement(i, j, random.nextGaussian());
            }
        }
        return matrix;
    }

    public static Matrix randomOneHotMatrix(int height, int width) {
        Matrix matrix = new Matrix(width, height);
        int randIndex;
        for (int i = 0; i < height; i++) {
            randIndex = random.nextInt(width);
            matrix.setElement(i, randIndex, 1);
        }
        return matrix;
    }

    /**
     * Generates matrix of dimensions [<code>height</code>, <code>width</code>], where every element of matrix has value <code>value</code>.
     *
     * @param height Height of matrix.
     * @param width Width of matrix.
     * @param value Value at which all elements in matrix will be set.
     */
    public static Matrix singleValueMatrix(int height, int width, double value) {
        Matrix matrix = new Matrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix.setElement(i, j, value);
            }
        }
        return matrix;
    }

    /**
     * Indicates that matrices are incompatible for multiplication.
     */
    public static class IncompatibleMatricesException extends RuntimeException {
        IncompatibleMatricesException(String message) {
            super(message);
        }
    }

    /**
     * Indicates that matrix is not square.
     */
    public static class MatrixNotSquareException extends RuntimeException {
        MatrixNotSquareException() {
            super("Matrix is not square");
        }
    }
}
