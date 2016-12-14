package hr.fer.zemris.nenr.lab5.matrix;

/**
 * Util class for applying operations on matrix.
 * <p>
 * Created by luka on 10/21/16.
 */
public class MatrixOperations {

    /**
     * Adds two matrices.
     *
     * @return Matrix which is result of addition of matrices <code>first</code> and <code>second</code>.
     * @throws MatrixUtil.IncompatibleMatricesException If matrices <code>first</code> and <code>second</code> don't
     *                                                  have same dimensions.
     */
    public static Matrix addMatrices(Matrix first, Matrix second) throws MatrixUtil.IncompatibleMatricesException {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        Matrix addition = new Matrix(first.getWidth(), first.getHeight());
        for (int i = 0; i < first.getHeight(); ++i) {
            for (int j = 0; j < first.getWidth(); ++j) {
                addition.setElement(i, j, first.getElement(i, j) + second.getElement(i, j));
            }
        }
        return addition;
    }

    /**
     * Subtracts two matrices.
     *
     * @return Matrix which is result of subtraction of matrices <code>first</code> and <code>second</code>.
     * @throws MatrixUtil.IncompatibleMatricesException If matrices <code>first</code> and <code>second</code> don't
     *                                                  have same dimensions.
     */
    public static Matrix subtractMatrices(Matrix first, Matrix second) throws MatrixUtil.IncompatibleMatricesException {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        Matrix subtraction = new Matrix(first.getWidth(), first.getHeight());
        for (int i = 0; i < first.getHeight(); ++i) {
            for (int j = 0; j < first.getWidth(); ++j) {
                subtraction.setElement(i, j, first.getElement(i, j) - second.getElement(i, j));
            }
        }
        return subtraction;
    }

    /**
     * Multiplies matrix with scalar. THis method changes <code>matrix</code> in-place.
     *
     * @param matrix Matrix to be multiplied.
     * @param scalar Scalar with which we multiply.
     */
    public static void multiplyWithScalar(Matrix matrix, double scalar) {
        for (int i = 0; i < matrix.getHeight(); ++i) {
            for (int j = 0; j < matrix.getWidth(); ++j) {
                matrix.setElement(i, j, matrix.getElement(i, j) * scalar);
            }
        }
    }


    /**
     * Multiplies two matrices and new matrix as result.
     *
     * @param first  matrix of dimensions m x n
     * @param second matrix of dimensions n x k
     * @return Result of matrix multiplication. Dimension of resulting matrix is m x k.
     * @throws MatrixUtil.IncompatibleMatricesException If first matrix number of columns doesn't match second
     *                                                  matrix number of rows.
     */
    public static Matrix multiply(Matrix first, Matrix second) throws MatrixUtil.IncompatibleMatricesException {
        if (first.getWidth() != second.getHeight()) {
            throw new MatrixUtil.IncompatibleMatricesException("First matrix number of columns must match second matrix number of rows.");
        }
        Matrix result = new Matrix(second.getWidth(), first.getHeight());
        for (int i = 0; i < first.getHeight(); ++i) {
            for (int j = 0; j < second.getWidth(); ++j) {
                for (int k = 0; k < second.getHeight(); ++k) {
                    double value = first.getElement(i, k) * second.getElement(k, j);
                    result.setElement(i, j, result.getElement(i, j) + value);
                }
            }
        }
        return result;
    }
}
