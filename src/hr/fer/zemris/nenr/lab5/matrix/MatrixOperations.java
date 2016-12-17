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
     * Adds two matrices in-place, result is stored in <code>first</code> matrix.
     *
     * @throws MatrixUtil.IncompatibleMatricesException If matrices <code>first</code> and <code>second</code> don't
     *                                                  have same dimensions.
     */
    public static void addMatricesInPlace(Matrix first, Matrix second) throws MatrixUtil.IncompatibleMatricesException {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        for (int i = 0; i < first.getHeight(); ++i) {
            for (int j = 0; j < first.getWidth(); ++j) {
                first.setElement(i, j, first.getElement(i, j) + second.getElement(i, j));
            }
        }
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
     * Subtracts two matrices in-place, result is stored in <code>first</code> matrix.
     *
     * @throws MatrixUtil.IncompatibleMatricesException If matrices <code>first</code> and <code>second</code> don't
     *                                                  have same dimensions.
     */
    public static void subtractMatricesInPlace(Matrix first, Matrix second) throws MatrixUtil.IncompatibleMatricesException {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        for (int i = 0; i < first.getHeight(); ++i) {
            for (int j = 0; j < first.getWidth(); ++j) {
                first.setElement(i, j, first.getElement(i, j) - second.getElement(i, j));
            }
        }
    }

    /**
     * Multiplies matrix with scalar. This method changes <code>matrix</code> in-place.
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

    /**
     * Performs element-wise multiplication of matrices <code>first</code> and <code>second</code>.
     *
     * @return New <code>{@link Matrix}</code> which contains result of this operation.
     */
    public static Matrix convolve(Matrix first, Matrix second) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        Matrix result = new Matrix(first.getWidth(), first.getHeight());
        double tmp;
        for (int i = 0; i < first.getHeight(); i++) {
            for (int j = 0; j < first.getWidth(); j++) {
                tmp = first.getElement(i, j) * second.getElement(i, j);
                result.setElement(i, j, tmp);
            }
        }
        return result;
    }

    /**
     * Performs element-wise multiplication of matrices <code>first</code> and <code>second</code>.
     * WARNING this method performs in-place convolution and stores result of this operation in matrix <code>first</code>.
     *
     * @return <code>{@link Matrix}</code> which contains result of this operation. Return result is in fact reference
     *          to matrix <code>first</code> which contains result.
     */
    public static Matrix convolveInPlace(Matrix first, Matrix second) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(first, second);
        double tmp;
        for (int i = 0; i < first.getHeight(); i++) {
            for (int j = 0; j < first.getWidth(); j++) {
                tmp = first.getElement(i, j) * second.getElement(i, j);
                first.setElement(i, j, tmp);
            }
        }
        return first;
    }

    /**
     * Calculates Frobenius matrix norm. More details https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm.
     *
     * @param matrix <code>{@link Matrix}</code> who's norm we will compute
     * @return Norm value of <code>matrix</code>.
     */
    public static double norm(Matrix matrix) {
        double norm = 0;
        for (int i = 0; i < matrix.getHeight(); i++) {
            for (int j = 0; j < matrix.getWidth(); j++) {
                norm += Math.pow(matrix.getElement(i, j), 2);
            }
        }
        return Math.sqrt(norm);
    }

    /**
     * Performs <code>operation</code> on every element of <code>matrix</code> and stores result in <code>matrix</code>
     * in-place. WARNING this method changes <code>matrix</code>.
     *
     * @param matrix <code>{@link Matrix}</code> on which we apply <code>operation</code>
     * @param operation <code>{@link ElementOperation}</code> which we apply to every element of matrix.
     * @return Reference on <code>matrix</code> for convenience.
     */
    public static Matrix elementwiseOperation(Matrix matrix, ElementOperation operation) {
        double element;
        for (int i = 0; i < matrix.getHeight(); i++) {
            for (int j = 0; j < matrix.getWidth(); j++) {
                element = matrix.getElement(i, j);
                matrix.setElement(i, j, operation.transform(element));
            }
        }
        return matrix;
    }

    /**
     * Interface for applying operations on matrix elements.
     */
    public interface ElementOperation {
        /**
         * Applies operation tranformation on element <code>x</code> and returns result of operation.
         */
        double transform(double x);
    }
}
