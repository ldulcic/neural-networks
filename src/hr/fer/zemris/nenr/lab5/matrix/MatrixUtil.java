package hr.fer.zemris.nenr.lab5.matrix;
/**
 * Util class for handling <code>Matrix</code>.
 *
 * Created by luka on 10/22/16.
 */
public class MatrixUtil {

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

    public static void checkIfMatricesHaveSameDimensions(Matrix first, Matrix second) {
        if (first.getHeight() != second.getHeight() || first.getWidth() != second.getWidth()) {
            throw new IncompatibleMatricesException("Matrices must have same dimensions.");
        }
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
