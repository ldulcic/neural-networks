package hr.fer.zemris.nenr.lab5.nn.util;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.matrix.MatrixOperations;
import hr.fer.zemris.nenr.lab5.matrix.MatrixUtil;
import hr.fer.zemris.nenr.lab5.nn.layers.FullyConnectedLayer;
import hr.fer.zemris.nenr.lab5.nn.layers.Layer;
import hr.fer.zemris.nenr.lab5.nn.layers.SigmoidLayer;
import hr.fer.zemris.nenr.lab5.nn.loss.MeanSquaredError;
import hr.fer.zemris.nenr.lab5.nn.loss.SoftmaxWithCrossEntropyLoss;

/**
 * Util class for performing gradient check for neural network layers.
 *
 * Created by luka on 12/17/16.
 */
public class GradientCheck {

    private static final double h = 1E-5;

    private static double relativeError(Matrix X, Matrix Y) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(X, Y);
        Matrix errors = new Matrix(X.getWidth(), X.getHeight());
        double x;
        double y;
        double err;
        for (int i = 0; i < X.getHeight(); i++) {
            for (int j = 0; j < X.getWidth(); j++) {
                x = X.getElement(i, j);
                y = Y.getElement(i, j);
                err = Math.abs(x - y) / ( Math.max(1E-8, Math.abs(x) + Math.abs(y)) );
                errors.setElement(i, j, err);
            }
        }
        return errors.max();
    }

    private static Matrix evaluateNumericalGradient(Function f, Matrix x, Matrix df) {
        Matrix grads = new Matrix(x.getWidth(), x.getHeight());
        for (int i = 0; i < x.getHeight(); i++) {
            for (int j = 0; j < x.getWidth(); j++) {
                double oldX = x.getElement(i, j);
                x.setElement(i, j, oldX + h);
                Matrix pos = f.valueAt(x);
                x.setElement(i, j, oldX - h);
                Matrix neg = f.valueAt(x);
                x.setElement(i, j, oldX);

                double gradient = pos.subtract(neg).convolve(df).sum() / (2*h);
                grads.setElement(i, j, gradient);
            }
        }
        return grads;
    }

    private static void checkGradsInputs(Layer layer, Matrix x, Matrix gradsOut) {
        layer.forward(x);
        Matrix grad = layer.backwardInputs(gradsOut.clone());
        Matrix gradNum = evaluateNumericalGradient(layer::forward, x, gradsOut.clone());
        System.out.println("Relative error = " + relativeError(grad, gradNum));
        System.out.println("Error norm = " + MatrixOperations.norm(grad.subtract(gradNum)));
    }

    private static void checkGradsParams(Layer layer, Matrix x, Matrix w, Matrix bias, Matrix gradsOut) {
        Function func = x1 -> layer.forward(x);
        Matrix gradsWeightsNum = evaluateNumericalGradient(func, w, gradsOut.clone());
        Matrix gradsBiasNum = evaluateNumericalGradient(func, bias, gradsOut.clone());
        layer.forward(x);
        GradsHolder gradsHolder = layer.backwardParams(gradsOut);

        System.out.println("Check weights:");
        System.out.println("Relative error = " + relativeError(gradsHolder.getWeightGrads(), gradsWeightsNum));
        System.out.println("Error norm = " + MatrixOperations.norm(gradsHolder.getWeightGrads().subtract(gradsWeightsNum)));
        System.out.println("Check biases:");
        System.out.println("Relative error = " + relativeError(gradsHolder.getBiasGrads(), gradsBiasNum));
        System.out.println("Error norm = " + MatrixOperations.norm(gradsHolder.getBiasGrads().subtract(gradsBiasNum)));
    }

    public static void main(String[] args) {
        Matrix x;
        Matrix gradsOut;

        /** FULLY CONNECTED */
        System.out.println("FULLY CONNECTED LAYER:\n");
        x = MatrixUtil.randomNormalMatrix(20, 40);
        gradsOut = MatrixUtil.randomNormalMatrix(20, 30);
        FullyConnectedLayer fc = new FullyConnectedLayer(40, 30);
        System.out.println("Check grads wrt inputs");
        checkGradsInputs(fc, x, gradsOut);
        System.out.println("\nCheck grads wrt params");
        checkGradsParams(fc, x, fc.getWeights(), fc.getBias(), gradsOut);
        /** fully connected */

        /** SIGMOID */
        System.out.println("\n\nSIGMOID LAYER:\n");
        x = MatrixUtil.randomNormalMatrix(20, 40);
        gradsOut = MatrixUtil.randomNormalMatrix(20, 40);
        SigmoidLayer sigmoid = new SigmoidLayer();
        checkGradsInputs(sigmoid, x, gradsOut);
        /** sigmoid */

        /** MSE LOSS */
        System.out.println("\n\nMSE LOSS:\n");
        x = MatrixUtil.randomNormalMatrix(20, 40);
        final Matrix mseY = MatrixUtil.randomNormalMatrix(20, 1);
        MeanSquaredError mse = new MeanSquaredError();
        Function func = x1 -> {
            Matrix matrix = new Matrix(1, 1);
            matrix.setElement(0, 0, mse.forward(x1, mseY));
            return matrix;
        };
        Matrix grad = mse.backwardsInputs(x, mseY);
        Matrix gradNum = evaluateNumericalGradient(func, x, MatrixUtil.singleValueMatrix(1, 1, 1));
        System.out.println("Relative error = " + relativeError(grad, gradNum));
        System.out.println("Error norm = " + MatrixOperations.norm(grad.subtract(gradNum)));
        /** mse loss */

        /** CROSS ENTROPY LOSS */
        System.out.println("\n\nSOFTMAX-CROSS-ENTROPY LOSS:\n");
        x = MatrixUtil.randomNormalMatrix(5, 5);
        final Matrix sceY = MatrixUtil.randomOneHotMatrix(5, 5);
        SoftmaxWithCrossEntropyLoss sce = new SoftmaxWithCrossEntropyLoss();
        func = x1 -> {
            Matrix matrix = new Matrix(1, 1);
            matrix.setElement(0, 0, sce.forward(x1, sceY));
            return matrix;
        };
        grad = sce.backwardsInputs(x, sceY);
        gradNum = evaluateNumericalGradient(func, x, MatrixUtil.singleValueMatrix(1, 1, 1));
        System.out.println("Relative error = " + relativeError(grad, gradNum));
        System.out.println("Error norm = " + MatrixOperations.norm(grad.subtract(gradNum)));
        /** cross entropy loss */
    }

    interface Function {
        Matrix valueAt(Matrix x);
    }
}
