package hr.fer.zemris.nenr.lab5.nn.loss;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.matrix.MatrixUtil;

/**
 * Implements cross entropy loss function. Inputs are first transformed with softmax to get probabilistic distribution
 * over outputs. This class assumes that class labels are one-hot encoded.
 *
 * Created by luka on 12/30/16.
 */
public class SoftmaxWithCrossEntropyLoss implements Loss {

    @Override
    public double forward(Matrix inputs, Matrix outputs) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(inputs, outputs);
        double loss = 0;
        double softmaxSum;
        for (int i = 0; i < inputs.getHeight(); i++) {
            softmaxSum = 0;
            for (int j = 0; j < inputs.getWidth(); j++) {
                softmaxSum += Math.exp(inputs.getElement(i, j));
            }
            loss += -Math.log( Math.exp(inputs.getElement(i, argmax(outputs, i))) / softmaxSum );
        }
        return loss;
    }

    /**
     * Finds index of element in row which has maximum value. This method assumes that <code>matrix</code> holds
     * one-hot vectors.
     */
    private int argmax(Matrix matrix, int row) {
        for (int i = 0; i < matrix.getWidth(); i++) {
            if (matrix.getElement(row, i) == 1) {
                return i;
            }
        }
        throw new IllegalStateException("matrix does not hold one-hot vectors!");
    }

    @Override
    public Matrix backwardsInputs(Matrix inputs, Matrix outputs) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(inputs, outputs);
        Matrix grads = new Matrix(inputs.getWidth(), inputs.getHeight());
        double softmaxSum;
        for (int i = 0; i < inputs.getHeight(); i++) {
            softmaxSum = 0;
            for (int j = 0; j < inputs.getWidth(); j++) {
                softmaxSum += Math.exp(inputs.getElement(i, j));
            }
            for (int j = 0; j < inputs.getWidth(); j++) {
                grads.setElement(i, j, Math.exp( inputs.getElement(i, j) ) / softmaxSum );
            }
            int rightIndex = argmax(outputs, i); //index of right class
            grads.setElement(i, rightIndex, grads.getElement(i, rightIndex) - 1);
        }
        return grads;
    }
}
