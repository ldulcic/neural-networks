package hr.fer.zemris.nenr.lab5.nn.loss;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.matrix.MatrixUtil;

/**
 * Implements mean squared error loss function for neural network.
 *
 * Created by luka on 12/16/16.
 */
public class MeanSquaredError implements Loss {

    @Override
    public double forward(Matrix inputs, Matrix outputs) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(inputs, outputs);
        int n = inputs.getHeight();
        double loss = 0;
        double h;
        double y;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                h = inputs.getElement(i, j);
                y = outputs.getElement(i, j);
                loss += Math.pow(h - y, 2);
            }
        }
        return (1./(2*n)) * loss;
    }

    @Override
    public Matrix backwardsInputs(Matrix inputs, Matrix outputs) {
        MatrixUtil.checkIfMatricesHaveSameDimensions(inputs, outputs);
        Matrix grads = new Matrix(inputs.getWidth(), inputs.getHeight());
        int n = inputs.getHeight();
        double h;
        double y;
        double gradient;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                h = inputs.getElement(i, j);
                y = outputs.getElement(i, j);
                gradient =  h - y;
                grads.setElement(i, j, gradient);
            }
        }
        return grads.multiply(1./n);
    }
}
