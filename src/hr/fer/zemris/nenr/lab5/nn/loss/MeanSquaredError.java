package hr.fer.zemris.nenr.lab5.nn.loss;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Implements mean squared error loss function for neural network.
 *
 * Created by luka on 12/16/16.
 */
public class MeanSquaredError implements Loss {

    @Override
    public double forward(Matrix inputs, Matrix outputs) {
        assert outputs.isVector() || outputs.isScalar();
        int n = inputs.getHeight();
        int loss = 0;
        double h;
        double y;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                h = inputs.getElement(i, j);
                y = outputs.getElement(i, 0);
                loss += (1./(2*n)) * Math.pow(h - y, 2);
            }
        }
        return loss;
    }

    @Override
    public Matrix backwardsInputs(Matrix inputs, Matrix outputs) {
        assert outputs.isVector() || outputs.isScalar();
        int n = inputs.getHeight();
        double h;
        double y;
        double gradient;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                h = inputs.getElement(i, j);
                y = outputs.getElement(i, 0);
                gradient = (1./n) * (h - y);
                inputs.setElement(i, j, gradient);
            }
        }
        return inputs;
    }
}
