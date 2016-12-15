package hr.fer.zemris.nenr.lab5.nn.layers;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Implements fully connected layer of neural network.
 *
 * Created by luka on 12/15/16.
 */
public class FullyConnectedLayer implements Layer {

    private Matrix weights;
    private Matrix bias;
    private Matrix inputs;

    @Override
    public Matrix forward(Matrix inputs) {
        this.inputs = inputs;
        return inputs.dot(weights.getTransposed());
    }

    @Override
    public Matrix backwardInputs(Matrix grads) {
        return grads.dot(weights);
    }

    @Override
    public boolean hasParams() {
        return true;
    }

    @Override
    public GradsHolder backwardParams(Matrix grads) {
        Matrix weightGrads = grads.getTransposed().dot(inputs);
        Matrix biasGrads = grads.sum(Matrix.MatrixAxis.HORIZONTAL);
        return new GradsHolder(weights, weightGrads, bias, biasGrads);
    }
}
