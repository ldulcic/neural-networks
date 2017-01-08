package hr.fer.zemris.nenr.lab5.nn.layers;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.util.ParametersInitializer;

import java.io.Serializable;

/**
 * Implements fully connected layer of neural network.
 *
 * Created by luka on 12/15/16.
 */
public class FullyConnectedLayer implements Layer, Serializable {

    private Matrix weights;
    private Matrix bias;
    private Matrix inputs;

    public FullyConnectedLayer(int numOfInputs, int numOfNeurons) {
        weights = ParametersInitializer.initializeWeights(numOfNeurons, numOfInputs);
        bias = ParametersInitializer.initializeBias(numOfNeurons);
    }

    @Override
    public Matrix forward(Matrix inputs) {
        this.inputs = inputs.clone();
        return inputs.dot(weights.getTransposed()).add(getBiasMatrix(inputs.getHeight()));
    }

    /**
     * Broadcast bias vector into matrix so it can be added to result of input weights dot product.
     */
    private Matrix getBiasMatrix(int n) {
        double[][] biasMatrix = new double[n][bias.getWidth()];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < bias.getWidth(); j++) {
                biasMatrix[i][j] = bias.getElement(0, j);
            }
        }
        return new Matrix(biasMatrix);
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
        Matrix biasGrads = grads.sum(Matrix.MatrixAxis.VERTICAL);
        return new GradsHolder(weights, weightGrads, bias, biasGrads);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBias() {
        return bias;
    }
}
