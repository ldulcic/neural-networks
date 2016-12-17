package hr.fer.zemris.nenr.lab5.matrix;

/**
 * Simple helper class which stores weights, biases and its respective gradient of neural net layer and offers API for
 * performing weights and bias update. This class is used for mentioned data until we do backward pass of neural net,
 * after that we update weights and biases.
 *
 * Created by luka on 12/15/16.
 */
public class GradsHolder {

    private Matrix weights;
    private Matrix weightGrads;
    private Matrix bias;
    private Matrix biasGrads;

    public GradsHolder(Matrix weights, Matrix weightGrads, Matrix bias, Matrix biasGrads) {
        this.weights = weights;
        this.weightGrads = weightGrads;
        this.bias = bias;
        this.biasGrads = biasGrads;
    }

    public void performParameterUpdates() {
        //TODO implement proper learning rate
        weightGrads.multiply(0.01);
        biasGrads.multiply(0.01);
        weights.subtract(weightGrads);
        bias.subtract(biasGrads);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getWeightGrads() {
        return weightGrads;
    }

    public Matrix getBias() {
        return bias;
    }

    public Matrix getBiasGrads() {
        return biasGrads;
    }
}
