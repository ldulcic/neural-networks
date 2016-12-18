package hr.fer.zemris.nenr.lab5.nn.optimization;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.loss.Loss;

import java.util.List;
import java.util.Objects;

/**
 * Implements standard backpropagation optimization of neural network. This method uses whole dataset in each
 * iteration of neural network backpropagation.
 *
 * Created by luka on 12/18/16.
 */
public class StandardBackpropagation implements NeuralNetworkOptimizer {

    private NeuralNetwork nn;
    private Loss loss;
    private Matrix inputs;
    private Matrix outputs;
    private int maxIter;
    private double minError;

    private StandardBackpropagation(Builder builder) {
        this.nn = builder.nn;
        this.loss = builder.loss;
        this.inputs = builder.inputs;
        this.outputs = builder.outputs;
        this.maxIter = builder.maxIter;
        this.minError = builder.minError;
    }

    @Override
    public NeuralNetwork optimize() {
        double currentLoss;
        Matrix nnOutput;
        Matrix lossGrads;
        List<GradsHolder> nnGrads;
        for (int i = 0; i < maxIter; i++) {
            nnOutput = nn.forward(inputs);
            currentLoss = loss.forward(nnOutput, outputs);

            if ((i + 1) % 100 == 0) {
                System.out.printf("Iter %d/%d, loss = ", i + 1, maxIter);
                System.out.println(currentLoss);
            }

            if (currentLoss < minError) {
                System.out.printf("Finished after %d/%d, loss = ", i + 1, maxIter);
                System.out.println(currentLoss);
                break;
            }

            lossGrads = loss.backwardsInputs(nnOutput, outputs);
            nnGrads = nn.backward(lossGrads);

            for (GradsHolder grads : nnGrads) {
                grads.performParameterUpdates();
            }
        }

        return nn;
    }

    public static class Builder {

        NeuralNetwork nn;
        Loss loss;
        Matrix inputs;
        Matrix outputs;

        int maxIter = Integer.MAX_VALUE;
        double minError = Double.MIN_VALUE;

        public Builder neuralNetwork(NeuralNetwork nn) {
            this.nn = nn;
            return this;
        }

        public Builder loss(Loss loss) {
            this.loss = loss;
            return this;
        }

        public Builder inputs(Matrix inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder outputs(Matrix outputs) {
            this.outputs = outputs;
            return this;
        }

        public Builder maxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }

        public Builder minError(double minError) {
            this.minError = minError;
            return this;
        }

        public StandardBackpropagation build() {
            Objects.requireNonNull(nn, "neural network == null");
            Objects.requireNonNull(loss, "loss == null");
            Objects.requireNonNull(inputs, "inputs == null");
            Objects.requireNonNull(outputs, "outputs == null");
            if (inputs.getHeight() != outputs.getHeight()) {
                throw new IllegalArgumentException("Number of examples in inputs and outputs don't match.");
            }
            if (!outputs.isVector()) {
                throw new IllegalArgumentException("Outputs matrix is expected to be column-vector.");
            }
            return new StandardBackpropagation(this);
        }
    }
}
