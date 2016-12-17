package hr.fer.zemris.nenr.lab5.nn;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.layers.Layer;
import hr.fer.zemris.nenr.lab5.nn.loss.Loss;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Implements feed-forward neural network with backpropagation learning algorithm.
 *
 * Created by luka on 12/15/16.
 */
public class NeuralNetwork {

    private List<Layer> layers;
    private Loss loss;

    private NeuralNetwork(Builder builder) {
        this.layers = builder.layers;
        this.loss = builder.loss;
    }

    public Matrix forward(Matrix input) {
        for (Layer layer : layers) {
            input = layer.forward(input);
        }
        return input;
    }

    public void train(Matrix inputs, Matrix outputs) {
        List<GradsHolder> params = new ArrayList<>(layers.size());
        for (int i = 0; i < 100000; i++) {
            System.out.printf("Current loss: %f\n", loss.forward(forward(inputs), outputs));
            Matrix grads = loss.backwardsInputs(forward(inputs), outputs);
            for (int j = layers.size() - 1; j >= 0; --j) {
                Layer layer = layers.get(j);
                if (layer.hasParams()) {
                    params.add(layer.backwardParams(grads));
                }
                grads = layer.backwardInputs(grads);
            }
            for (GradsHolder param : params) {
                param.performParameterUpdates();
            }
            params.clear();
        }
    }

    public static class Builder {

        List<Layer> layers = new ArrayList<>();
        Loss loss;

        public Builder layers(List<Layer> layers) {
            Objects.requireNonNull(layers);
            this.layers = layers;
            return this;
        }

        public Builder addLayer(Layer layer) {
            Objects.requireNonNull(layer);
            this.layers.add(layer);
            return this;
        }

        public Builder loss(Loss loss) {
            Objects.requireNonNull(loss);
            this.loss = loss;
            return this;
        }

        public NeuralNetwork build() {
            Objects.requireNonNull(loss, "Loss function must be defined.");
            if (layers.isEmpty()) {
                throw new IllegalArgumentException("Neural network must contain at least one layer.");
            }
            return new NeuralNetwork(this);
        }
    }
}
