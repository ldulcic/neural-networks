package hr.fer.zemris.nenr.lab5.nn;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.layers.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Implements feed-forward neural network with backpropagation learning algorithm.
 *
 * Created by luka on 12/15/16.
 */
public class NeuralNetwork {

    private List<Layer> layers;
    private Layer loss;

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

    public static class Builder {

        List<Layer> layers = new ArrayList<>();
        Layer loss;

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

        public Builder loss(Layer layer) {
            Objects.requireNonNull(layer);
            this.loss = loss;
            return this;
        }

        public NeuralNetwork build() {
            if (layers.isEmpty()) {
                throw new IllegalArgumentException("Neural network must contain at least one layer.");
            }
            return new NeuralNetwork(this);
        }
    }
}
