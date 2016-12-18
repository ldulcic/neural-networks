package hr.fer.zemris.nenr.lab5.nn;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
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
public class FeedForwardNeuralNetwork implements NeuralNetwork {

    private List<Layer> layers;

    private FeedForwardNeuralNetwork(Builder builder) {
        this.layers = builder.layers;
    }

    public Matrix forward(Matrix inputs) {
        for (Layer layer : layers) {
            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    @Override
    public List<GradsHolder> backward(Matrix grads) {
        List<GradsHolder> params = new ArrayList<>(layers.size());
        for (int j = layers.size() - 1; j >= 0; --j) {
            Layer layer = layers.get(j);
            if (layer.hasParams()) {
                params.add(layer.backwardParams(grads));
            }
            grads = layer.backwardInputs(grads);
        }
        return params;
    }

    public static class Builder {

        List<Layer> layers = new ArrayList<>();

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

        public FeedForwardNeuralNetwork build() {
            if (layers.isEmpty()) {
                throw new IllegalArgumentException("Neural network must contain at least one layer.");
            }
            return new FeedForwardNeuralNetwork(this);
        }
    }
}
