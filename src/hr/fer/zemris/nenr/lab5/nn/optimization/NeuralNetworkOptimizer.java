package hr.fer.zemris.nenr.lab5.nn.optimization;

import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;

/**
 * Interface for neural network optimizer.
 *
 * Created by luka on 12/18/16.
 */
public interface NeuralNetworkOptimizer {

    /**
     * Optimizes neural network with backpropagation algorithm and returns trained neural network.
     */
    NeuralNetwork optimize();
}
