package hr.fer.zemris.nenr.lab5.nn;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;

import java.util.List;

/**
 * Interface for neural network.
 *
 * Created by luka on 12/18/16.
 */
public interface NeuralNetwork {

    /**
     * Performs forward pass of neural network.
     *
     * @param inputs Input data.
     * @return Result of neural network forward pass.
     */
    Matrix forward(Matrix inputs);

    /**
     * Performs backward pass (backpropagation) of neural network.
     *
     * @param grads Input gradient to backward pass, gradients are calculated from loss function.
     * @return <code>{@link List<GradsHolder>}</code> where <code>{@link GradsHolder}</code> contain weights of
     *          neural network layers and its respective gradients.
     */
    List<GradsHolder> backward(Matrix grads);
}
