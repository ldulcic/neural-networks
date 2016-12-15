package hr.fer.zemris.nenr.lab5.nn.layers;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Interface for neural network layer.
 *
 * Created by luka on 12/15/16.
 */
public interface Layer {

    /**
     * Performs forward pass trough neural net layer.
     *
     * @param inputs Input data of dimensions [N, D], where N is num of examples and D is dimensionality of each example.
     * @return <code>Matrix</code> which is result of layer transformation of dimensions [N, H] where N is num of
     *          examples and H is num of neurons in this layer.
     */
    Matrix forward(Matrix inputs);

    /**
     * Performs backward pass of neural net layer with regards to inputs.
     *
     * @param grads Input gradients to layer of dimensions [N, H] where is N is num of examples and H is number of
     *              neurons in this layer.
     * @return <code>Matrix</code> which represents gradients with respect to inputs of this layer. <code>Matrix</code>
     *          is of dimensions [N, H] where N is num of examples and H is num of neurons in this layer.
     */
    Matrix backwardInputs(Matrix grads);

    /**
     * Method for determining if layer has (trainable) parameters. Not all neural net layer have parameters.
     *
     * @return <code>true</code> if layer has (trainable) parameters, <code>false</code> otherwise.
     */
    boolean hasParams();

    /**
     * Performs backward pass of neural net layer with regards to parameters of layer.
     *
     * @param grads Input gradients to layer of dimensions [N, H] where is N is num of examples and H is number of
     *              neurons in this layer.
     * @return <code>{@link GradsHolder}</code> which holds weights, biases and its respective gradients.
     */
    GradsHolder backwardParams(Matrix grads);
}
