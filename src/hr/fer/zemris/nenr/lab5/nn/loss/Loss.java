package hr.fer.zemris.nenr.lab5.nn.loss;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Interface defining loss function for neural network.
 *
 * Created by luka on 12/16/16.
 */
public interface Loss {

    /**
     * Calculates loss of <code>inputs</code> with respect to <code>outputs</code>.
     *
     * @param inputs <code>{@link Matrix}</code> containing inputs to loss function (predicted values).
     * @param outputs <code>{@link Matrix}</code> containing real outputs.
     * @return Average loss over all examples.
     */
    double forward(Matrix inputs, Matrix outputs);

    /**
     * Calculates gradient of loss function with respect to <code>inputs</code>.
     *
     * @param inputs <code>{@link Matrix}</code> containing inputs to loss function (predicted values).
     * @param outputs <code>{@link Matrix}</code> containing real outputs.
     * @return <code>{@link Matrix}</code> containing gradients with respect to <code>inputs</code>.
     */
    Matrix backwardsInputs(Matrix inputs, Matrix outputs);
}
