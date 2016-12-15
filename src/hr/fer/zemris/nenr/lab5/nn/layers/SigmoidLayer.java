package hr.fer.zemris.nenr.lab5.nn.layers;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Implements sigmoid layer of neural network.
 *
 * Created by luka on 12/15/16.
 */
public class SigmoidLayer implements Layer {

    @Override
    public Matrix forward(Matrix inputs) {
        double element;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                element = inputs.getElement(i, j);
                inputs.setElement(i, j, 1. / ( 1 + Math.exp(-element) ) );
            }
        }
        return inputs;
    }

    @Override
    public Matrix backwardInputs(Matrix grads) {
        double element;
        double sigmoid;
        for (int i = 0; i < grads.getHeight(); i++) {
            for (int j = 0; j < grads.getWidth(); j++) {
                element = grads.getElement(i, j);
                sigmoid = 1. / ( 1 + Math.exp(-element) );
                grads.setElement(i, j, element * ( sigmoid * ( 1 - sigmoid ) ) );
            }
        }
        return grads;
    }

    @Override
    public boolean hasParams() {
        return false;
    }

    @Override
    public GradsHolder backwardParams(Matrix grads) {
        throw new LayerWithNoParametersException();
    }
}
