package hr.fer.zemris.nenr.lab5.nn.layers;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.matrix.MatrixUtil;

import java.util.Objects;

/**
 * Implements sigmoid layer of neural network.
 *
 * Created by luka on 12/15/16.
 */
public class SigmoidLayer implements Layer {

    private Matrix inputs;

    @Override
    public Matrix forward(Matrix inputs) {
        this.inputs = inputs.clone();
        Matrix outputs = new Matrix(inputs.getWidth(), inputs.getHeight());
        double element;
        for (int i = 0; i < inputs.getHeight(); i++) {
            for (int j = 0; j < inputs.getWidth(); j++) {
                element = inputs.getElement(i, j);
                outputs.setElement(i, j, 1. / ( 1 + Math.exp(-element) ) );
            }
        }
        return outputs;
    }

    @Override
    public Matrix backwardInputs(Matrix grads) {
        Objects.requireNonNull(inputs);
        MatrixUtil.checkIfMatricesHaveSameDimensions(inputs, grads);
        double gradient;
        double sigmoid;
        for (int i = 0; i < grads.getHeight(); i++) {
            for (int j = 0; j < grads.getWidth(); j++) {
                gradient = grads.getElement(i, j);
                sigmoid = 1. / ( 1 + Math.exp(-inputs.getElement(i, j) ) );
                grads.setElement(i, j, gradient * sigmoid * ( 1 - sigmoid ) );
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
