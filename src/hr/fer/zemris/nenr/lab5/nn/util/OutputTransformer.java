package hr.fer.zemris.nenr.lab5.nn.util;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;

/**
 * Util class for transforming output of neural network.
 *
 * Created by luka on 12/18/16.
 */
public class OutputTransformer {

    /**
     * Calculates softmax over inputs and multiplies every element by 100 to get percentage of every input.
     */
    public static double[] getPropabilisticDistribution(Matrix nnOutput) {
        assert nnOutput.isVector();
        double[] probabilities = new double[nnOutput.getWidth()];

        double softmaxSum = 0;
        for (int i = 0; i < nnOutput.getWidth(); i++) {
            softmaxSum += Math.exp(nnOutput.getElement(0, i));
        }

        for (int i = 0; i < nnOutput.getWidth(); i++) {
            probabilities[i] = Math.exp( nnOutput.getElement(0, i) ) / softmaxSum;
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] *= 100;
        }

        return probabilities;
    }
}
