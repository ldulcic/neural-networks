package hr.fer.zemris.nenr.lab5.nn.util;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;

import java.util.Random;

/**
 * Util class for initializing neural network parameters.
 *
 * Created by luka on 12/16/16.
 */
public class ParametersInitializer {

    private static Random random = new Random();

    public static Matrix initializeWeights(int numOfNeurons, int numOfInputs) {
        double[][] weights = new double[numOfNeurons][numOfInputs];
        for (int i = 0; i < numOfNeurons; i++) {
            for (int j = 0; j < numOfInputs; j++) {
                weights[i][j] = random.nextGaussian() * Math.sqrt(2./numOfInputs);
            }
        }
        return new Matrix(weights);
    }

    public static Matrix initializeBias(int numOfNeurons) {
        return new Matrix(numOfNeurons, 1);
    }
}
