package hr.fer.zemris.nenr.lab5;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.FeedForwardNeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.layers.FullyConnectedLayer;
import hr.fer.zemris.nenr.lab5.nn.layers.SigmoidLayer;
import hr.fer.zemris.nenr.lab5.nn.loss.MeanSquaredError;
import hr.fer.zemris.nenr.lab5.nn.optimization.NeuralNetworkOptimizer;
import hr.fer.zemris.nenr.lab5.nn.optimization.MiniBatchBackpropagation;

import java.io.IOException;

public class Main {

    private static Matrix inputs;
    private static Matrix outputs;

    public static void main(String[] args) throws IOException {
        loadData();
        FeedForwardNeuralNetwork nn = new FeedForwardNeuralNetwork.Builder()
                .addLayer(new FullyConnectedLayer(1, 100))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(100, 4))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(4, 4))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(4, 1))
                .build();
        NeuralNetworkOptimizer optimizer = new MiniBatchBackpropagation.Builder()
                .neuralNetwork(nn)
                .loss(new MeanSquaredError())
                .inputs(inputs)
                .outputs(outputs)
                .batchSize(1)
                .minError(1E-5)
                .build();
        optimizer.optimize();
    }

    private static void loadData() throws IOException {
        Matrix tmp = Matrix.fromFile("input/test.txt");
        inputs = new Matrix(1, tmp.getHeight());
        outputs = new Matrix(1, tmp.getHeight());
        for (int i = 0; i < tmp.getHeight(); i++) {
            inputs.setElement(i, 0, tmp.getElement(i, 0));
            outputs.setElement(i, 0, tmp.getElement(i, 1));
        }
    }
}
