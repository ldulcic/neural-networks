package hr.fer.zemris.nenr.lab5;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.layers.FullyConnectedLayer;
import hr.fer.zemris.nenr.lab5.nn.layers.SigmoidLayer;
import hr.fer.zemris.nenr.lab5.nn.loss.MeanSquaredError;

import java.io.FileWriter;
import java.io.IOException;

public class Main {

    private static Matrix inputs;
    private static Matrix outputs;

    public static void main(String[] args) throws IOException {
        loadData();
        NeuralNetwork nn = new NeuralNetwork.Builder()
                .addLayer(new FullyConnectedLayer(1, 100))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(100, 4))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(4, 1))
                .loss(new MeanSquaredError())
                .build();
        nn.train(inputs, outputs);
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
