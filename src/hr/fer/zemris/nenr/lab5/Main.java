package hr.fer.zemris.nenr.lab5;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.layers.FullyConnectedLayer;
import hr.fer.zemris.nenr.lab5.nn.layers.SigmoidLayer;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        Matrix input = Matrix.fromFile("input/test.txt");
        NeuralNetwork nn = new NeuralNetwork.Builder()
                .addLayer(new FullyConnectedLayer(5, 2))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(100, 5))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(1, 100))
                .build();
        System.out.println(nn.forward(input));
    }
}
