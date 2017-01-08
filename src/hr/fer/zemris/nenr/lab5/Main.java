package hr.fer.zemris.nenr.lab5;

import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.FeedForwardNeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.layers.FullyConnectedLayer;
import hr.fer.zemris.nenr.lab5.nn.layers.SigmoidLayer;
import hr.fer.zemris.nenr.lab5.nn.loss.MeanSquaredError;
import hr.fer.zemris.nenr.lab5.nn.loss.SoftmaxWithCrossEntropyLoss;
import hr.fer.zemris.nenr.lab5.nn.optimization.NeuralNetworkOptimizer;
import hr.fer.zemris.nenr.lab5.nn.optimization.MiniBatchBackpropagation;

import java.io.*;
import java.nio.file.Files;
import java.util.List;

public class Main {

    private static Matrix inputs;
    private static Matrix outputs;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        loadDataset();

        NeuralNetwork nn = new FeedForwardNeuralNetwork.Builder()
                .addLayer(new FullyConnectedLayer(60, 200))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(200, 100))
                .addLayer(new SigmoidLayer())
                .addLayer(new FullyConnectedLayer(100, 5))
                .build();

        NeuralNetworkOptimizer optimizer = new MiniBatchBackpropagation.Builder()
                .neuralNetwork(nn)
                .loss(new SoftmaxWithCrossEntropyLoss())
                .inputs(inputs)
                .outputs(outputs)
                .batchSize(1)
                .numOfEpoch(10000)
                .minError(1E-4)
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

    private static void loadDataset() throws IOException {
        inputs = new Matrix(60, 100);
        outputs = new Matrix(5, 100);

        int row = 0;
        String[] directories = new String[] { "Dataset/alpha/", "Dataset/beta/", "Dataset/gama/", "Dataset/delta/", "Dataset/epsilon/" };
        for (String directory : directories) {
            for (int i = 1; i <= 20; i++) {
                File file = new File(directory + String.valueOf(i));
                List<String> lines = Files.readAllLines(file.toPath());
                for (int j = 0; j < lines.size(); j++) {
                    String number = lines.get(j).trim();
                    if (!number.isEmpty()) {
                        inputs.setElement(row, j, Double.parseDouble(number));
                    }
                }
                ++row;
            }
        }

        for (int i = 0; i < 100; i++) {
            if (i < 20) {
                outputs.setElement(i, 0, 1);
            } else if (i < 40) {
                outputs.setElement(i, 1, 1);
            } else if (i < 60) {
                outputs.setElement(i, 2, 1);
            } else if (i < 80) {
                outputs.setElement(i, 3, 1);
            } else {
                outputs.setElement(i, 4, 1);
            }
        }
    }
}
