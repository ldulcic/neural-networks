package hr.fer.zemris.nenr.lab5.nn.optimization;

import hr.fer.zemris.nenr.lab5.matrix.GradsHolder;
import hr.fer.zemris.nenr.lab5.matrix.Matrix;
import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;
import hr.fer.zemris.nenr.lab5.nn.loss.Loss;

import java.util.*;

/**
 * Implements mini-batch backpropagation optimization of neural network. This method uses only part of dataset (batch)
 * in each iteration of neural network backpropagation. If batch size is 1 this method degrades into stochastic
 * backpropagation.
 *
 * Created by luka on 12/18/16.
 */
public class MiniBatchBackpropagation implements NeuralNetworkOptimizer {

    private NeuralNetwork nn;
    private Loss loss;
    private Matrix inputs;
    private Matrix outputs;
    private Dataset dataset;
    private int batchSize;
    private int numOfEpoch;
    private double minError;

    private MiniBatchBackpropagation(Builder builder) {
        this.nn = builder.nn;
        this.loss = builder.loss;
        this.inputs = builder.inputs;
        this.outputs = builder.outputs;
        this.batchSize = builder.batchSize;
        this.numOfEpoch = builder.numOfEpoch;
        this.minError = builder.minError;
        this.dataset = new Dataset(inputs, outputs, batchSize);
    }

    @Override
    public NeuralNetwork optimize() {
        int numOfBatches = inputs.getHeight() / batchSize;
        double currentLoss;
        Matrix nnOutput;
        Matrix lossGrads;
        List<GradsHolder> nnGrads;
        BatchHolder batch;
        for (int i = 0; i < numOfEpoch; i++) {
            nnOutput = nn.forward(inputs);
            currentLoss = loss.forward(nnOutput, outputs);

            if ((i + 1) % 100 == 0) {
                System.out.printf("Epoch %d/%d, loss = ", i + 1, numOfEpoch);
                System.out.println(currentLoss);
            }

            if (currentLoss < minError) {
                System.out.printf("Finished after %d/%d epoch, loss = ", i + 1, numOfEpoch);
                System.out.println(currentLoss);
                break;
            }

            for (int j = 0; j < numOfBatches; j++) {
                batch = dataset.nextBatch();

                nnOutput = nn.forward(batch.inputsBatch);
                lossGrads = loss.backwardsInputs(nnOutput, batch.outputsBatch);
                nnGrads = nn.backward(lossGrads);

                for (GradsHolder grads : nnGrads) {
                    grads.performParameterUpdates();
                }
            }
        }

        return nn;
    }

    public class Dataset {

        private List<double[]> inputExamples;
        private List<double[]> outputExamples;
        private final int batchSize;
        private final int numOfBatches;
        private int nextBatchIndex = 0;
        private final int inputDimen;
        private final int outputDimen;

        private Random random = new Random();

        public Dataset(Matrix inputs, Matrix outputs, int batchSize) {
            this.batchSize = batchSize;
            this.numOfBatches = inputs.getHeight() / batchSize;
            this.inputDimen = inputs.getWidth();
            this.outputDimen = outputs.getWidth();
            breakDatasetIntoSingleExamples(inputs, outputs);
        }

        private void breakDatasetIntoSingleExamples(Matrix inputs, Matrix outputs) {
            inputExamples = new ArrayList<>(inputs.getHeight());
            outputExamples = new ArrayList<>(outputs.getHeight());
            for (int i = 0; i < inputs.getHeight(); i++) {
                inputExamples.add(inputs.getRow(i));
                outputExamples.add(outputs.getRow(i));
            }
            shuffleDataset();
        }

        private void shuffleDataset() {
            long seed = random.nextLong();
            Collections.shuffle(inputExamples, new Random(seed));
            Collections.shuffle(outputExamples, new Random(seed));
        }

        public BatchHolder nextBatch() {
            if (nextBatchIndex == numOfBatches) {
                shuffleDataset();
                nextBatchIndex = 0;
            }
            double[][] inputsBatch = inputExamples.subList(nextBatchIndex * batchSize, (nextBatchIndex + 1) * batchSize)
                                            .toArray(new double[batchSize][inputDimen]);
            double[][] outputsBatch = outputExamples.subList(nextBatchIndex * batchSize, (nextBatchIndex + 1) * batchSize)
                    .toArray(new double[batchSize][outputDimen]);
            ++nextBatchIndex;
            return new BatchHolder(new Matrix(inputsBatch), new Matrix(outputsBatch));
        }
    }

    private class BatchHolder {

        Matrix inputsBatch;
        Matrix outputsBatch;

        BatchHolder(Matrix inputsBatch, Matrix outputsBatch) {
            this.inputsBatch = inputsBatch;
            this.outputsBatch = outputsBatch;
        }

//        @Override
//        public String toString() {
//            StringBuilder sb = new StringBuilder();
//            sb.append("Inputs:\n");
//            sb.append(inputsBatch.toString());
//            sb.append("Outputs:\n");
//            sb.append(outputsBatch.toString());
//            return sb.toString();
//        }
    }

    public static class Builder {

        NeuralNetwork nn;
        Loss loss;
        Matrix inputs;
        Matrix outputs;

        int batchSize = 50;
        int numOfEpoch = Integer.MAX_VALUE;
        double minError = Double.MIN_VALUE;

        public Builder neuralNetwork(NeuralNetwork nn) {
            this.nn = nn;
            return this;
        }

        public Builder loss(Loss loss) {
            this.loss = loss;
            return this;
        }

        public Builder inputs(Matrix inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder outputs(Matrix outputs) {
            this.outputs = outputs;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder numOfEpoch(int numOfEpoch) {
            this.numOfEpoch = numOfEpoch;
            return this;
        }

        public Builder minError(double minError) {
            this.minError = minError;
            return this;
        }

        public MiniBatchBackpropagation build() {
            Objects.requireNonNull(nn, "neural network == null");
            Objects.requireNonNull(loss, "loss == null");
            Objects.requireNonNull(inputs, "inputs == null");
            Objects.requireNonNull(outputs, "outputs == null");
            if (inputs.getHeight() != outputs.getHeight()) {
                throw new IllegalArgumentException("Number of examples in inputs and outputs don't match.");
            }
            if (!outputs.isVector()) {
                throw new IllegalArgumentException("Outputs matrix is expected to be column-vector.");
            }
            if (inputs.getHeight() < batchSize) {
                throw new IllegalArgumentException("Batch size is greater than number of examples in dataset.");
            }
            if (inputs.getHeight() % batchSize != 0) {
                throw new IllegalArgumentException("numOfExamples / batchSize is not integer.");
            }
            return new MiniBatchBackpropagation(this);
        }
    }
}
