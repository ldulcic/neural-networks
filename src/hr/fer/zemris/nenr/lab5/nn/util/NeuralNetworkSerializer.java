package hr.fer.zemris.nenr.lab5.nn.util;

import hr.fer.zemris.nenr.lab5.nn.NeuralNetwork;

import java.io.*;

/**
 * Util class for serializing and deserializing <code>{@link NeuralNetwork}</code> class instances.
 *
 * Created by luka on 12/18/16.
 */
public class NeuralNetworkSerializer {

    public static void serializeNetwork(NeuralNetwork nn, String path) throws IOException {
        FileOutputStream fos = new FileOutputStream(path);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(nn);
        oos.close();
        fos.close();
    }

    public static NeuralNetwork deserializeNetwork(String path) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(path);
        ObjectInputStream ois = new ObjectInputStream(fis);
        NeuralNetwork nn = (NeuralNetwork) ois.readObject();
        ois.close();
        fis.close();
        return nn;
    }
}
