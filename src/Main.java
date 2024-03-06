import java.util.Arrays;

public class Main {
    public static void main(String[] args) {


        int inputDim = 2;
        int hiddenDim = 2;
        int outputDim = 1;

        NeuralNetworkFeedForward neuralNetwork = new NeuralNetworkFeedForward(inputDim, hiddenDim, outputDim);

        System.out.println("Neural Network com valoes aleatórios");
        System.out.println(neuralNetwork);

        double[] values = {
                0.1, 0.2,
                0.3, 0.4,
                0.5, 0.6,
                0.7,
                0.8,
                0.9
        };
        NeuralNetworkFeedForward neuralNetworkWithValues = new NeuralNetworkFeedForward(inputDim, hiddenDim, outputDim, values);
        System.out.println(System.lineSeparator() + "Neural Network com valores definidos");
        System.out.println(neuralNetworkWithValues);
        System.out.println(Arrays.toString(neuralNetworkWithValues.getNeuralNetwork()));
    }

}