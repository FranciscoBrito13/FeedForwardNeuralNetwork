import java.util.Arrays;

public class Main {
    public static void main(String[] args) {


        int inputDim = 3;
        int hiddenDim = 2;
        int outputDim = 1;

        double[] values = {0.5,0.5,0.5,0.5,0.5,0.5,0.2,0.1,0.3,0.3,0.5};

        NeuralNetworkFeedForward neuralNetwork = new NeuralNetworkFeedForward(inputDim, hiddenDim, outputDim, values);

        double[] input = {0.0,0.0,1.0};
        double[] result = neuralNetwork.forward(input);
        System.out.println(neuralNetwork);
        for (int i = 0; i < result.length; i++) {
            System.out.println(" Result for neuron ouput"+(i+1)+
                    "= "+result[i]);
        }



        /*
        NeuralNetworkFeedForward network = new NeuralNetworkFeedForward(2, 2, 1);
        double[] input = {15, 9};
        double[] inputLabel = {1};
        System.out.println("My initial output value = " + Arrays.toString(network.forward(input)));
        System.out.println("My initial squared error = " + network.squaredErrorOneOutput(input, inputLabel[0]));
        System.out.println("My initial neural network:" + System.lineSeparator() + network);
        //System.out.println(network);
        network.bogoImprove(input, inputLabel[0]);

        System.out.println("My final output value = " + Arrays.toString(network.forward(input)));
        System.out.println("My final squared error = " + network.squaredErrorOneOutput(input, inputLabel[0]));
        System.out.println("My improved neural network:" + System.lineSeparator() + network);

         */
    }


}
