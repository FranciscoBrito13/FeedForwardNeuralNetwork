
public class NeuralNetworkFeedForward {
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private double[][] hiddenWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    public NeuralNetworkFeedForward(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;

        initializeParameters();
    }

    public NeuralNetworkFeedForward(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;

        initializeParameters(values);
    }

    public double[] forward(double[] inputValues) {
        if (inputValues.length != inputDim) {
            throw new IllegalArgumentException("Wrong amount of input values");
        }
        double[] hiddenLayerOutput = new double[hiddenDim];

        for (int i = 0; i < hiddenDim; i++) {
            double neuronSum = 0.0;
            for (int j = 0; j < inputDim; j++) {
                neuronSum += inputValues[j] * hiddenWeights[j][i];
            }
            neuronSum += hiddenBiases[i];
            hiddenLayerOutput[i] = sigmoid(neuronSum);
        }

        double[] outputLayerOutput = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            double neuronSum = 0.0;
            for (int j = 0; j < hiddenDim; j++) {
                neuronSum += hiddenLayerOutput[j] * outputWeights[j][i];
            }
            neuronSum += outputBiases[i];
            outputLayerOutput[i] = sigmoid(neuronSum);
        }

        return outputLayerOutput;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double[] getNeuralNetwork() {
        double[] values = new double[(inputDim * hiddenDim) + hiddenDim + (hiddenDim * outputDim) + outputDim];

            int index = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                values[index++] = hiddenWeights[i][j];
            }
        }
        for (int i = 0; i < hiddenDim; i++) {
            values[index++] = hiddenBiases[i];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                values[index++] = outputWeights[i][j];
            }
        }
        for (int i = 0; i < outputDim; i++) {
            values[index++] = outputBiases[i];
        }
        return values;
    }

    private void initializeParameters(double[] values) {
        int expectedLength = inputDim * hiddenDim + hiddenDim + hiddenDim * outputDim + outputDim;
        if (values.length != expectedLength) {
            throw new IllegalArgumentException("Wrong amount of arguments");
        }

        hiddenBiases = new double[hiddenDim];
        outputBiases = new double[outputDim];

        //Creates the weights matrix depending on the amount of neurons
        hiddenWeights = new double[inputDim][hiddenDim];
        outputWeights = new double[hiddenDim][outputDim];

        int index = 0;

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                hiddenWeights[i][j] = values[index];
                index++;
            }
        }


        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = values[index];
            index++;
        }


        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = values[index];
                index++;
            }
        }

        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = values[index];
            index++;
        }
    }

    private void initializeParameters(){
        //Creates the bias Array depending on the amount of neurons
        hiddenBiases = new double[hiddenDim];
        outputBiases = new double[outputDim];

        //Creates the weights matrix depending on the amount of neurons
        hiddenWeights = new double[inputDim][hiddenDim];
        outputWeights = new double[hiddenDim][outputDim];

        //Sets random values for the bias
        for(int i = 0; i < hiddenDim; i++){
            hiddenBiases[i] = (Math.random()*0.50);
        }
        for(int i = 0; i < outputDim; i++){
            outputBiases[i] = (Math.random()*0.5);
        }

        //Sets random values for the weights
        for(int i = 0; i < hiddenWeights.length; i++){
            for(int j = 0; j < hiddenWeights[0].length; j++){
                hiddenWeights[i][j] = (Math.random()*0.50);
            }
        }
        for(int i = 0; i < outputWeights.length; i++){
            for(int j = 0; j < outputWeights[0].length; j++){
                outputWeights[i][j] = (Math.random()*0.50);
            }
        }
    }

    @Override
    public String toString() {
        String result = "Neural Network: \nNumber of inputs: "
                + inputDim + "\n"
                + "Weights between input and hidden layer with " + hiddenDim + " neurons: \n";
        String hidden = "";
        for (int input = 0; input < inputDim; input++) {
            for (int i = 0; i < hiddenDim; i++) {
                hidden += " input" + input + "-hidden" + i + ": "
                        + hiddenWeights[input][i] + "\n";
            }
        }
        result += hidden;
        String biasHidden = "Hidden biases: \n";
        for (int i = 0; i < hiddenDim; i++) {
            biasHidden += " bias hidden" + i + ": " + hiddenBiases[i] + "\n";
        }
        result += biasHidden;
        String output = "Weights between hidden and output layer with "
                + outputDim + " neurons: \n";
        for (int hiddenw = 0; hiddenw < hiddenDim; hiddenw++) {
            for (int i = 0; i < outputDim; i++) {
                output += " hidden" + hiddenw + "-output" + i + ": "
                        + outputWeights[hiddenw][i] + "\n";
            }
        }
        result += output;
        String biasOutput = "Ouput biases: \n";
        for (int i = 0; i < outputDim; i++) {
            biasOutput += " bias ouput" + i + ": " + outputBiases[i] + "\n";
        }
        result += biasOutput;
        return result;
    }

}
