import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])
epochs = 1000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

hidden_weights = np.random.uniform(size = (inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size = (1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size= (1, outputLayerNeurons))

print("Hidden Weights: ", *hidden_weights)
print("Hiddden Bias: ", *hidden_bias)
print("Output Weights: ", *output_weights)
print("Output Bias: ", *output_bias)

for _ in range(epochs):
    HlayerActivation = np.dot(inputs, hidden_weights)
    HlayerActivation += hidden_bias
    HlayerOutput = sigmoid(HlayerActivation)

    OlayerActivation = np.dot(HlayerOutput, output_weights)
    OlayerActivation += output_bias
    predicted_output = sigmoid(OlayerActivation)

    #BackPropagation
    error = expected_output - predicted_output
    d_predictedOutput = error * sigmoid_derivative(predicted_output)

    error_hiddenLayer = d_predictedOutput.dot(output_weights.T)
    d_hiddenLayer = error_hiddenLayer*sigmoid_derivative(HlayerOutput)

    hidden_weights += inputs.T.dot(d_hiddenLayer)*lr
    hidden_bias += np.sum(d_hiddenLayer, axis=0, keepdims=True)*lr

    output_weights += HlayerOutput.T.dot(d_predictedOutput)*lr
    output_bias += np.sum(d_predictedOutput, axis=0, keepdims=True)*lr

print("Final Hidden Weights: ", *hidden_weights)
print("Final Hidden Bias: ", *hidden_bias)
print("Final Output Weights: ", *output_weights)
print("Final Output Bias: ", *output_bias)

print("Output from neural network: ", *predicted_output)
