import numpy as np
import matplotlib.pyplot as plt

# Function to initialize the neural network weights
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output


# Leaky relu activation function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# Derivative of leaky relu function
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# Function to train the neural network
def train_neural_network(
    train_inputs,
    train_outputs,
    hidden_size,
    learning_rate,
    epochs,
    test_inputs=None,
    test_outputs=None,
):

    # Determine input and output sizes from data
    input_size = train_inputs.shape[1]
    output_size = train_outputs.shape[1]

    # Initialize weights
    weights_input_hidden, weights_hidden_output = initialize_weights(
        input_size, hidden_size, output_size
    )

    # Dictionary to track errors
    errors = {"train": [], "test": []}

    for epoch in range(epochs):

        # Forward pass
        hidden_layer_input = np.dot(train_inputs, weights_input_hidden)
        hidden_layer_output = leaky_relu(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = leaky_relu(output_layer_input)

        # Calculate training error
        train_error = train_outputs - predicted_output
        errors["train"].append(np.mean(np.abs(train_error)))

        # Calculate testing error if data provided
        if test_inputs is not None and test_outputs is not None:
            test_hidden = leaky_relu(np.dot(test_inputs, weights_input_hidden))
            test_output = leaky_relu(np.dot(test_hidden, weights_hidden_output))
            test_error = test_outputs - test_output
            errors["test"].append(np.mean(np.abs(test_error)))

        # Backpropagation
        output_delta = train_error * leaky_relu_derivative(predicted_output)
        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * leaky_relu_derivative(
            hidden_layer_output
        )

        # Update weights
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        weights_input_hidden += train_inputs.T.dot(hidden_layer_delta) * learning_rate

        # Print epoch status
        if epoch % 100 == 0:
            print(
                f'Epoch {epoch}, Training Error: {errors["train"][-1]}, Testing Error: {errors["test"][-1]}'
            )

    return weights_input_hidden, weights_hidden_output, errors


# Gather data
data = np.genfromtxt("yellow-paint.txt") # Choose from various BRDFs 
angles = data[:, :4]
brdf_values = data[:, 4:]
angles_normalized = angles / np.max(angles, axis=0)
brdf_values_normalized = brdf_values / np.max(brdf_values, axis=0)

# Split data into training and test sets
split_ratio = 0.5
split_index = int(len(data) * split_ratio)

train_angles, test_angles = (
    angles_normalized[:split_index],
    angles_normalized[split_index:],
)
train_brdf, test_brdf = (
    brdf_values_normalized[:split_index],
    brdf_values_normalized[split_index:],
)

# Further split training data
split_index = len(train_angles) // 2
train_angles = train_angles[:split_index]
train_brdf = train_brdf[:split_index]

# Define network parameters
hidden_size = 10
learning_rate = 0.0001
epochs = 1000

# Train network
(
    trained_weights_input_hidden,
    trained_weights_hidden_output,
    errors,
) = train_neural_network(
    train_angles, train_brdf, hidden_size, learning_rate, epochs, test_angles, test_brdf
)

# Save final weights
np.savetxt(
    "final_weights.txt",
    np.concatenate(
        [
            trained_weights_input_hidden.flatten(),
            trained_weights_hidden_output.flatten(),
        ]
    ),
)

# Plot errors
plt.plot(range(0, len(errors["train"])), errors["train"], label="Training Error")
plt.plot(range(0, len(errors["test"])), errors["test"], label="Testing Error")
plt.title("Training and Testing Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()