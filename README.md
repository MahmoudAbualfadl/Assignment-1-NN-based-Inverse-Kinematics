# Assignment-1-NN-based-Inverse-Kinematics
# Neural Network-based Inverse Kinematics

## Overview
Inverse kinematics is a classic problem in robotics that involves determining the required joint angles for a robot to reach a specific position with its end-effector. This project uses a neural network to solve the inverse kinematics problem by treating it as a machine learning task, removing the need for complex mathematical modeling.

With deep learning, we are able to approximate the relationship between joint angles and end-effector positions without explicit mathematical equations. This approach leverages large datasets, neural networks, and optimization techniques to handle non-linear and complex relationships in inverse kinematics.

## Project Structure
1. **Data Generation**: We generate training and testing datasets using a forward kinematics function that computes end-effector positions from random joint angles.
2. **Model Building**: We construct a neural network to learn the mapping between (x, y) positions and joint angles (theta1, theta2).
3. **Training**: The model is trained on the generated dataset, using Mean Squared Error (MSE) as the loss function and Mean Absolute Error (MAE) as an evaluation metric.
4. **Evaluation**: The model’s performance is evaluated using test data, and training history is visualized to check for overfitting or underfitting.
5. **Prediction**: The trained model can predict joint angles for given end-effector positions, allowing for practical applications in robotic control.

## Model Architecture
- **Input Layer**: Takes two inputs, x and y, representing the target position of the end-effector.
- **Hidden Layers**: Multiple dense layers with ReLU activation functions, each followed by a dropout layer to prevent overfitting. These layers allow the model to learn complex, non-linear relationships.
- **Output Layer**: Produces two outputs, theta1 and theta2, which represent the joint angles.

## Training Details
- **Optimizer**: Adam optimizer, an adaptive algorithm that adjusts weights to minimize error.
- **Loss Function**: Mean Squared Error (MSE) between predicted and actual joint angles.
- **Metric**: Mean Absolute Error (MAE) is tracked to measure the average prediction error.
- **Epochs**: The model is trained over multiple epochs to allow it to learn from the data.
- **Validation**: A portion of the training data is set aside to evaluate model performance during training and detect overfitting.

## Results
### Training History
The following plots visualize the training and validation performance:
- **Model Loss**: Shows the MSE loss for training and validation over time. The loss decreases and stabilizes, indicating the model is learning effectively.
- **Mean Absolute Error (MAE)**: Shows the average prediction error for both training and validation. The low and consistent MAE indicates that the model’s predictions are close to actual joint angles.

### Observations
- The close alignment of training and validation curves suggests that the model generalizes well.
- Stabilization of loss and MAE values indicates convergence, meaning the model has learned the underlying pattern in the data.

### Next Steps
- **Increasing Epochs**: If the validation loss and MAE are still decreasing, additional epochs might further improve accuracy.
- **Adjusting Model Complexity**: If there’s significant fluctuation or overfitting, simplifying the model or increasing dropout may help.

## Files
- **train_inputs.csv**: Training dataset for end-effector positions (x, y).
- **train_outputs.csv**: Training dataset for joint angles (theta1, theta2).
- **test_inputs.csv**: Testing dataset for end-effector positions.
- **test_outputs.csv**: Testing dataset for joint angles.
- **inverse_kinematics_model.h5**: Trained model saved for future use.
- **training_history.png**: Visualization of model training and validation loss and MAE.
- **predictions.png**: Visualization of model predictions against actual joint angles.

## Requirements
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## Usage
1. Run the script to generate training and testing datasets.
2. Train the neural network using the generated data.
3. Evaluate the model and visualize the training history.
4. Use the model to predict joint angles for a given (x, y) position.

## License
This project is licensed under the MIT License.

