# import numpy as np
# import pandas as pd
# from tensorflow.keras.layers import Dense, Dropout
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# # Forward kinematics function
# def forward_kinematics(theta1, theta2, a1=1, a2=1):
#     """
#     Calculate end-effector position given joint angles
#     Args:
#         theta1, theta2: joint angles in radians
#         a1, a2: link lengths
#     Returns:
#         x, y: end-effector position
#     """
#     x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
#     y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
#     return x, y

# # Generate training data
# def generate_dataset(num_samples):
#     """
#     Generate training/testing dataset
#     Args:
#         num_samples: number of samples to generate
#     Returns:
#         inputs (x,y positions) and outputs (joint angles)
#     """
#     # Generate random joint angles
#     theta1 = np.random.uniform(-np.pi, np.pi, num_samples)
#     theta2 = np.random.uniform(-np.pi, np.pi, num_samples)
    
#     # Calculate corresponding end-effector positions
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
    
#     for i in range(num_samples):
#         x[i], y[i] = forward_kinematics(theta1[i], theta2[i])
    
#     # Combine into input/output arrays
#     X = np.column_stack((x, y))
#     y = np.column_stack((theta1, theta2))
    
#     return X, y

# def main():
#     # Generate datasets
#     print("Generating datasets...")
#     X_train, y_train = generate_dataset(10000)  # Training data
#     X_test, y_test = generate_dataset(2000)     # Testing data

#     # Save datasets to CSV
#     print("Saving datasets to CSV files...")
#     pd.DataFrame(X_train, columns=['x', 'y']).to_csv('train_inputs.csv', index=False)
#     pd.DataFrame(y_train, columns=['theta1', 'theta2']).to_csv('train_outputs.csv', index=False)
#     pd.DataFrame(X_test, columns=['x', 'y']).to_csv('test_inputs.csv', index=False)
#     pd.DataFrame(y_test, columns=['theta1', 'theta2']).to_csv('test_outputs.csv', index=False)

#     # Build the neural network model
#     print("Creating neural network model...")
# model = models.Sequential([
#         layers.Dense(512, activation='relu', input_shape=(2,)),
#         Dropout(0.3),
#         layers.Dense(512, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(256, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(256, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(128, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(128, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(64, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(32, activation='relu'),
#         Dropout(0.3),
#         layers.Dense(2)  # Output layer (theta1, theta2)
#     ])
    
#     model.compile(optimizer='adam',
#                  loss='mse',
#                  metrics=['mae'])

#     # Train the model
#     print("Training the model...")
#     history = model.fit(X_train, y_train,
#                        epochs=300,
#                        batch_size=32,
#                        validation_split=0.2,
#                        verbose=1)

#     # Evaluate the model
#     print("\nEvaluating the model...")
#     test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Mean Absolute Error: {test_mae:.4f} radians")

#     # Make predictions
#     predictions = model.predict(X_test[:10])

#     # Plot training history
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['mae'], label='Training MAE')
#     plt.plot(history.history['val_mae'], label='Validation MAE')
#     plt.title('Model Mean Absolute Error')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('training_history.png')
#     plt.show()

#     # Plot predictions
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.scatter(range(10), y_test[:10, 0], label='True θ1', alpha=0.5)
#     plt.scatter(range(10), predictions[:, 0], label='Predicted θ1', alpha=0.5)
#     plt.title('Joint Angle 1 Predictions')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Angle (radians)')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.scatter(range(10), y_test[:10, 1], label='True θ2', alpha=0.5)
#     plt.scatter(range(10), predictions[:, 1], label='Predicted θ2', alpha=0.5)
#     plt.title('Joint Angle 2 Predictions')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Angle (radians)')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('predictions.png')
#     plt.show()

#     # Save the model
#     print("Saving the model...")
#     model.save('inverse_kinematics_model.h5')
#     print("Done! Model saved as 'inverse_kinematics_model.h5'")

# if __name__ == "__main__":
#     main()
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Forward kinematics function
def forward_kinematics(theta1, theta2, a1=1, a2=1):
    x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
    y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
    return x, y

# Generate training data
def generate_dataset(num_samples):
    theta1 = np.random.uniform(-np.pi, np.pi, num_samples)
    theta2 = np.random.uniform(-np.pi, np.pi, num_samples)
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        x[i], y[i] = forward_kinematics(theta1[i], theta2[i])
    X = np.column_stack((x, y))
    y = np.column_stack((theta1, theta2))
    return X, y

def main():
    print("Generating datasets...")
    X_train, y_train = generate_dataset(10000)
    X_test, y_test = generate_dataset(2000)

    print("Saving datasets to CSV files...")
    pd.DataFrame(X_train, columns=['x', 'y']).to_csv('train_inputs.csv', index=False)
    pd.DataFrame(y_train, columns=['theta1', 'theta2']).to_csv('train_outputs.csv', index=False)
    pd.DataFrame(X_test, columns=['x', 'y']).to_csv('test_inputs.csv', index=False)
    pd.DataFrame(y_test, columns=['theta1', 'theta2']).to_csv('test_outputs.csv', index=False)

    print("Creating neural network model...")
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(2,)),
        Dropout(0.3),
        layers.Dense(512, activation='relu'),
        Dropout(0.3),
        layers.Dense(256, activation='relu'),
        Dropout(0.3),
        layers.Dense(256, activation='relu'),
        Dropout(0.3),
        layers.Dense(128, activation='relu'),
        Dropout(0.3),
        layers.Dense(128, activation='relu'),
        Dropout(0.3),
        layers.Dense(64, activation='relu'),
        Dropout(0.3),
        layers.Dense(32, activation='relu'),
        Dropout(0.3),
        layers.Dense(2)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2, verbose=1)

    print("\nEvaluating the model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Mean Absolute Error: {test_mae:.4f} radians")

    predictions = model.predict(X_test[:10])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(10), y_test[:10, 0], label='True θ1', alpha=0.5)
    plt.scatter(range(10), predictions[:, 0], label='Predicted θ1', alpha=0.5)
    plt.title('Joint Angle 1 Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (radians)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(10), y_test[:10, 1], label='True θ2', alpha=0.5)
    plt.scatter(range(10), predictions[:, 1], label='Predicted θ2', alpha=0.5)
    plt.title('Joint Angle 2 Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (radians)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

    print("Saving the model...")
    model.save('inverse_kinematics_model.h5')
    print("Done! Model saved as 'inverse_kinematics_model.h5'")

if __name__ == "__main__":
    main()
