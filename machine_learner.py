import pandas as pd
import numpy as np
import csv
import math
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

print(f"\n######  Libraries have been imported  ######\n")

# ["AWR", "AGR", "ASR", "PSH", "PSA", "PWHB", "PWAB",
#  "PGHB", "PGAB", "PSHB", "PSAB", "PWHS", "PWAS", "PGHS",
#  "PGAS", "PSHS", "PSAS", "TGH", "TGA", "DOG",
#  "Result"]

NON_NUMERICS = ["AWR", "AGR", "ASR", "PSH", "PSA", "PWHB", "PWAB",
                "PGHB", "PGAB", "PSHB", "PSAB", "PWHS", "PWAS", "PGHS",
                "PGAS", "PSHS", "PSAS", "TGH", "TGA", "DOG",
                "Result"]

FEATURES_NUM  = ["AWR", "AGR", "ASR", "PSH", "PSA", "PWHB", "PWAB",
                 "PGHB", "PGAB", "PSHB", "PSAB", "PWHS", "PWAS", "PGHS",
                 "PGAS", "PSHS", "PSAS", "TGH", "TGA"]

FEATURES_INT  = ["DOG"]

def main():
    # Load the data
    with open("cooked_data/feature_gen_1_list.csv") as f:
        reader = csv.reader(f)
        for i in range(380*1):
            # Skips the games with little info
            next(reader)

        data = []
        for row in reader:
            text = row[24]
            text = text.strip("'[]")
            text = text.split(", ")
            category_labels = [int(text[0]), int(text[1]), int(text[2])]
            data.append({
                "game_features": [float(cell) for cell in row[4:24]],
                "game_label": category_labels
            })
    
    # Separates training and testing sets
    game_data   = [row["game_features"] for row in data]
    game_data   = normalized_data(game_data)
    game_labels = [row["game_label"] for row in data]
    X_training, X_testing, y_training, y_testing = train_test_split(
        game_data, game_labels, test_size=0.2
    )
    print(f"\nThe training set has {len(y_training)} samples, and the testing set has {len(y_testing)}\n")

    # Create a neural network
    model = tf.keras.models.Sequential()

    # Add a hidden layer with 40 units, with ReLU activation
    model.add(tf.keras.layers.Dense(100, input_shape=(20,), activation="relu"))

    # Adds another layer
    model.add(tf.keras.layers.Dense(30, activation="relu"))

    # Add output layer with 1 unit, with sigmoid activation
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    # Train neural network
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )
    model.fit(X_training, y_training, batch_size=512, epochs=960)

    # `rankdir='LR'` is to make the graph horizontal.
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')

    # Evaluate how well model performs
    model.evaluate(X_testing, y_testing, batch_size=1024, verbose=2)


def normalized_data(data):
    """
    Normalizes the data
    """
    norm_data   = []
    number_feat = len(data[0])
    number_game = len(data)
    # Normalizes each feature
    for i in range(number_feat):
        temp_data = []
        for j in range(number_game):
            temp_data.append(data[j][i])
        mean_t = mean_list(temp_data)
        std  = std_list(temp_data, mean_t)
        for j in range(number_game):
            data[j][i] = (temp_data[j] - mean_t) / std
    return data


def mean_list(some_list):
    """
    Gives the mean
    """
    N = len(some_list)
    mean = 0
    for number in some_list:
        mean += number
    return mean / N


def std_list(some_list, mean):
    """
    Gives the std dev
    """
    square = 0
    N = len(some_list)
    for num in some_list:
        square += (num - mean) * (num - mean)
    square = square / N
    square = math.sqrt(square)
    return square


if __name__ == "__main__":
    main()
