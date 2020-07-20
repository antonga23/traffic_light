import numpy as np


def main():
    weights = np.array([0.7, 0.2, -0.5])
    alpha = 0.1
    street_lights = np.array([[0, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],
                             [1, 0, 1]])

    walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

    for iteration in range(40):
        error_for_all_lights = 0
        for row_index in range(len(walk_vs_stop)):
            input = street_lights[row_index]
            goal_prediction = walk_vs_stop[row_index]

            prediction = input.dot(weights)
            error = (prediction - goal_prediction) ** 2
            error_for_all_lights += error

            delta = prediction - goal_prediction
            weights = weights - (alpha * (input * delta))
            print("Prediction: " + str(prediction))
        print("Weights:" + str(weights))
        print("Error:" + str(error))


if __name__ == "__main__":
    main()
