import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 200


def generate_random_coefficients():
    theta_0 = np.random.rand() * 5
    theta_1 = np.random.rand() * 10

    return theta_0, theta_1


def generate_random_data(theta_0, theta_1):
    # X = np.random.uniform(low=-10, high=10, size=(M, 1))  # low=-50, high=50,
    X = np.random.normal(size=(M, 1))
    Y = theta_1 * X + theta_0 + np.random.normal(size=(M, 1)) * 5

    return X, Y

def normalize_data_points(X):
    X = X - X.mean()
    X = X / X.std()

    return X


def hypothesis(theta_0, theta_1, training_examples):
    theta = np.array([theta_0, theta_1])

    augmented_x = np.hstack(
        (np.ones(shape=(M, 1), dtype=np.float32), training_examples))
    y_hat = augmented_x @ theta

    return y_hat


def J(theta_0, theta_1, training_examples, true_labels):  # Loss Function
    error = hypothesis(theta_0, theta_1, training_examples) - true_labels

    return np.linalg.norm(error ** 2, axis=0) / (2 * M)

def grad_J(theta_0, theta_1, training_examples, true_labels):
    y_hat = hypothesis(theta_0, theta_1, training_examples)
    augmented_x = np.hstack((np.ones(shape=(M, 1), dtype=np.float32), training_examples))

    grad = 1 / M * (augmented_x.T @ (y_hat - true_labels))
    return grad


def plot_vector_field(X, Y, start, end, points):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(
        0, 1), projection='3d', colspan=4, rowspan=4)
    # ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 0))

    x = np.linspace(start, end, points)
    y = np.linspace(start, end, points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_input = x_mesh.flatten()
    y_input = y_mesh.flatten()

    Z = grad_J(x_input, y_input, X, Y)
    Z = Z.reshape(2, points, points)

    u = Z[0, :, :]
    v = Z[1, :, :]
    ax1.quiver(x_mesh, y_mesh, u, v)

    Z = J(x_input, y_input, X, Y)
    Z = Z.reshape(points, points)

    ax2.plot_surface(x_mesh, y_mesh, Z, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
    # ax1.contour(x_mesh, y_mesh, Z, 30, cmap='RdGy')

def main():
    theta_0, theta_1 = generate_random_coefficients()
    X, Y = generate_random_data(theta_0, theta_1)
    plot_vector_field(X, Y, -20, 20, 100)
    plt.show()
    

if __name__ == '__main__':
    main()
