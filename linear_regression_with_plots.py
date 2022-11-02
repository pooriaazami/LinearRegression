from timeit import repeat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
M = 200


def generate_random_coefficients():
    theta_0 = np.random.rand() * 5
    theta_1 = np.random.rand() * 10

    return theta_0, theta_1


def generate_random_data(theta_0, theta_1):
    X = np.random.uniform(low=-10, high=10, size=(M, 1))  # low=-50, high=50,
    Y = theta_1 * X + theta_0 + np.random.normal(size=(M, 1)) * 5

    return X, Y

def normalize_data_points(X):
    X = X - X.mean()
    X = X / X.std()

    return X


def plot_data(X, Y, ax):
    ax.scatter(X, Y, label='Data Points', color='blue', marker='+')


def hypothesis(theta_0, theta_1, training_examples):
    theta = np.array([theta_0, theta_1])

    augmented_x = np.hstack(
        (np.ones(shape=(M, 1), dtype=np.float32), training_examples))
    y_hat = augmented_x @ theta

    return y_hat


def J(theta_0, theta_1, training_examples, true_labels):  # Loss Function
    error = hypothesis(theta_0, theta_1, training_examples) - true_labels

    return np.linalg.norm(error ** 2, axis=0) / (2 * M)


def plot_loss_function(X, Y, ax, start=-10, end=10, points=100):
    x = np.linspace(start, end, points)
    y = np.linspace(start, end, points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_input = x_mesh.flatten()
    y_input = y_mesh.flatten()

    Z = J(x_input, y_input, X, Y)
    Z = Z.reshape(points, points)

    ax.plot_surface(x_mesh, y_mesh, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')

    ax.set_xlabel('\u03B80')
    ax.set_ylabel('\u03B81')
    ax.set_zlabel('J(\u03B80, \u03B81)')


def plot_loss_function_contour(X, Y, ax, start=-20, end=20, points=100):
    x = np.linspace(start, end, points)
    y = np.linspace(start, end, points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_input = x_mesh.flatten()
    y_input = y_mesh.flatten()

    Z = J(x_input, y_input, X, Y)
    Z = Z.reshape(points, points)

    ax.contour(x_mesh, y_mesh, Z, 25, cmap='RdGy')


def step(theta_0, theta_1, X, Y, learning_rate):
    theta = np.array([theta_0, theta_1])[..., np.newaxis]

    y_hat = hypothesis(theta_0, theta_1, X)[..., np.newaxis]
    augmented_x = np.hstack((np.ones(shape=(M, 1), dtype=np.float32), X))

    grad = 1 / M * (augmented_x.T @ (y_hat - Y))

    theta = theta - learning_rate * grad
    theta = theta.squeeze()

    return theta[0], theta[1]


def train(X, Y, learning_rate=0.05, steps=100):
    theta_0, theta_1 = 5., 5.

    theta_0_history = [theta_0]
    theta_1_history = [theta_1]
    loss_history = [J([theta_0], [theta_1], X, Y)]

    for _ in range(1, steps+1):
        theta_0, theta_1 = step(theta_0, theta_1, X, Y, learning_rate)
        loss = J([theta_0], [theta_1], X, Y)

        theta_0_history.append(theta_0)
        theta_1_history.append(theta_1)

        loss_history.append(loss)

    return theta_0, theta_1, loss_history, theta_0_history, theta_1_history

def plot_loss_value(ax, loss_history):
    ax.plot(loss_history, color='black')


def generate_animation_function(ax1, ax2, ax3, ax4, loss_history, theta_0_history, theta_1_history, X, Y, true_theta_0, true_theta_1):
    loss_history_cache = []
    theta_0_history_cache = []
    theta_1_history_cache = []

    def animate(step):
        theta_0 = theta_0_history.pop(0)
        theta_1 = theta_1_history.pop(0)
        loss = loss_history.pop(0)

        theta_0_history_cache.append(theta_0)
        theta_1_history_cache.append(theta_1)
        loss_history_cache.append(loss)

        ax1.clear()
        plot_data(X, Y, ax1)
        plot_line(ax1, true_theta_1, true_theta_0,
                  'black', 'Real Line', -10, 10)
        plot_line(ax1, theta_1, theta_0, 'red', 'Predicted Line', -10, 10)
        plot_loss_history(ax2, loss_history_cache,
                          theta_1_history_cache, theta_0_history_cache)
        plot_loss_function_contour_history(
            ax3, theta_0_history_cache, theta_1_history_cache)
        plot_loss_value(ax4, loss_history_cache)

    return animate


def plot(X, Y, theta_1, theta_0):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0))
    ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), projection='3d', colspan=4, rowspan=4)
    ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 0))
    ax4 = plt.subplot2grid(shape=(3, 3), loc=(2, 0))

    plot_data(X, Y, ax1)
    plot_loss_function(X, Y, ax2)
    plot_loss_function_contour(X, Y, ax3)

    return fig, ax1, ax2, ax3, ax4


def plot_loss_history(ax, loss_history, theta_0_history, theta_1_history):
    ax.scatter(theta_1_history, theta_0_history, loss_history, color='green')


def plot_loss_function_contour_history(ax, theta_0_history, theta_1_history):
    ax.scatter(theta_0_history, theta_1_history, color='green', marker='*')


def plot_line(ax, theta_1, theta_0, color, label, x_start, x_end):
    x = np.linspace(x_start, x_end, 1000)
    y = theta_1 * x + theta_0

    ax.plot(x, y, color=color, label=label)


def main(steps=100):
    theta_0, theta_1 = generate_random_coefficients()
    X, Y = generate_random_data(theta_0, theta_1)
    # X = normalize_data_points(X)
    theta_0_hat, theta_1_hat, loss_history, theta_0_history, theta_1_history = train(
        X, Y, learning_rate=0.01, steps=steps)

    fig, ax1, ax2, ax3, ax4 = plot(X, Y, theta_0, theta_1)
    animation_function = generate_animation_function(
        ax1, ax2, ax3, ax4, loss_history, theta_0_history, theta_1_history, X, Y, theta_0, theta_1)
    ani = FuncAnimation(fig, animation_function, frames=20,
                        interval=steps, repeat=False)

    plt.show()

    print(f'True values: {theta_0}, {theta_1}')
    print(f'Predicted values: {theta_0_hat}, {theta_1_hat}')
    print(f'Error of true parameters: { J([theta_0], [theta_1], X, Y)[0]}')
    print(f'Error of estimated parameters: { J([theta_0_hat], [theta_1_hat], X, Y)[0]}')


if __name__ == '__main__':
    main()
