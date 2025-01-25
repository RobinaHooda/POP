import numpy as np
import matplotlib.pyplot as plt
import json


def sphere_function(x):
    return x[0]**2 + x[1]**2


def rosenbrock_function(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def eggholder_function(x):
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + (x[0] / 2) + 47))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))


def plot_function(function, x_range, y_range, title, points=None):
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)
    X, Y = np.meshgrid(x, y)

    Z = function(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5)

    if points is not None:
        points = np.array(points)
        xs, ys = points[:, 0], points[:, 1]
        zs = function(points.T)
        ax.plot(xs, ys, zs)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    plt.show()


points = []
with open("results/genetic_algorithm_results.json", 'r') as file:
    data = json.load(file)
for entry in data:
    if "midpoints" in entry:
        points.extend(entry["midpoints"])


#plot_function(sphere_function, (-5.12, 5), (-5.12, 5.12), "Sphere function")
plot_function(eggholder_function, (-512, 512), (-512, 512), "Eggholder function", points)
#plot_function(rosenbrock_function, (-5, 10), (-5, 10), "Rosenbrock function")
