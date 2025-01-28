import numpy as np
import matplotlib.pyplot as plt


def sphere_function(x):
    return x[0]**2 + x[1]**2


def rosenbrock_function(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def eggholder_function(x):
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + (x[0] / 2) + 47))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))

def plot_function(function, x_range, y_range, title, save=False, experiment_midpoints=None, experiment_bestpoints=None, best=None):
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)
    X, Y = np.meshgrid(x, y)

    Z = function(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.25)
    ax.view_init(90, -90)

    if experiment_midpoints is not None:
        for experiment_no in range(len(experiment_midpoints)):
            midpoints = np.array(experiment_midpoints[experiment_no])
            xs, ys = midpoints[:, 0], midpoints[:, 1]
            zs = function(midpoints.T)
            ax.plot(xs, ys, zs, color='blue')

    if experiment_bestpoints is not None:
        for experiment_no in range(len(experiment_bestpoints)):
            bestpoints = np.array(experiment_bestpoints[experiment_no])
            xs, ys = bestpoints[:, 0], bestpoints[:, 1]
            zs = function(bestpoints.T)
            ax.plot(xs, ys, zs, color='green')

    if best is not None:
        best = np.array(best)
        xs, ys = best[:, 0], best[:, 1]
        zs = function(best.T)
        ax.plot(xs, ys, zs, marker='o', color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    if save:
        plt.tight_layout()
        plt.savefig(f"results/{title}.png")
        plt.close()
    else:
        plt.show()