import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load datasets
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Train (store data)
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Predict on validation set
    y_pred = model.predict(x_eval)


    # Plot
    # Force y_pred to be a flat 1D array
    y_pred = np.array(y_pred).flatten()

    # Now calculate MSE and plot
    mse = np.mean((y_eval - y_pred)**2)
    print(f"MSE: {mse}")

    plt.figure()
    # Plot valid data (red 'x')
    plt.plot(x_eval[:, 1], y_eval, 'rx', label='Validation Data')
    # Plot predictions (blue 'o')
    plt.plot(x_eval[:, 1], y_pred, 'bo', label='LWLR Predictions') 
    plt.legend()
    plt.savefig('output/p05b.png')



class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y


    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m, n = np.shape(x)
        y_pred = np.zeros(m)

        for i in range(m):
            diff = self.x - x[i]
            w = np.exp(-np.sum(diff**2, axis = 1)/(2 * self.tau**2))

            W = np.diag(w)

            XTWX = self.x.T @ W @ self.x
            XTWy = self.x.T @ W @ self.y
            theta = np.linalg.solve(XTWX, XTWy)

            y_pred[i] = x[i] @ theta
        
        return y_pred
