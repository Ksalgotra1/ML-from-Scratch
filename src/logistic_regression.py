import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # 1. Train the model
    clf = LogisticRegression(eps=1e-5)
    clf.fit(x_train, y_train)

    # 2. Plot the results
    output_plot_path = pred_path.replace('.txt', '.png')
    util.plot(x_train, y_train, clf.theta, output_plot_path)

    # 3. Evaluate on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)

    # 4. Save predictions and print accuracy
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print('Accuracy: {}'.format(np.mean((y_pred > 0.5) == y_eval)))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)  # Initialize theta with zeros

        while True:
            # 1. Hypothesis (Sigmoid)
            z = x @ self.theta
            h_x = 1 / (1 + np.exp(-z))

            # 2. Gradient: (1/m) * X^T * (h - y)
            gradient = (1 / m) * (x.T @ (h_x - y))

            # 3. Hessian: (1/m) * X^T * S * X
            # S is diagonal matrix where S_ii = h_i * (1 - h_i)
            S = np.diag(h_x * (1 - h_x))
            H = (1 / m) * x.T @ S @ x

            # 4. Newton Update: theta := theta - H_inv * gradient
            H_inv = np.linalg.inv(H)
            update = H_inv @ gradient
            self.theta = self.theta - update

            # 5. Check Convergence
            if np.linalg.norm(update) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        z = x @ self.theta
        return 1 / (1 + np.exp(-z))

