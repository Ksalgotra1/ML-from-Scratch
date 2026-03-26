import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Predict and Save
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d') # Save as binary predictions

    # (Optional) Print Accuracy
    accuracy = np.mean((y_pred > 0.5) == y_eval)
    print(f"GDA Accuracy: {accuracy * 100:.2f}%")
    
    util.plot(x_eval, y_eval, model.theta, "output/p01e_{}.png".format(pred_path[-5]))


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape

        phi =  np.mean(y)


        mu_0 = np.mean(x[y == 0], axis=0) 
        mu_1 = np.mean(x[y == 1], axis=0)

        # 3. Calculate Sigma
        # We need to subtract the appropriate mean from each example
        # Create a matrix of means matching the shape of x
        mu_matrix = np.where(y[:, None] == 0, mu_0, mu_1) # shape (m, n)
        x_centered = x - mu_matrix
        
        # Sigma = (1/m) * (X_centered.T @ X_centered)
        sigma = (x_centered.T @ x_centered) / m

        # 4. Calculate Theta and Theta_0
        # Invert Sigma
        sigma_inv = np.linalg.inv(sigma)
        
        # Theta = Sigma_inv * (mu_1 - mu_0)
        self.theta = sigma_inv @ (mu_1 - mu_0)
        
        # Theta_0 (Scalar terms)
        # Note: We use .dot for vector-matrix-vector multiplication
        term1 = 0.5 * (mu_0.T @ sigma_inv @ mu_0)
        term2 = -0.5 * (mu_1.T @ sigma_inv @ mu_1)
        term3 = np.log(phi / (1 - phi))
        
        self.theta_0 = term1 + term2 + term3
        
        # Combine into one parameter vector for compatibility with util.plot
        # The first element is intercept, the rest are weights
        self.theta = np.insert(self.theta, 0, self.theta_0)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # Since we added theta_0 to self.theta at index 0, we need to handle x carefully
        # If x doesn't have an intercept column, we can compute manually:
        
        # Option 1: Use the sigmoid function manually
        # theta_0 is self.theta[0], weights are self.theta[1:]
        theta_0 = self.theta[0]
        weights = self.theta[1:]
        
        z = (x @ weights) + theta_0
        return 1 / (1 + np.exp(-z))
