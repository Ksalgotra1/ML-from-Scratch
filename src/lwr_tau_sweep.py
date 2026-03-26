import matplotlib.pyplot as plt
import numpy as np
import util

from locally_weighted_regression import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = None
    best_mse = float('inf')

    plt.figure(figsize=(15, 5))
    
    for i, tau in enumerate(tau_values):
        print(f"Evaluating Tau = {tau}...")
        
        # 1. Initialize and Fit
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        
        # 2. Predict on Validation Set
        y_valid_pred = model.predict(x_valid)
        
        # 3. Compute Validation MSE
        mse = np.mean((y_valid - y_valid_pred) ** 2)
        print(f"Tau: {tau} | Validation MSE: {mse}")
        
        # 4. Check if this is the best tau
        if mse < best_mse:
            best_mse = mse
            best_tau = tau

        # 5. Plotting (Subplot for each tau)
        ax = plt.subplot(1, len(tau_values), i + 1)
        # Plot raw validation data (red x)
        ax.plot(x_valid[:, 1], y_valid, 'rx', label='Valid Data')
        # Plot predictions (blue o) - sorting for cleaner line if you used plot, but 'bo' is fine
        ax.plot(x_valid[:, 1], y_valid_pred, 'bo', label=f'Pred (tau={tau})')
        
        ax.set_title(f"Tau = {tau}, MSE = {mse:.4f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if i == 0: ax.legend()

    # Save the comparison plot
    plt.savefig('output/p05c_tau_sweep.png')
    print(f"Best Tau found: {best_tau} with MSE: {best_mse}")

    # 6. Run on Test Set with the Best Tau
    print(f"Running best model (tau={best_tau}) on Test Set...")
    best_model = LocallyWeightedLinearRegression(best_tau)
    best_model.fit(x_train, y_train)
    y_test_pred = best_model.predict(x_test)

    # 7. Compute and Print Test MSE
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    print(f"Test Set MSE: {test_mse}")

    # 8. Save Predictions
    np.savetxt(pred_path, y_test_pred)


