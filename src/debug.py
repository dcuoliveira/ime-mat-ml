import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.autograd import grad

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.autograd import grad

def generate_sem_data(n, mu_x, sigma_x, sigma_y, sigma_x2, rho=0):
    X1 = np.random.normal(mu_x, sigma_x, n)
    # If rho is not 0, we introduce correlation between X1 and noise in Y
    noise_y = np.random.normal(0, sigma_y * np.sqrt(1 - rho**2), n) + rho * X1
    Y = X1 + noise_y
    X2 = Y + np.random.normal(0, sigma_x2, n)
    return np.column_stack((Y, X1, X2))

class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        self.best_reg = 0
        self.best_err = float('inf')

        # Assumes the last environment is the validation set
        x_val, y_val = environments[-1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = torch.mean((x_val @ self.solution() - y_val) ** 2).item()

            if args["verbose"]:
                print(f"IRM (reg={reg:.3f}) has {err:.3f} validation error.")

            if err < self.best_err:
                self.best_err = err
                self.best_reg = reg
                self.best_phi = self.phi.clone()

        self.phi = self.best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].shape[1]

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones((dim_x, 1), requires_grad=True)

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        mse_loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                preds = x_e @ self.phi @ self.w
                error_e = mse_loss(preds, y_e)
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e.item()

            opt.zero_grad()
            loss = reg * error + (1 - reg) * penalty
            loss.backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = ' '.join(f'{w:.2f}' for w in self.solution().view(-1))
                print(f"{iteration:05d} | {reg:.5f} | {error:.5f} | {penalty:.5f} | {w_str}")

    def solution(self):
        return self.phi @ self.w

def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)


if __name__ == "__main__":
    # Training dataset with variances as described in training environments
    n_train = 1000  # Number of samples in training

    train_env1 = generate_sem_data(n_train, 0, np.sqrt(10), np.sqrt(10), 1)
    train_env2 = generate_sem_data(n_train, 0, np.sqrt(20), np.sqrt(20), 1)
    train_data = np.vstack((train_env1, train_env2))

    train_env3 = generate_sem_data(n_train, 0, np.sqrt(10), np.sqrt(10), 1)

    # Test dataset with variance shift
    test_var_shift = generate_sem_data(n_train//3, 0, np.sqrt(50), np.sqrt(50), 1)
    test_data_var_shift = np.vstack(test_var_shift)

    # Test dataset with mean shift
    test_mean_shift = generate_sem_data(n_train, 15, np.sqrt(20), np.sqrt(20), 1)
    test_data_mean_shift = np.vstack(test_mean_shift)

    # Test dataset with correlation shift
    test_corr_shift = generate_sem_data(n_train, 0, np.sqrt(20), np.sqrt(20), 1, rho=0.7)
    test_data_corr_shift = np.vstack(test_corr_shift)

    # Combine into dataframes for easier handling
    columns = ['Y', 'X1', 'X2']
    train_df = pd.DataFrame(train_data, columns=columns)
    train_df2 = pd.DataFrame(train_data, columns=columns)
    test_var_shift_df = pd.DataFrame(test_data_var_shift, columns=columns)
    test_mean_shift_df = pd.DataFrame(test_data_mean_shift, columns=columns)
    test_corr_shift_df = pd.DataFrame(test_data_corr_shift, columns=columns)

    # Convert the datasets to PyTorch tensors
    train_tensors1 = [[train_env1[:, 0:2], train_env1[:, 2:]], [train_env2[:, 0:2], train_env2[:, 2:]]]
    train_tensors2 = [train_env3[:, 0:2], train_env3[:, 2:]]
    test_var_shift_tensors = [test_var_shift[:, 0:2], test_var_shift[:, 2:]]
    test_mean_shift_tensors = [test_mean_shift[:, 0:2], test_mean_shift[:, 2:]]
    test_corr_shift_tensors = [test_corr_shift[:, 0:2], test_corr_shift[:, 2:]]

    train_tensors1 = [(to_tensor(env[0]), to_tensor(env[1])) for env in train_tensors1]
    train_tensors2 = [to_tensor(tensor_obj) for tensor_obj in train_tensors2]
    test_var_shift_tensors = [to_tensor(tensor_obj) for tensor_obj in test_var_shift_tensors]
    test_mean_shift_tensors = [to_tensor(tensor_obj) for tensor_obj in test_mean_shift_tensors]
    test_corr_shift_tensors = [to_tensor(tensor_obj) for tensor_obj in test_corr_shift_tensors]


    # Define arguments for training
    args = {
        "lr": 1e-3,
        "n_iterations": 5000,
        "verbose": False
    }

    # Initialize the IRM model and train on the training environments
    irm_model = InvariantRiskMinimization(train_tensors1, args)

    # Evaluate on the test datasets
    names = ["standard", 'variance_shift', 'mean_shift', 'correlation_shift']
    test_datasets = [train_tensors2, test_var_shift_tensors, test_mean_shift_tensors, test_corr_shift_tensors]
    for i, (x_test, y_test) in enumerate(test_datasets):
        y_pred = x_test @ irm_model.solution()
        mse =  mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Test {names[i]}, MSE: {mse.item()}, MAE: {mae.item()}")