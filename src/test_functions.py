import gpytorch
import torch
from torch import Tensor


def negate_function(func):
    def negated_func(x):
        return -func(x)

    return negated_func


def sinusoidal_synthetic(x: Tensor) -> Tensor:
    r"""
    Computes the function f(x) = -(x-1)^2 * \sin(3x + 5^{-1} + 1) for a given tensor input x.

    Args:
        x (Tensor): Input tensor of shape (N, 1) where N is the number of data points.
                    If the input is (N,), it will be automatically reshaped to (N, 1).

    Returns:
        Tensor: Output tensor of shape (N, 1) representing the computed values of f(x).

    f(x) = -(x-1)^2 \sin(3x + 5^{-1} + 1)
    """
    # If the input is of shape (N,), reshape it to (N, 1)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    # If the input is already of shape (N, 1), proceed as is
    elif x.ndim == 2 and x.shape[1] == 1:
        pass
    # If the input shape is invalid, raise an error
    else:
        raise ValueError("Input must be of shape (N,) or (N, 1)")

    # Compute the function
    term1 = -((x - 1) ** 2)
    term2 = torch.sin(3 * x + 1 / 5 + 1)
    val = term1 * term2
    return val


def branin_hoo(x: Tensor) -> Tensor:
    r"""
    Computes the Branin-Hoo function, typically used for benchmarking optimization algorithms.

    Description:
    The Branin-Hoo function is a commonly used test function for optimization algorithms.
    It has three global minima.

    Input Domain:
    The function is usually evaluated on the domain {(x1, x2) : 0 <= x1 <= 15, -5 <= x2 <= 15}.
    It is a two-dimensional function.

    Global Minimum:
    The global minimum values occur at:
    - (x1, x2) = (-π, 12.275), (π, 2.275), and (9.42478, 2.475).
    At these points, f(x) = 0.397887.

    The function is defined as:

    f(x) = (x_2 - (5.1 / (4 * pi^2)) * x_1^2 + (5 / pi) * x_1 - 6)^2 + 10 * (1 - 1 / (8 * pi)) * cos(x_1) + 10

    Args:
        x (Tensor): Input tensor of shape (N, 2), where N is the number of data points, and
                    each data point contains 2 dimensions [x_1, x_2].

    Returns:
        Tensor: Output tensor of shape (N, 1), representing the computed values of the Branin-Hoo function.

    Raises:
        ValueError: If the input tensor is not two-dimensional or does not have exactly 2 features per data point.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            "Input tensor must be two-dimensional with exactly two features per data point."
        )

    # Extract x1 and x2
    x1 = x[:, 0]
    x2 = x[:, 1]

    pi = torch.pi

    # Compute the Branin-Hoo function components
    term1 = (x2 - (5.1 / (4 * pi**2)) * x1**2 + (5 / pi) * x1 - 6) ** 2
    term2 = 10 * (1 - 1 / (8 * pi)) * torch.cos(x1)

    # Final value computation and reshaping to (N, 1)
    val = (term1 + term2 + 10).unsqueeze(1)

    return val


def hartmann6(x: Tensor) -> Tensor:
    r"""
    Computes the 6-dimensional Hartmann function, typically used for benchmarking optimization algorithms.

    Description:
    The 6-dimensional Hartmann function has 6 local minima.

    Input Domain:
    The function is usually evaluated on the hypercube x_i ∈ (0, 1) for all i = 1, ..., 6.

    Global Minimum:
    The global minimum value is f(x*) = -3.32237 at:
    x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    The function is defined as:

    f(x) = - sum_{i=1}^{4} alpha_i * exp( - sum_{j=1}^{6} A_ij * (x_j - P_ij)^2 )

    Args:
        x (Tensor): Input tensor of shape (N, 6), where N is the number of data points, and
                    each data point contains 6 dimensions.

    Returns:
        Tensor: Output tensor of shape (N, 1), representing the computed values of the Hartmann-6 function.

    Raises:
        ValueError: If the input tensor is not two-dimensional or does not have exactly 6 features per data point.
    """
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(
            "Input tensor must be two-dimensional with exactly six features per data point."
        )

    # Define constants for the Hartmann function
    alpha = torch.tensor([1.00, 1.20, 3.00, 3.20], dtype=torch.float32)
    A = torch.tensor(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ],
        dtype=torch.float32,
    )
    P = torch.tensor(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ],
        dtype=torch.float32,
    )

    # Compute the Hartmann function
    outer_sum = 0
    for i in range(4):
        inner_sum = torch.sum(A[i] * (x - P[i]) ** 2, dim=1)
        outer_sum += alpha[i] * torch.exp(-inner_sum)

    # Negate the result to match the typical form of the Hartmann-6 function
    val = -outer_sum.unsqueeze(1)

    return val


# if __name__ == "__main__":

#     gp_info = {
#         "kernel": {
#             "type": "rational_quadratic",  # matern or rational_quadratic
#             "alpha_mean": 0.1,
#             "alpha_std": 0.05,
#             "lengthscale_mean": 0.1,
#             "lengthscale_std": 0.05,
#             # outputscale を指定しない場合は使用しない
#             # "outputscale_mean": 1.0,
#             # "outputscale_std": 0.1,
#         },
#         "mean": {
#             "type": "constant",
#             "value": 0.0,
#         },
#     }

#     search_space = torch.tensor([[-10] * 6, [10] * 6]).to(torch.float32)

#     # クラスのインスタンスを作成
#     gpsample = GPSamplePathWithOutliers(gp_info=gp_info, search_space=search_space)
#     print(gpsample)

#     # テストデータでの予測
#     x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                       [2, -5, 0, 0.3, -9, 8.6]], dtype=torch.float32)

#     with torch.no_grad():
#         mean = gpsample(x)
#         print("Predicted mean:", mean)
#         print(f"Output shape: {mean.shape}")


# class GPSamplePathWithOutliers(gpytorch.models.ExactGP):
#     def __init__(
#         self,
#         gp_info: dict,
#         search_space: Tensor,
#         outlier_ratio: float = 0.1,
#         training_iterations: int = 50,
#     ) -> None:
#         self.search_space = search_space
#         self.training_iterations = training_iterations
#         self.gp_info = gp_info
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()

#         # 初期データを生成
#         train_X, train_Y = self._generate_initial_data()
#         super(GPSamplePathWithOutliers, self).__init__(train_X, train_Y, likelihood)

#         self.outlier_ratio = outlier_ratio

#         self.mean_module = self._build_mean_module()
#         self.covar_module = self._build_covariance_module()

#         # モデルの訓練を初期化時に実行
#         self.fit()

#     def _generate_initial_data(self) -> tuple:
#         num_samples = 5
#         num_features = self.search_space.shape[1]

#         lower_bounds = self.search_space[0]
#         upper_bounds = self.search_space[1]
#         train_X = lower_bounds + (upper_bounds - lower_bounds) * torch.rand((num_samples, num_features))

#         train_Y = sinusoidal_synthetic(train_X)

#         return train_X, train_Y

#     def _build_mean_module(self) -> gpytorch.means.Mean:
#         mean_info = self.gp_info["mean"]
#         if mean_info["type"] == "constant":
#             mean_module = gpytorch.means.ConstantMean()
#             mean_module.initialize(constant=mean_info["value"])
#         else:
#             raise ValueError("Invalid mean type specified.")
#         return mean_module

#     def _build_covariance_module(self) -> gpytorch.kernels.Kernel:
#         kernel_info = self.gp_info["kernel"]

#         # ベースカーネルを構築
#         if kernel_info["type"] == "matern":
#             base_kernel = gpytorch.kernels.MaternKernel(
#                 nu=kernel_info["nu"],
#                 lengthscale_prior=gpytorch.priors.NormalPrior(kernel_info["lengthscale_mean"], kernel_info["lengthscale_std"]),
#             )
#         elif kernel_info["type"] == "rational_quadratic":
#             base_kernel = gpytorch.kernels.RQKernel(
#                 alpha_prior=gpytorch.priors.NormalPrior(kernel_info["alpha_mean"], kernel_info["alpha_std"]),
#                 lengthscale_prior=gpytorch.priors.NormalPrior(kernel_info["lengthscale_mean"], kernel_info["lengthscale_std"]),
#             )
#         else:
#             raise ValueError("Invalid kernel type specified.")

#         # outputscale が gp_info に指定されているかどうかを確認
#         if "outputscale" in kernel_info:
#             covariance_module = gpytorch.kernels.ScaleKernel(
#                 base_kernel,
#                 outputscale_prior=gpytorch.priors.NormalPrior(kernel_info["outputscale_mean"], kernel_info["outputscale_std"])
#             )
#         else:
#             # outputscale を使用しない場合
#             covariance_module = base_kernel

#         return covariance_module

#     def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
#         # 訓練および推論時に使用するガウス過程の分布を返す
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#     def fit(self) -> None:
#         # モデルの訓練プロセス
#         self.train()
#         self.likelihood.train()

#         # Marginal Log Likelihood を設定
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

#         for i in range(self.training_iterations):
#             optimizer.zero_grad()
#             output = self.forward(self.train_inputs[0])  # forward メソッドを使用
#             loss = -mll(output, self.train_targets)

#             # 平均を取ってスカラーにする
#             loss = loss.mean()

#             loss.backward()
#             print(f"Iteration {i + 1}/{self.training_iterations} - Loss: {loss.item():.3f}")
#             optimizer.step()

#         print("Training completed!")

#         # カーネルが ScaleKernel でラップされているか確認し、ハイパーパラメータを取得
#         if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
#             self.gp_info["kernel"]["lengthscale"] = self.covar_module.base_kernel.lengthscale.item()
#             self.gp_info["kernel"]["outputscale"] = self.covar_module.outputscale.item()
#             if hasattr(self.covar_module.base_kernel, "alpha"):
#                 self.gp_info["kernel"]["alpha"] = self.covar_module.base_kernel.alpha.item()
#         else:
#             # ScaleKernel を使用していない場合
#             self.gp_info["kernel"]["lengthscale"] = self.covar_module.lengthscale.item()
#             if hasattr(self.covar_module, "alpha"):
#                 self.gp_info["kernel"]["alpha"] = self.covar_module.alpha.item()

#         print("Updated kernel hyperparameters in gp_info:", self.gp_info)


#     def __call__(self, x: Tensor) -> Tensor:
#         with torch.no_grad():
#             output = self.forward(x)
#             return output.mean


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     gp_info = {
#         "kernel": {
#             "type": "rational_quadratic",  # matern or rational_quadratic
#             "alpha_mean": 0.1,
#             "alpha_std": 0.05,
#             "lengthscale_mean": 0.1,
#             "lengthscale_std": 0.05,
#             # outputscale を指定しない場合は使用しない
#             # "outputscale_mean": 1.0,
#             # "outputscale_std": 0.1,
#         },
#         "mean": {
#             "type": "constant",
#             "value": 0.0,
#         },
#     }

#     search_space = torch.tensor([[-10], [10]]).to(torch.float32)

#     # クラスのインスタンスを作成
#     gpsample = GPSamplePathWithOutliers(gp_info=gp_info, search_space=search_space)
#     print(gpsample)

#     # 1次元入力データの作成
#     x = torch.linspace(-10, 10, 100).unsqueeze(1)  # 100点を [-10, 10] の範囲で生成
#     y = gpsample(x)

#     x = x.detach().numpy().flatten()
#     y = y.detach().numpy()

#     # 予測結果を描画
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, label="Predicted mean")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.title("Predicted Mean Function")
#     plt.show()
