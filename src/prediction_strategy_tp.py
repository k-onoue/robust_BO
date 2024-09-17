import torch
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached
# from linear_operator.operators import LinearOperator

class StudentTPredictionStrategy(DefaultPredictionStrategy):
    @property
    @cached(name="beta1")
    def beta1(self) -> torch.Tensor:
        # Compute beta1 = (y - Φ)^T K^{-1} (y - Φ)
        residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
        train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
        beta1 = residual.transpose(-1, -2).matmul(train_train_covar_inv).squeeze()
        return beta1

    def exact_predictive_mean(self, test_mean, test_train_covar):
        # Compute K_*^T K^{-1} (y - Φ)
        residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
        train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
        predictive_mean = test_train_covar.matmul(train_train_covar_inv).squeeze(-1) + test_mean
        return predictive_mean

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        # Compute K_{**} - K_*^T K^{-1} K_*
        predictive_covar = test_test_covar - test_train_covar.matmul(
            self.lik_train_train_covar.inv_matmul(test_train_covar.transpose(-1, -2))
        )

        # Scale predictive covariance
        n = self.train_labels.shape[-1]
        nu = self.likelihood.df
        beta1 = self.beta1
        scaling_factor = (nu + beta1 - 2) / (nu + n - 2)
        predictive_covar = predictive_covar.mul(scaling_factor)

        return predictive_covar

    def exact_predictive(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
        test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

        # Compute predictive mean and covariance
        predictive_mean = self.exact_predictive_mean(test_mean, test_train_covar)
        predictive_covar = self.exact_predictive_covar(test_test_covar, test_train_covar)

        return predictive_mean, predictive_covar




# if __name__ == "__main__":
#     prediction_strategy = StudentTPredictionStrategy()
#     print(prediction_strategy)