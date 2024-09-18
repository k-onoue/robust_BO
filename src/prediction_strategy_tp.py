# import torch
# from linear_operator.operators import RootLinearOperator
# from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
# from gpytorch.utils.memoize import add_to_cache, cached
# from .linear_cg import linear_cg
# from .multivariate_student_t import MultivariateStudentT  # Assuming you have this class

# class StudentTPredictionStrategy(DefaultPredictionStrategy):
#     def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
#         # Set training shape
#         self._train_shape = train_prior_dist.event_shape

#         # Flatten the training labels
#         try:
#             train_labels = train_labels.reshape(
#                 *train_labels.shape[: -len(self.train_shape)], self._train_shape.numel()
#             )
#         except RuntimeError:
#             raise RuntimeError(
#                 "Flattening the training labels failed. The most common cause of this error is "
#                 + "that the shapes of the prior mean and the training labels are mismatched. "
#                 + "The shape of the train targets is {0}, ".format(train_labels.shape)
#                 + "while the reported shape of the mean is {0}.".format(train_prior_dist.mean.shape)
#             )

#         # Initialize the base class
#         self.train_inputs = train_inputs
#         self.train_prior_dist = train_prior_dist
#         self.train_labels = train_labels
#         self.likelihood = likelihood
#         self._last_test_train_covar = None
        
#         # Check if the prior distribution is MultivariateStudentT
#         if isinstance(train_prior_dist, MultivariateStudentT):
#             self.lik_train_train_covar = train_prior_dist.lazy_covariance_matrix
#         else:
#             raise RuntimeError("StudentTPredictionStrategy expects a MultivariateStudentT prior distribution.")
        
#         if root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLinearOperator(root))
        
#         if inv_root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLinearOperator(inv_root))

#     @property
#     def train_shape(self):
#         return self._train_shape

#     @property
#     @cached(name="beta1")
#     def beta1(self) -> torch.Tensor:
#         """
#         Compute beta1 = (y - Φ)^T K^{-1} (y - Φ), which is a key part of the
#         scaling factor for the predictive covariance in the Student's t-process.
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Use inv_matmul to compute K^{-1} (y - Φ) without solving for a dense matrix
#         train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
#         # Compute beta1 = (y - Φ)^T K^{-1} (y - Φ)
#         beta1 = residual.transpose(-1, -2).matmul(train_train_covar_inv).squeeze()
#         return beta1

#     def exact_predictive_mean(self, test_mean, test_train_covar):
#         """
#         Compute the predictive mean for the Student's t-process:
#         Predictive Mean = K_*^T K^{-1} (y - Φ) + Φ_*
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Convert lazy tensor to dense
#         train_train_covar = self.lik_train_train_covar.evaluate()
#         # Use linear_cg to solve K^{-1} (y - Φ)
#         train_train_covar_inv = linear_cg(lambda x: train_train_covar.matmul(x), residual)
#         # Compute predictive mean
#         predictive_mean = test_train_covar.matmul(train_train_covar_inv).squeeze(-1) + test_mean
#         return predictive_mean


#     def exact_predictive_covar(self, test_test_covar, test_train_covar):
#         """
#         Compute the predictive covariance for the Student's t-process:
#         Predictive Covariance = K_{**} - K_*^T K^{-1} K_* scaled by the Student's t scaling factor.
#         """
#         # Convert lazy tensor to dense
#         train_train_covar = self.lik_train_train_covar.evaluate()
#         # Use linear_cg to solve K^{-1} K_*
#         train_train_covar_dense = test_train_covar.transpose(-1, -2).evaluate()  # Ensure dense form
#         train_train_covar_inv = linear_cg(lambda x: train_train_covar.matmul(x), train_train_covar_dense)
#         predictive_covar = test_test_covar - test_train_covar.matmul(train_train_covar_inv)

#         # Compute the scaling factor: (nu + beta1 - 2) / (nu + n - 2)
#         n = self.train_labels.shape[-1]  # Number of training points
#         nu = self.likelihood.df  # Degrees of freedom from the Student's t likelihood
#         beta1 = self.beta1
#         scaling_factor = (nu + beta1 - 2) / (nu + n - 2)

#         # Scale the predictive covariance by the scaling factor
#         predictive_covar = predictive_covar.mul(scaling_factor)

#         return predictive_covar


#     def exact_predictive(self, joint_mean, joint_covar):
#         """
#         Compute both the predictive mean and covariance for the test points.
#         """
#         # Find the components of the distribution that contain test data
#         test_mean = joint_mean[..., self.num_train :]
#         test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
#         test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

#         # Compute the predictive mean and covariance
#         predictive_mean = self.exact_predictive_mean(test_mean, test_train_covar)
#         predictive_covar = self.exact_predictive_covar(test_test_covar, test_train_covar)

#         # Return the predictive distribution as a MultivariateStudentT
#         return MultivariateStudentT(
#             loc=predictive_mean, covariance_matrix=predictive_covar, df=self.likelihood.df
#         )




# import torch
# from linear_operator.operators import RootLinearOperator
# from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
# from gpytorch.utils.memoize import add_to_cache, cached
# from src.multivariate_student_t import MultivariateStudentT  # Assuming you have this class

# class StudentTPredictionStrategy(DefaultPredictionStrategy):
#     def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
#         # Set training shape
#         self._train_shape = train_prior_dist.event_shape

#         # Flatten the training labels
#         try:
#             train_labels = train_labels.reshape(
#                 *train_labels.shape[: -len(self.train_shape)], self._train_shape.numel()
#             )
#         except RuntimeError:
#             raise RuntimeError(
#                 "Flattening the training labels failed. The most common cause of this error is "
#                 + "that the shapes of the prior mean and the training labels are mismatched. "
#                 + "The shape of the train targets is {0}, ".format(train_labels.shape)
#                 + "while the reported shape of the mean is {0}.".format(train_prior_dist.mean.shape)
#             )

#         # Initialize the base class
#         self.train_inputs = train_inputs
#         self.train_prior_dist = train_prior_dist
#         self.train_labels = train_labels
#         self.likelihood = likelihood
#         self._last_test_train_covar = None
        
#         # Check if the prior distribution is MultivariateStudentT
#         if isinstance(train_prior_dist, MultivariateStudentT):
#             self.lik_train_train_covar = train_prior_dist.lazy_covariance_matrix
#         else:
#             raise RuntimeError("StudentTPredictionStrategy expects a MultivariateStudentT prior distribution.")
        
#         if root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLinearOperator(root))
        
#         if inv_root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLinearOperator(inv_root))

#     @property
#     def train_shape(self):
#         return self._train_shape

#     @property
#     @cached(name="beta1")
#     def beta1(self) -> torch.Tensor:
#         """
#         Compute beta1 = (y - Φ)^T K^{-1} (y - Φ), which is a key part of the
#         scaling factor for the predictive covariance in the Student's t-process.
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Use inv_matmul to compute K^{-1} (y - Φ) without solving for a dense matrix
#         train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
#         # Compute beta1 = (y - Φ)^T K^{-1} (y - Φ)
#         beta1 = residual.transpose(-1, -2).matmul(train_train_covar_inv).squeeze()
#         return beta1

#     def exact_predictive_mean(self, test_mean, test_train_covar):
#         """
#         Compute the predictive mean for the Student's t-process:
#         Predictive Mean = K_*^T K^{-1} (y - Φ) + Φ_*
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Use inv_matmul to compute K^{-1} (y - Φ)
#         train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
#         # Compute predictive mean
#         predictive_mean = test_train_covar.matmul(train_train_covar_inv).squeeze(-1) + test_mean
#         return predictive_mean

#     def exact_predictive_covar(self, test_test_covar, test_train_covar):
#         """
#         Compute the predictive covariance for the Student's t-process:
#         Predictive Covariance = K_{**} - K_*^T K^{-1} K_* scaled by the Student's t scaling factor.
#         """
#         # Use inv_matmul to avoid issues with dense matrix solving
#         predictive_covar = test_test_covar - test_train_covar.matmul(
#             self.lik_train_train_covar.inv_matmul(test_train_covar.transpose(-1, -2))
#         )

#         # Compute the scaling factor: (nu + beta1 - 2) / (nu + n - 2)
#         n = self.train_labels.shape[-1]  # Number of training points
#         nu = self.likelihood.df  # Degrees of freedom from the Student's t likelihood
#         beta1 = self.beta1
#         scaling_factor = (nu + beta1 - 2) / (nu + n - 2)

#         # Scale the predictive covariance by the scaling factor
#         predictive_covar = predictive_covar.mul(scaling_factor)

#         return predictive_covar

#     def exact_predictive(self, joint_mean, joint_covar):
#         """
#         Compute both the predictive mean and covariance for the test points.
#         """
#         # Find the components of the distribution that contain test data
#         test_mean = joint_mean[..., self.num_train :]
#         test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
#         test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

#         # Compute the predictive mean and covariance
#         predictive_mean = self.exact_predictive_mean(test_mean, test_train_covar)
#         predictive_covar = self.exact_predictive_covar(test_test_covar, test_train_covar)

#         # Return the predictive distribution as a MultivariateStudentT
#         return MultivariateStudentT(predictive_mean, predictive_covar, self.likelihood.df)


# import torch
# from linear_operator.operators import RootLinearOperator
# from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
# from gpytorch.utils.memoize import add_to_cache, cached
# from .linear_cg import linear_cg
# from .multivariate_student_t import MultivariateStudentT  # Assuming you have this class

# class StudentTPredictionStrategy(DefaultPredictionStrategy):
#     def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
#         # Set training shape
#         self._train_shape = train_prior_dist.event_shape

#         # Flatten the training labels
#         try:
#             train_labels = train_labels.reshape(
#                 *train_labels.shape[: -len(self.train_shape)], self._train_shape.numel()
#             )
#         except RuntimeError:
#             raise RuntimeError(
#                 "Flattening the training labels failed. The most common cause of this error is "
#                 + "that the shapes of the prior mean and the training labels are mismatched. "
#                 + "The shape of the train targets is {0}, ".format(train_labels.shape)
#                 + "while the reported shape of the mean is {0}.".format(train_prior_dist.mean.shape)
#             )

#         # Initialize the base class
#         self.train_inputs = train_inputs
#         self.train_prior_dist = train_prior_dist
#         self.train_labels = train_labels
#         self.likelihood = likelihood
#         self._last_test_train_covar = None
        
#         # Check if the prior distribution is MultivariateStudentT
#         if isinstance(train_prior_dist, MultivariateStudentT):
#             self.lik_train_train_covar = train_prior_dist.lazy_covariance_matrix
#         else:
#             raise RuntimeError("StudentTPredictionStrategy expects a MultivariateStudentT prior distribution.")
        
#         if root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLinearOperator(root))
        
#         if inv_root is not None:
#             add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLinearOperator(inv_root))

#     @property
#     def train_shape(self):
#         return self._train_shape

#     @property
#     @cached(name="beta1")
#     def beta1(self) -> torch.Tensor:
#         """
#         Compute beta1 = (y - Φ)^T K^{-1} (y - Φ), which is a key part of the
#         scaling factor for the predictive covariance in the Student's t-process.
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Use inv_matmul to compute K^{-1} (y - Φ) without solving for a dense matrix
#         train_train_covar_inv = self.lik_train_train_covar.inv_matmul(residual)
#         # Compute beta1 = (y - Φ)^T K^{-1} (y - Φ)
#         beta1 = residual.transpose(-1, -2).matmul(train_train_covar_inv).squeeze()
#         return beta1

#     def exact_predictive_mean(self, test_mean, test_train_covar):
#         """
#         Compute the predictive mean for the Student's t-process:
#         Predictive Mean = K_*^T K^{-1} (y - Φ) + Φ_*
#         """
#         residual = (self.train_labels - self.train_prior_dist.mean).unsqueeze(-1)
#         # Use linear_cg to solve K^{-1} (y - Φ)
#         # train_train_covar_inv = linear_cg(self.lik_train_train_covar, residual)
#         train_train_covar = self.lik_train_train_covar.evaluate() 
#         train_train_covar_inv = linear_cg(lambda x: train_train_covar.matmul(x), residual)
#         # Compute predictive mean
#         predictive_mean = test_train_covar.matmul(train_train_covar_inv).squeeze(-1) + test_mean
#         return predictive_mean


#     def exact_predictive_covar(self, test_test_covar, test_train_covar):
#         """
#         Compute the predictive covariance for the Student's t-process:
#         Predictive Covariance = K_{**} - K_*^T K^{-1} K_* scaled by the Student's t scaling factor.
#         """
#         # Use linear_cg to solve K^{-1} K_*
#         # train_train_covar_inv = linear_cg(self.lik_train_train_covar, test_train_covar.transpose(-1, -2))
#         train_train_covar = self.lik_train_train_covar.evaluate() 
#         train_train_covar_inv = linear_cg(lambda x: train_train_covar.matmul(x), test_train_covar.transpose(-1, -2))
#         predictive_covar = test_test_covar - test_train_covar.matmul(train_train_covar_inv)

#         # Compute the scaling factor: (nu + beta1 - 2) / (nu + n - 2)
#         n = self.train_labels.shape[-1]  # Number of training points
#         nu = self.likelihood.df  # Degrees of freedom from the Student's t likelihood
#         beta1 = self.beta1
#         scaling_factor = (nu + beta1 - 2) / (nu + n - 2)

#         # Scale the predictive covariance by the scaling factor
#         predictive_covar = predictive_covar.mul(scaling_factor)

#         return predictive_covar


#     def exact_predictive(self, joint_mean, joint_covar):
#         """
#         Compute both the predictive mean and covariance for the test points.
#         """
#         # Find the components of the distribution that contain test data
#         test_mean = joint_mean[..., self.num_train :]
#         test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
#         test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

#         # Compute the predictive mean and covariance
#         predictive_mean = self.exact_predictive_mean(test_mean, test_train_covar)
#         predictive_covar = self.exact_predictive_covar(test_test_covar, test_train_covar)

#         # Return the predictive distribution as a MultivariateStudentT
#         return MultivariateStudentT(predictive_mean, predictive_covar, self.likelihood.df)




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
        # Convert lazy tensors to dense
        train_train_covar_dense = self.lik_train_train_covar.evaluate()
        test_train_covar_dense = test_train_covar.evaluate().transpose(-1, -2)

        # Compute K_{**} - K_*^T K^{-1} K_* using inv_matmul with dense tensors
        train_train_covar_inv = torch.cholesky_inverse(train_train_covar_dense.cholesky())
        predictive_covar = test_test_covar - test_train_covar.matmul(train_train_covar_inv.matmul(test_train_covar_dense))

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