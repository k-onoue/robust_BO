#!/usr/bin/env python3

import math
import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch import settings
from linear_operator.operators import MaskedLinearOperator

# Import the StudentTLikelihood class
# Replace 'your_module' with the actual module name where StudentTLikelihood is defined
from .student_t_likelihood import StudentTLikelihood


class ExactStudentTMarginalLogLikelihood(MarginalLogLikelihood):
    r"""
    The exact marginal log likelihood (MLL) for an exact Student-t process with a Student's t-likelihood.

    .. math::
        \begin{align*}
        \log p(y|\nu, K) &= \log \Gamma\left( \frac{\nu + n}{2} \right )
        - \log \Gamma\left( \frac{\nu}{2} \right )
        - \frac{n}{2} \log \left( (\nu - 2)\pi \right )
        - \frac{1}{2} \log |K| \\
        &\quad - \left( \frac{\nu + n}{2} \right ) \log \left( 1 + \frac{ (y - \Phi)^\top K^{-1} (y - \Phi) }{ \nu - 2 } \right )
        \end{align*}

    :param StudentTLikelihood likelihood: The Student's t-likelihood for the model
    :param ExactTP model: The exact Student-t process model

    Example:
        >>> # model is an instance of ExactTP
        >>> # likelihood is an instance of StudentTLikelihood
        >>> mll = ExactStudentTMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(likelihood, StudentTLikelihood):
            raise RuntimeError(
                "Likelihood must be Student's t-likelihood for exact inference with Student-t process"
            )
        super().__init__(likelihood, model)

    def forward(self, function_dist, target, *params, **kwargs):
        r"""
        Computes the exact marginal log likelihood.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`ExactTP` model)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        # Validate input
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactStudentTMarginalLogLikelihood can only operate on Gaussian random variables")

        # Prior mean and covariance
        mean = function_dist.mean
        covar = function_dist.lazy_covariance_matrix

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(target, function_dist.event_shape)
            covar = MaskedLinearOperator(covar, observed.reshape(-1), observed.reshape(-1))
            mean = mean[..., observed]
            target = target[..., observed]
        elif nan_policy == "fill":
            raise ValueError("NaN observation policy 'fill' is not supported by ExactStudentTMarginalLogLikelihood!")

        # Compute residuals
        residual = (target - mean).unsqueeze(-1)  # Shape: (..., n, 1)

        # Solve K^{-1} (y - Phi)
        covar_inv_residual = covar.inv_matmul(residual)  # Shape: (..., n, 1)

        # Compute (y - Phi)^T K^{-1} (y - Phi)
        beta = residual.transpose(-1, -2).matmul(covar_inv_residual).squeeze(-1).squeeze(-1)  # Shape: (...)

        # Compute log determinant of K
        log_det_K = covar.log_det()  # Shape: (...)

        # Degrees of freedom and number of data points
        nu = self.likelihood.df
        n = target.size(-1)

        # Compute lgamma terms
        lgamma_nu_plus_n_over_2 = torch.lgamma(0.5 * (nu + n))
        lgamma_nu_over_2 = torch.lgamma(0.5 * nu)
        lgamma_term = lgamma_nu_plus_n_over_2 - lgamma_nu_over_2

        # Compute log scale
        log_scale = 0.5 * n * torch.log((nu - 2) * math.pi) + 0.5 * log_det_K

        # Compute the log term
        log_term = (0.5 * (nu + n)) * torch.log1p(beta / (nu - 2))

        # Compute final log likelihood
        res = lgamma_term - log_scale - log_term

        # Add any additional terms (e.g., priors)
        res = self._add_other_terms(res, params)

        # Scale by the number of data points
        num_data = n
        return res.div_(num_data)
