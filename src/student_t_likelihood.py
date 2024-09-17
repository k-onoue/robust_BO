#!/usr/bin/env python3
# import math
# import warnings
from typing import Any, Optional, Union

import torch
from linear_operator.operators import LinearOperator, MaskedLinearOperator
from torch import Tensor
# from torch.distributions import Distribution

from gpytorch import settings
# from gpytorch.constraints import Interval, Positive
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import Prior
# from gpytorch.utils.warnings import GPInputWarning
from gpytorch.likelihoods.likelihood import Likelihood
# from gpytorch.likelihoods.noise_models import FixedNoise, HomoskedasticNoise, Noise
from gpytorch.likelihoods.noise_models import HomoskedasticNoise, Noise

# Import the MultivariateStudentT class you defined earlier
from .multivariate_student_t import MultivariateStudentT


class _StudentTLikelihoodBase(Likelihood):
    """Base class for Student-t Likelihoods."""

    has_analytic_marginal = True

    def __init__(
        self,
        noise_covar: Noise,
        df: Union[float, Tensor] = 3.0,
        df_prior: Optional[Prior] = None,
        df_constraint: Optional[Interval] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.noise_covar = noise_covar

        if df_constraint is None:
            # Ensure degrees of freedom > 2 for variance to be defined
            df_constraint = Interval(2.0 + 1e-6, 1e8)

        self.register_parameter(
            name="raw_df", parameter=torch.nn.Parameter(torch.tensor(float(df)).log())
        )
        if df_prior is not None:
            self.register_prior(
                "df_prior", df_prior, lambda m: m.df, lambda m, v: m._set_df(v)
            )

        self.register_constraint("raw_df", df_constraint)

    @property
    def df(self) -> Tensor:
        return self.raw_df_constraint.transform(self.raw_df).exp()

    @df.setter
    def df(self, value: Tensor) -> None:
        self._set_df(value)

    def _set_df(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        self.initialize(raw_df=self.raw_df_constraint.inverse_transform(value.log()))

    def _shaped_noise_covar(
        self, base_shape: torch.Size, *params: Any, **kwargs: Any
    ) -> Union[Tensor, LinearOperator]:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def forward(
        self, function_samples: Tensor, *params: Any, **kwargs: Any
    ) -> MultivariateStudentT:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs)
        # Compute the scale_tril (Cholesky factor)
        if isinstance(noise, LinearOperator):
            scale_tril = noise.cholesky()
        else:
            scale_tril = noise.sqrt().diag_embed()
        return MultivariateStudentT(
            loc=function_samples, scale_tril=scale_tril, df=self.df
        )

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(observations, marginal.event_shape)
            marginal = MultivariateStudentT(
                loc=marginal.loc[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    marginal.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                ),
                df=self.df,
            )
            observations = observations[..., observed]
        elif nan_policy == "fill":
            missing = torch.isnan(observations)
            observations = settings.observation_nan_policy._fill_tensor(observations)

        res = marginal.log_prob(observations)

        if nan_policy == "fill":
            res = res * ~missing

        # Do appropriate summation for multitask Student-t likelihoods
        num_event_dim = len(marginal.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def marginal(
        self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultivariateStudentT:
        mean = function_dist.mean
        covar = function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return MultivariateStudentT(loc=mean, covariance_matrix=full_covar, df=self.df)


class StudentTLikelihood(_StudentTLikelihoodBase):
    r"""
    The likelihood for regression using a Student's t-distribution.
    Assumes a homoskedastic noise model with Student's t-distributed errors:

    .. math::
        p(y \mid f) = \text{StudentT}(y \mid f, \sigma^2, \nu)

    where :math:`\sigma^2` is a scale parameter and :math:`\nu` is the degrees of freedom.

    :param df: Degrees of freedom parameter :math:`\nu`.
    :param df_prior: Prior for degrees of freedom :math:`\nu`.
    :param df_constraint: Constraint for degrees of freedom :math:`\nu`.
    :param noise_prior: Prior for scale parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for scale parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (scale)
    :ivar torch.Tensor df: :math:`\nu` parameter (degrees of freedom)
    """

    def __init__(
        self,
        df: Union[float, Tensor] = 3.0,
        df_prior: Optional[Prior] = None,
        df_constraint: Optional[Interval] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(
            noise_covar=noise_covar, df=df, df_prior=df_prior, df_constraint=df_constraint
        )

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def marginal(
        self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> MultivariateStudentT:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)



# if __name__ == "__main__":
#     likelihood = StudentTLikelihood(df=3.0)
#     print(likelihood)