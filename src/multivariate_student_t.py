from __future__ import annotations

import math
from numbers import Number
from typing import Optional, Union

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import LinearOperator
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import _standard_normal, lazy_property



class MultivariateStudentT(Distribution):
    r"""
    Constructs a multivariate Student's t-distribution with degrees of freedom `df`,
    mean vector `loc`, and scale matrix `scale_tril` or covariance matrix.

    This class is designed to be used with GPyTorch, and supports lazy evaluation
    of the scale matrix using LinearOperators.

    :param loc: Mean vector of the distribution (`... x N`).
    :param scale_tril: Lower-triangular factor of the scale matrix (`... x N x N`).
        Either `scale_tril` or `covariance_matrix` must be specified.
    :param covariance_matrix: Covariance matrix of the distribution (`... x N x N`).
        Either `scale_tril` or `covariance_matrix` must be specified.
    :param df: Degrees of freedom of the distribution.
    :param validate_args: Whether to validate input arguments (default: `False`).
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_tril": constraints.independent(constraints.lower_cholesky, 1),
        "covariance_matrix": constraints.positive_definite,
        "df": constraints.positive,
    }
    support = constraints.real_vector

    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Optional[Union[Tensor, LinearOperator]] = None,
        scale_tril: Optional[Tensor] = None,
        df: Union[float, Tensor] = 1.0,
        validate_args: bool = False,
    ):
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril must be specified.")
        self.loc = loc
        self.df = df
        self._validate_args = validate_args

        if covariance_matrix is not None:
            self._islazy = isinstance(covariance_matrix, LinearOperator)
            if self._islazy:
                self._covariance_matrix = covariance_matrix
            else:
                self.covariance_matrix = covariance_matrix
        else:
            self._islazy = False
            self.scale_tril = scale_tril

        batch_shape = torch.broadcast_shapes(
            self.loc.shape[:-1],
            self.covariance_matrix.shape[:-2] if covariance_matrix is not None else self.scale_tril.shape[:-2],
        )
        event_shape = self.loc.shape[-1:]

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def islazy(self) -> bool:
        return self._islazy

    @property
    def mean(self) -> Tensor:
        if self.df > 1:
            return self.loc
        else:
            return torch.full_like(self.loc, float("nan"))

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        if self.islazy:
            return self._covariance_matrix.to_dense()
        else:
            return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    @lazy_property
    def lazy_covariance_matrix(self) -> LinearOperator:
        if self.islazy:
            return self._covariance_matrix
        else:
            return to_linear_operator(self.covariance_matrix)

    @property
    def variance(self) -> Tensor:
        if self.df > 2:
            scale = self.df / (self.df - 2)
            return self.covariance_matrix * scale
        else:
            shape = self.covariance_matrix.shape
            return torch.full(shape, float("nan"), dtype=self.loc.dtype, device=self.loc.device)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        df = self.df
        d = self.event_shape[0]
        diff = value - self.loc
        covar = self.lazy_covariance_matrix

        # Compute quadratic form and log determinant
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        # Compute log probability
        lgamma_term = torch.lgamma(0.5 * (df + d)) - torch.lgamma(0.5 * df)
        log_scale = 0.5 * logdet + 0.5 * d * math.log(df * math.pi)
        log_1_plus = 0.5 * (df + d) * torch.log1p(inv_quad / df)

        log_prob = lgamma_term - log_scale - log_1_plus
        return log_prob.squeeze(-1)

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        df = self.df
        loc = self.loc
        covar = self.lazy_covariance_matrix

        if base_samples is None:
            # Generate standard normal samples
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(shape, dtype=loc.dtype, device=loc.device)

        # Get root decomposition
        covar_root = covar.root_decomposition().root

        # Reshape base_samples to match covar_root for matmul
        base_samples = base_samples.reshape(-1, *loc.shape[:-1], covar_root.shape[-1])
        base_samples = base_samples.permute(*range(1, loc.dim() + 1), 0)

        # Compute z = covar_root @ base_samples
        z = covar_root.matmul(base_samples)

        # Reshape z back to sample_shape
        z = z.permute(-1, *range(loc.dim())).contiguous()
        z = z.view(sample_shape + self.loc.shape)

        # Sample from Chi2 distribution
        chi2 = torch.distributions.Chi2(df)
        u = chi2.rsample(sample_shape + self.batch_shape)

        # Scale z
        sqrt_scale = torch.sqrt(u / df).unsqueeze(-1)
        res = loc + z / sqrt_scale

        return res

    def sample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def expand(self, batch_shape: torch.Size) -> MultivariateStudentT:
        new_loc = self.loc.expand(batch_shape + self.loc.shape[-1:])
        if self.islazy:
            new_covar = self._covariance_matrix.expand(batch_shape + self._covariance_matrix.shape[-2:])
            return self.__class__(new_loc, covariance_matrix=new_covar, df=self.df)
        else:
            new_scale_tril = self.scale_tril.expand(batch_shape + self.scale_tril.shape[-2:])
            return self.__class__(new_loc, scale_tril=new_scale_tril, df=self.df)

    def __getitem__(self, idx) -> MultivariateStudentT:
        new_loc = self.loc[idx]
        if self.islazy:
            new_covar = self.lazy_covariance_matrix[idx]
            return self.__class__(new_loc, covariance_matrix=new_covar, df=self.df)
        else:
            new_scale_tril = self.scale_tril[idx]
            return self.__class__(new_loc, scale_tril=new_scale_tril, df=self.df)

    def __add__(self, other: Union[MultivariateStudentT, Number]) -> MultivariateStudentT:
        if isinstance(other, MultivariateStudentT):
            if self.df != other.df:
                raise ValueError("Degrees of freedom must be the same for addition.")
            new_loc = self.loc + other.loc
            new_covar = self.lazy_covariance_matrix + other.lazy_covariance_matrix
            return self.__class__(new_loc, covariance_matrix=new_covar, df=self.df)
        elif isinstance(other, Number):
            return self.__class__(self.loc + other, covariance_matrix=self.lazy_covariance_matrix, df=self.df)
        else:
            raise RuntimeError(f"Unsupported type {type(other)} for addition with MultivariateStudentT.")

    def __radd__(self, other: Union[MultivariateStudentT, Number]) -> MultivariateStudentT:
        return self.__add__(other)

    def __mul__(self, other: Number) -> MultivariateStudentT:
        if not isinstance(other, Number):
            raise RuntimeError("Can only multiply by a scalar.")
        return self.__class__(
            self.loc * other,
            covariance_matrix=self.lazy_covariance_matrix * (other**2),
            df=self.df,
        )

    def __rmul__(self, other: Number) -> MultivariateStudentT:
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> MultivariateStudentT:
        return self.__mul__(1.0 / other)

    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return sample_shape + self._batch_shape + self._event_shape



# if __name__ == "__main__":
#     import torch
#     import plotly.graph_objs as go
#     import numpy as np

#     # Parameters for the first 1D t-distribution
#     loc1 = torch.tensor([0.0])  # Mean for the first distribution
#     cov1 = torch.tensor([[1.0]])  # Variance for the first distribution
#     df1 = 3.0  # Degrees of freedom for the first distribution

#     # Parameters for the second 1D t-distribution
#     loc2 = torch.tensor([0.0])  # Mean for the second distribution
#     cov2 = torch.tensor([[1.0]])  # Variance for the second distribution
#     df2 = 10.0  # Degrees of freedom for the second distribution

#     # Create the two 1D multivariate Student's t-distributions
#     multivariate_t1 = MultivariateStudentT(loc=loc1, covariance_matrix=cov1, df=df1)
#     multivariate_t2 = MultivariateStudentT(loc=loc2, covariance_matrix=cov2, df=df2)

#     # Generate samples for both distributions
#     num_samples = 10000
#     samples1 = multivariate_t1.rsample(sample_shape=torch.Size([num_samples])).squeeze(-1)
#     samples2 = multivariate_t2.rsample(sample_shape=torch.Size([num_samples])).squeeze(-1)

#     # Convert samples to NumPy arrays for Plotly visualization
#     x1 = samples1.numpy()
#     x2 = samples2.numpy()

#     # Create histograms for both distributions with density normalization
#     hist1 = go.Histogram(x=x1, nbinsx=100, opacity=0.6, name="df=3", marker_color='blue', histnorm='probability density')
#     hist2 = go.Histogram(x=x2, nbinsx=100, opacity=0.6, name="df=10", marker_color='red', histnorm='probability density')

#     # Layout for the plot
#     layout = go.Layout(
#         title="1D Visualization of Two Student's t-distributions with Different Degrees of Freedom",
#         xaxis_title="Sample Value",
#         yaxis_title="Density",  # Set y-axis to show density
#         barmode="overlay"
#     )

#     # Create the figure with both histograms
#     fig = go.Figure(data=[hist1, hist2], layout=layout)

#     # Show the plot
#     fig.show()




# if __name__ == "__main__":
#     import torch
#     import plotly.graph_objs as go
#     import numpy as np

#     # Parameters for the first distribution
#     loc1 = torch.tensor([0.0, 0.0])  # Mean vector for the first 2D distribution
#     # scale_tril1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Cholesky decomposition for the first distribution
#     cov1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  
#     df1 = 3.0  # Degrees of freedom for the first distribution

#     # Parameters for the second distribution
#     loc2 = torch.tensor([0.0, 0.0])  # Mean vector for the second 2D distribution
#     cov2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  
#     df2 = 10.0  # Degrees of freedom for the second distribution

#     # Create the two multivariate Student's t-distributions
#     # multivariate_t1 = MultivariateStudentT(loc=loc1, scale_tril=scale_tril1, df=df1)
#     # multivariate_t2 = MultivariateStudentT(loc=loc2, scale_tril=scale_tril2, df=df2)
#     multivariate_t1 = MultivariateStudentT(loc=loc1, covariance_matrix=cov1, df=df1)
#     multivariate_t2 = MultivariateStudentT(loc=loc2, covariance_matrix=cov2, df=df2)

#     # Generate samples for both distributions
#     num_samples = 10000
#     samples1 = multivariate_t1.rsample(sample_shape=torch.Size([num_samples]))
#     samples2 = multivariate_t2.rsample(sample_shape=torch.Size([num_samples]))

#     # Convert samples to NumPy arrays for Plotly visualization
#     x1 = samples1[:, 0].numpy()
#     y1 = samples1[:, 1].numpy()
#     x2 = samples2[:, 0].numpy()
#     y2 = samples2[:, 1].numpy()

#     # Create 2D histograms (binning) for both distributions
#     hist1, x_edges1, y_edges1 = np.histogram2d(x1, y1, bins=100, density=True)
#     hist2, x_edges2, y_edges2 = np.histogram2d(x2, y2, bins=100, density=True)

#     # Apply log scaling to the histograms (logarithmic color scale)
#     hist1 = np.log1p(hist1)  # log1p to avoid log(0) issues
#     hist2 = np.log1p(hist2)

#     # Create coordinates for the centers of the bins
#     x_mid1 = (x_edges1[:-1] + x_edges1[1:]) / 2
#     y_mid1 = (y_edges1[:-1] + y_edges1[1:]) / 2
#     x_mid2 = (x_edges2[:-1] + x_edges2[1:]) / 2
#     y_mid2 = (y_edges2[:-1] + y_edges2[1:]) / 2
#     X1, Y1 = np.meshgrid(x_mid1, y_mid1)
#     X2, Y2 = np.meshgrid(x_mid2, y_mid2)

#     # Create surface plots for both distributions with solid colors
#     surface1 = go.Surface(
#         x=X1, 
#         y=Y1, 
#         z=hist1.T, 
#         surfacecolor=np.ones_like(hist1.T),  # Uniform color for the first surface
#         opacity=0.6,  # Set transparency level
#         showscale=False,  # Disable color scale
#         colorscale=[[0, 'blue'], [1, 'blue']],  # Solid blue color
#     )

#     surface2 = go.Surface(
#         x=X2, 
#         y=Y2, 
#         z=hist2.T, 
#         surfacecolor=np.ones_like(hist2.T),  # Uniform color for the second surface
#         opacity=0.6,  # Set transparency level
#         showscale=False,  # Disable color scale
#         colorscale=[[0, 'red'], [1, 'red']],  # Solid red color
#     )

#     # Layout for the plot
#     layout = go.Layout(
#         title="3D Visualization of Two Bivariate Student's t-distributions with Different Degrees of Freedom",
#         scene=dict(
#             xaxis_title='Dimension 1',
#             yaxis_title='Dimension 2',
#             zaxis_title='Log Density',
#         ),
#     )

#     # Create the figure with both surfaces
#     fig = go.Figure(data=[surface1, surface2], layout=layout)

#     # Show the plot
#     fig.show()
