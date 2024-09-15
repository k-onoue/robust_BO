import logging
from dataclasses import dataclass, field
from typing import Optional

import gpytorch
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from tptorch.distributions import MultivariateStudentT
from tptorch.likelihoods import StudentTLikelihood
from tptorch.mlls.exact_student_t_marginal_log_likelihood import (
    ExactStudentTMarginalLogLikelihood,
)
from tptorch.models.exact_tp import ExactTP

# ======================
# Model Configuration Data Classes
# ======================


@dataclass
class GPModelConfig:
    ard_num_dims: int
    lengthscale_prior: gpytorch.priors.Prior = field(
        default_factory=lambda: GammaPrior(3.0, 6.0)
    )
    outputscale_prior: gpytorch.priors.Prior = field(
        default_factory=lambda: GammaPrior(2.0, 0.15)
    )
    noise_prior: gpytorch.priors.Prior = field(
        default_factory=lambda: GammaPrior(1.1, 0.05)
    )
    lengthscale_constraint: gpytorch.constraints.Interval = field(
        default_factory=Positive
    )
    outputscale_constraint: gpytorch.constraints.Interval = field(
        default_factory=Positive
    )
    noise_constraint: gpytorch.constraints.Interval = field(
        default_factory=lambda: GreaterThan(1e-4)
    )
    trained_params: Optional[dict] = None
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float


@dataclass
class TPModelConfig:
    ard_num_dims: int
    nu: float = 5.0  # Degrees of freedom for the Student's T distribution
    lengthscale_prior: gpytorch.priors.Prior = field(
        default_factory=lambda: GammaPrior(3.0, 6.0)
    )
    outputscale_prior: gpytorch.priors.Prior = field(
        default_factory=lambda: GammaPrior(2.0, 0.15)
    )
    lengthscale_constraint: gpytorch.constraints.Interval = field(
        default_factory=Positive
    )
    outputscale_constraint: gpytorch.constraints.Interval = field(
        default_factory=Positive
    )
    trained_params: Optional[dict] = None
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float


# ======================
# Model Definition Module
# ======================


def create_kernel(config):
    """
    Creates the kernel for the GP or TP model.
    """
    # Create Matern kernel with nu=2.5
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=config.ard_num_dims,
        lengthscale_prior=config.lengthscale_prior,
        lengthscale_constraint=config.lengthscale_constraint,
    )

    # Set trained lengthscale if available
    if config.trained_params and "lengthscale" in config.trained_params:
        matern_kernel.lengthscale = torch.tensor(
            config.trained_params["lengthscale"]
        ).to(device=config.device, dtype=config.dtype)

    # Scale kernel
    kernel = ScaleKernel(
        base_kernel=matern_kernel,
        outputscale_prior=config.outputscale_prior,
        outputscale_constraint=config.outputscale_constraint,
    )

    # Set trained outputscale if available
    if config.trained_params and "outputscale" in config.trained_params:
        kernel.outputscale = torch.tensor(config.trained_params["outputscale"]).to(
            device=config.device, dtype=config.dtype
        )

    return kernel


def create_gaussian_likelihood(config: GPModelConfig):
    """
    Creates the Gaussian likelihood for the GP model.
    """
    likelihood = GaussianLikelihood(
        noise_prior=config.noise_prior,
        noise_constraint=config.noise_constraint,
    )
    if config.trained_params and "noise" in config.trained_params:
        likelihood.noise = torch.tensor(config.trained_params["noise"]).to(
            device=config.device, dtype=config.dtype
        )
    return likelihood


def create_student_t_likelihood(config: TPModelConfig):
    """
    Creates the Student's T likelihood for the TP model.
    """
    likelihood = StudentTLikelihood()
    if config.trained_params and "nu" in config.trained_params:
        likelihood.nu = torch.tensor(config.trained_params["nu"]).to(
            device=config.device, dtype=config.dtype
        )
    return likelihood


def create_gp_model(train_x, train_y, config: GPModelConfig):
    """
    Creates the GP model.
    """
    # Create kernel and likelihood using the config
    kernel = create_kernel(config)
    likelihood = create_gaussian_likelihood(config)

    # Create GP model with the specified kernel and likelihood
    model = SingleTaskGP(train_x, train_y, covar_module=kernel, likelihood=likelihood)

    return model, likelihood


# ======================
# CustomTPModel Definition
# ======================


class TPModel(ExactTP):
    def __init__(self, train_x, train_y, likelihood, config: TPModelConfig):
        super(TPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        nu = torch.tensor(config.nu, dtype=config.dtype, device=config.device)
        self.nu = torch.nn.Parameter(nu)
        self.data_num = torch.tensor(
            train_y.shape[0], dtype=config.dtype, device=config.device
        )

        # Define the kernel with ARD
        self.covar_module = create_kernel(config)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x_train_data = self.covar_module(self.train_inputs[0])
        inv_quad, _ = covar_x_train_data.inv_quad_logdet(
            inv_quad_rhs=self.train_targets - self.train_targets.mean(), logdet=False
        )

        tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)
        covar_x = tp_var_scale.float() * covar_x

        return MultivariateStudentT(mean_x, covar_x, self.nu, len(x))


def create_tp_model(train_x, train_y, config: TPModelConfig):
    """
    Creates the TP model.
    """
    # Create likelihood using the config
    likelihood = create_student_t_likelihood(config)

    # Create the CustomTPModel
    model = TPModel(train_x, train_y, likelihood, config)

    return model, likelihood


# ======================
# Training Functions
# ======================

def train_gp(train_x_normalized, train_y_standardized, trained_params=None, device=torch.device('cpu'), dtype=torch.float):
    """
    Trains the GP surrogate model.
    """
    # Ensure train_y_standardized is two-dimensional
    if train_y_standardized.dim() == 1:
        train_y_standardized = train_y_standardized.unsqueeze(-1)

    # Create a configuration instance
    config = GPModelConfig(
        ard_num_dims=train_x_normalized.shape[-1],
        trained_params=trained_params,
        device=device,
        dtype=dtype,
    )

    # Create and train the model
    model, likelihood = create_gp_model(train_x_normalized, train_y_standardized, config)
    trained_params = train_model_gp(
        model, likelihood, train_x_normalized, train_y_standardized,
        training_iterations=50, learning_rate=0.1
    )

    # Update the config with the trained parameters
    config.trained_params = trained_params

    # Return the trained model and updated trained parameters
    return model, trained_params

def train_tp(train_x_normalized, train_y_standardized, trained_params=None, device=torch.device('cpu'), dtype=torch.float):
    """
    Trains the TP surrogate model.
    """
    # Ensure train_y_standardized is two-dimensional
    if train_y_standardized.dim() == 1:
        train_y_standardized = train_y_standardized.unsqueeze(-1)

    # Create a configuration instance
    config = TPModelConfig(
        ard_num_dims=train_x_normalized.shape[-1],
        trained_params=trained_params,
        device=device,
        dtype=dtype,
    )

    # Create and train the model
    model, likelihood = create_tp_model(train_x_normalized, train_y_standardized, config)
    trained_params = train_model_tp(
        model, likelihood, train_x_normalized, train_y_standardized,
        training_iterations=50, learning_rate=0.1
    )

    # Update the config with the trained parameters
    config.trained_params = trained_params

    # Return the trained model and updated trained parameters
    return model, trained_params

# ======================
# Training Modules
# ======================

def train_model_gp(model, likelihood, train_x, train_y, training_iterations=50, learning_rate=0.1):
    """
    Trains the GP model using a custom training loop.
    """
    # Ensure train_y is two-dimensional
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    # Set model and likelihood in training mode
    model.train()
    likelihood.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=learning_rate
    )

    # Define the loss function (marginal log likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss = loss.sum()  # Ensure loss is scalar
        if (i + 1) % 10 == 0 or i == 0:
            loss_value = loss.item()
            logging.info(f'Iter {i + 1}/{training_iterations} - Loss: {loss_value:.3f}')
        loss.backward()
        optimizer.step()

    # Collect trained parameters in a dictionary
    trained_params = {
        "lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
        "outputscale": model.covar_module.outputscale.detach().cpu().numpy(),
        "noise": model.likelihood.noise.detach().cpu().numpy(),
    }

    # Log the trained parameters
    logging.info(f"Trained lengthscale: {trained_params['lengthscale']}")
    logging.info(f"Trained outputscale: {trained_params['outputscale']}")
    logging.info(f"Trained noise: {trained_params['noise']}")

    return trained_params

def train_model_tp(model, likelihood, train_x, train_y, training_iterations=50, learning_rate=0.1):
    """
    Trains the TP model using a custom training loop.
    """
    # Ensure train_y is two-dimensional
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    # Set model and likelihood in training mode
    model.train()
    likelihood.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=learning_rate
    )

    # Define the loss function (marginal log likelihood)
    mll = ExactStudentTMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss = loss.sum()  # Ensure loss is scalar
        if (i + 1) % 10 == 0 or i == 0:
            loss_value = loss.item()
            logging.info(f'Iter {i + 1}/{training_iterations} - Loss: {loss_value:.3f}')
        loss.backward()
        optimizer.step()

        # Clamp nu to be greater than 2
        with torch.no_grad():
            model.nu.clamp_(2.1)

    # Collect trained parameters in a dictionary
    trained_params = {
        "lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
        "outputscale": model.covar_module.outputscale.detach().cpu().numpy(),
        "nu": model.nu.detach().cpu().numpy(),
    }

    # Log the trained parameters
    logging.info(f"Trained lengthscale: {trained_params['lengthscale']}")
    logging.info(f"Trained outputscale: {trained_params['outputscale']}")
    logging.info(f"Trained nu: {trained_params['nu']}")

    return trained_params



# ======================
# Acquisition Function Optimization (optimize_acquisition_function)
# ======================


def optimize_acquisition_function(
    model, bounds, device, dtype, beta=0.1, num_restarts=5, raw_samples=20
):
    """
    Defines and optimizes the acquisition function.
    """
    # Set up the acquisition function
    acquisition_func = UpperConfidenceBound(model, beta=beta)

    acquisition_params = {
        "func": acquisition_func.__class__.__name__,
        "beta": beta,
    }

    logging.info(f"Acquisition Function: {acquisition_params['func']}")
    logging.info(f"Beta: {acquisition_params['beta']}")

    # Optimize the acquisition function
    candidate_normalized, acq_value = optimize_acqf(
        acquisition_func,
        bounds=torch.tensor(
            [[0.0] * bounds.size(1), [1.0] * bounds.size(1)], device=device, dtype=dtype
        ),
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )

    return candidate_normalized, acq_value
