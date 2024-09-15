import logging

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


# Train the Gaussian Process model with a Matern kernel (nu=2.5), ARD, and explicit kernel parameters
def train_gp(train_x, train_y, trained_params=None):
    # Determine the number of input dimensions for ARD
    ard_num_dims = train_x.shape[-1]  # Set ARD dimensions based on the input data

    # Check if we have trained parameters to initialize the kernel
    if trained_params is None:
        # First iteration: initialize with priors
        lengthscale_prior = GammaPrior(3.0, 6.0)  # Default prior for lengthscale
        outputscale_prior = GammaPrior(2.0, 0.15)  # Default prior for outputscale
        noise_prior = None  # Use default noise model
    else:
        # Subsequent iterations: use the learned parameters from the previous model
        lengthscale_prior = None  # No need to define a prior, use the learned value
        outputscale_prior = None  # No need to define a prior, use the learned value

    # Define the Matern Kernel with nu=2.5 (smoothness) and ARD enabled
    matern_kernel = MaternKernel(
        nu=2.5, ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior
    )

    # If we have trained parameters, set the learned lengthscale
    if trained_params is not None:
        matern_kernel.lengthscale = torch.tensor(trained_params["lengthscale"])

    # ScaleKernel allows for automatic scaling of the output
    kernel = ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)

    # If we have trained parameters, set the learned outputscale
    if trained_params is not None:
        kernel.outputscale = torch.tensor(trained_params["outputscale"])

    # Define the GP model with the specified kernel
    new_model = SingleTaskGP(train_x, train_y, covar_module=kernel)

    # Set the learned noise if trained parameters exist
    if trained_params is not None:
        new_model.likelihood.noise = torch.tensor(trained_params["noise"])

    # Marginal log likelihood for GP model training
    mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model)

    # Fit the GP model
    fit_gpytorch_mll(mll)

    # Collect trained parameters in a dictionary for logging
    trained_params = {
        "lengthscale": new_model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .numpy(),
        "outputscale": new_model.covar_module.outputscale.detach().cpu().numpy(),
        "noise": new_model.likelihood.noise.detach().cpu().numpy(),
    }

    # Log the trained parameters
    logging.info(f"Trained lengthscale: {trained_params['lengthscale']}")
    logging.info(f"Trained outputscale: {trained_params['outputscale']}")
    logging.info(f"Trained noise: {trained_params['noise']}")

    return new_model, trained_params


# Define and optimize the acquisition function, returning both the candidate and acquisition value
def optimize_acquisition_function(
    model,
    bounds,
    beta=0.1,
    num_restarts=5,
    raw_samples=20,
    device="cpu",
    dtype=torch.float,
):
    # Set up the acquisition function based on the given beta
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
    )

    return candidate_normalized, acq_value
