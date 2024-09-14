import configparser
import logging
import sys
import warnings

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.test_functions import branin_hoo, sinusoidal_synthetic
from src.utils_experiment import set_logger, set_random_seed


# Generate initial data based on Sobol samples
def generate_initial_data(objective, bounds, n=5, device="cpu", dtype=torch.float):
    initial_x = (
        draw_sobol_samples(bounds=bounds, n=n, q=1)
        .squeeze(1)
        .to(device=device, dtype=dtype)
    )
    initial_y = objective(initial_x).to(device)
    return initial_x, initial_y


# Train the Gaussian Process model with a Matern kernel (nu=2.5), ARD, and explicit kernel parameters
def train_gp(train_x, train_y, trained_params=None):
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

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
        nu=2.5,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=lengthscale_prior
    )

    # If we have trained parameters, set the learned lengthscale
    if trained_params is not None:
        matern_kernel.lengthscale = torch.tensor(trained_params['lengthscale'])

    # ScaleKernel allows for automatic scaling of the output
    kernel = ScaleKernel(
        matern_kernel,
        outputscale_prior=outputscale_prior
    )

    # If we have trained parameters, set the learned outputscale
    if trained_params is not None:
        kernel.outputscale = torch.tensor(trained_params['outputscale'])

    # Define the GP model with the specified kernel
    new_model = SingleTaskGP(train_x, train_y, covar_module=kernel)

    # Set the learned noise if trained parameters exist
    if trained_params is not None:
        new_model.likelihood.noise = torch.tensor(trained_params['noise'])

    # Marginal log likelihood for GP model training
    mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model)

    # Fit the GP model
    fit_gpytorch_mll(mll)

    # Collect trained parameters in a dictionary for logging
    trained_params = {
        "lengthscale": new_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
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
    model, bounds, beta=0.1, num_restarts=5, raw_samples=20, device="cpu", dtype=torch.float
):
    # Set up the acquisition function based on the given beta
    acquisition_func = UpperConfidenceBound(
        model, beta=beta
    )

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


# Perform normalization or standardization based on settings
def apply_transformation(train_x, train_y, bounds, settings):
    if settings.get("normalize", False):
        train_x_normalized = normalize(train_x, bounds)
    else:
        train_x_normalized = train_x  # No normalization

    if settings.get("standardize", False):
        train_y_standardized = standardize(train_y)
    else:
        train_y_standardized = train_y  # No standardization

    return train_x_normalized, train_y_standardized



# Main Bayesian Optimization function
def run_bo(settings: dict, train_surrogate_fn, optimize_acq_fn):
    logging.info(f"Running Bayesian Optimization with settings: {settings}")

    # Set the seed for reproducibility
    seed = settings.get("seed", 0)
    set_random_seed(seed)

    # Unpacking settings
    device = settings["device"]
    dtype = settings["dtype"]
    objective = settings["objective"]
    search_space = settings["search_space"]
    initial_data_size = settings["initial_data_size"]
    budget = settings["budget"]

    # Define bounds based on the search space
    bounds = torch.tensor(search_space).to(dtype=dtype, device=device)

    # Initialize trained parameters as None initially
    trained_params = None

    # Generate initial data
    train_x, train_y = generate_initial_data(
        objective, bounds, n=initial_data_size, device=device, dtype=dtype
    )

    # Apply normalization and standardization based on settings
    train_x_normalized, train_y_standardized = apply_transformation(
        train_x, train_y, bounds, settings
    )

    try:
        for iteration in range(budget):
            # Log the iteration number once at the start
            logging.info(f"Iteration {iteration + 1}/{budget}")

            # Train GP model and get the trained parameters
            model, trained_params = train_surrogate_fn(train_x_normalized, train_y_standardized, trained_params)

            # Optimize acquisition function and get both candidate and acquisition value
            candidate_normalized, acq_value = optimize_acq_fn(
                model, bounds, device=device, dtype=dtype
            )

            # Unnormalize the candidate
            candidate = unnormalize(candidate_normalized, bounds)

            # Evaluate objective at new candidate point
            new_x = candidate.detach().to(device)
            new_y = objective(new_x).to(device)

            # Log candidate, acquisition value, and objective evaluation
            logging.info(f"New Candidate X: {new_x.cpu().numpy()}")
            logging.info(f"Acquisition Value: {acq_value.item()}")
            logging.info(f"Objective Evaluation Y: {new_y.item()}")

            # Update training data
            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)

            # Apply transformation to updated data
            train_x_normalized, train_y_standardized = apply_transformation(
                train_x, train_y, bounds, settings
            )

            # Find current best x and y
            best_y, best_idx = train_y.max(dim=0)
            best_x = train_x[best_idx]

            # Log the current best values
            logging.info(f"Current Best X: {best_x.cpu().numpy()}")
            logging.info(f"Current Best Y: {best_y.item()}")

    except Exception as e:
        logging.exception(f"An error occurred during Bayesian Optimization: {e}")
        raise e

    return train_x, train_y



# Example usage with Branin-Hoo objective
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "dtype": torch.double,
        "seed": 42,  # Set a seed for reproducibility
        "initial_data_size": 5,
        "budget": 10,
        "objective": branin_hoo,  # Use the provided branin_hoo function
        "search_space": [
            [0.0, 0.0],  # x1 bounds
            [1.0, 1.0],  # x2 bounds
        ],
        "normalize": True,  # Enable normalization
        "standardize": False,  # Disable standardization (can change to True if needed)
        "memo": "",
    }

    set_logger(settings["name"], LOG_DIR)

    # Call run_bo with the specific train_gp and optimize_acquisition_function implementations
    run_bo(settings, train_gp, optimize_acquisition_function)
