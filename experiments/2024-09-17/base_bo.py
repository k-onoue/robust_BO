import logging

import torch
from botorch.utils.transforms import unnormalize

from src.utils_bo import apply_transformation, generate_initial_data
from src.utils_experiment import set_random_seed


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
            model, trained_params = train_surrogate_fn(
                train_x_normalized, train_y_standardized, trained_params
            )

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
