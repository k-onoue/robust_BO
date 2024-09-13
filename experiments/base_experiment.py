import configparser
import logging
import sys
import time
import warnings
from copy import deepcopy

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# 設定の読み込み
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.utils_bo import (InputTransformer, evaluate_candidate,
                          fit_pytorch_model, generate_integer_samples,
                          initialize_model, log_initial_data, negate_function)
from src.utils_experiment import set_logger


def run_bo(setting_dict: dict):
    try:
        logging.info(f"Running optimization with settings: \n{setting_dict}")
        device = setting_dict["device"]
        logging.info(f"Running on device: {device}")
        start_time = time.time()

        ######################################################################
        # Step 1: Define the Objective Function and Search Space
        def objective_function(X):
            return (X**2).sum(dim=-1)

        objective_function = negate_function(objective_function)

        search_space = torch.tensor([[-10] * 5, [10] * 5]).to(torch.float32).to(device)

        trans = InputTransformer(search_space, lower_bound=0, upper_bound=1)

        ######################################################################
        # Step 2: Generate Initial Data
        initial_data_size = setting_dict["initial_data_size"]
        X_train = (
            generate_integer_samples(search_space, initial_data_size).to(device).float()
        )
        y_train = objective_function(X_train)
        log_initial_data(X_train, y_train, initial_data_size)

        # Flatten X_train and move it to the correct device
        X_train_normalized = trans.normalize(X_train)
        y_train = y_train.to(device)

        ######################################################################
        # Step 3: Train the Bayesian MLP Model
        model = initialize_model(setting_dict, X_train_normalized, y_train)
        acq_optim_settings = setting_dict["acquisition_optim"]
        beta = deepcopy(acq_optim_settings["beta"])

        ucb = UpperConfidenceBound(model, beta=beta)
        model_optim_settings = setting_dict["model_optim"]

        final_loss = fit_pytorch_model(
            model,
            num_epochs=model_optim_settings["num_epochs"],
            learning_rate=model_optim_settings["learning_rate"],
        )

        logging.info(f"Final training loss: {final_loss:.6f}")

        ######################################################################
        # Step 4: Optimization Iterations
        n_iterations = setting_dict["bo_iter"]
        best_value = float("-inf")

        for iteration in range(n_iterations):
            iter_start_time = time.time()

            logging.info(f"Iteration {iteration + 1}/{n_iterations}")

            # Optimize Acquisition Function
            candidate, acq_value = optimize_acqf(
                acq_function=ucb,
                bounds=search_space,
                q=1,
                num_restarts=acq_optim_settings["num_restarts"],
                raw_samples=acq_optim_settings["raw_samples"],
            )

            # Evaluate Candidate
            candidate, y_new = evaluate_candidate(
                model, trans, ucb, candidate, objective_function, device
            )

            X_train = torch.cat([X_train.to(device), candidate.unsqueeze(0).to(device)])
            y_train = torch.cat([y_train.to(device), y_new.unsqueeze(-1).to(device)])

            # Update and refit the Bayesian MLP model
            X_train_normalized = trans.normalize(X_train)

            model.set_train_data(X_train_normalized, y_train)
            final_loss = fit_pytorch_model(
                model,
                num_epochs=model_optim_settings["num_epochs"],
                learning_rate=model_optim_settings["learning_rate"],
            )

            logging.info(f"Final training loss: {final_loss:.6f}")

            if y_new.item() > best_value:
                best_value = y_new.item()
                logging.info(f"New best value found: {best_value}")

            elapsed_time = time.time() - iter_start_time
            logging.info(f"Iteration time: {elapsed_time:.4f} seconds")

    except Exception as e:
        logging.exception(e)
        raise e

    # Final results
    logging.info("Optimization completed.")
    optim_idx = y_train.argmax()
    logging.info(f"Optimal solution: {X_train[optim_idx]}")
    logging.info(f"Function value: {y_train[optim_idx].item()}")

    elapsed_time = time.time() - start_time
    logging.info(f"Total time on {device}: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "bo_iter": 1000,
        "initial_data_size": 1,
        "model": {
            "hidden_unit_size": 64,
            "num_hidden_layers": 3,
            "activation_fn": torch.nn.ReLU(),
            "min_val": None,
            "max_val": None,
        },
        "model_optim": {
            "num_epochs": 100,
            "learning_rate": 0.001,
        },
        "acquisition_optim": {
            "beta": 0.1,
            "num_restarts": 5,
            "raw_samples": 20,
        },
        "memo": "",
    }

    set_logger(settings["name"], LOG_DIR)
    run_bo(settings)