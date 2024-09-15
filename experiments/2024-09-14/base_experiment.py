import configparser
import sys
import warnings

import torch

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.test_functions import branin_hoo, sinusoidal_synthetic
from src.utils_experiment import set_logger

from base_bo import run_bo
from base_components import train_gp, optimize_acquisition_function




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
