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

from src.test_functions import branin_hoo, sinusoidal_synthetic, hartmann6
from src.utils_experiment import set_logger

from base_bo import run_bo
from base_components import train_gp, train_tp, optimize_acquisition_function  # Import the functions we defined

# Example usage with Hartmann6 objective
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    # settings = {
    #     "name": name,
    #     "device": device,
    #     "dtype": torch.double,
    #     "seed": 42,  # Set a seed for reproducibility
    #     "initial_data_size": 5,
    #     "budget": 10,
    #     "objective": hartmann6,  # Use the provided Hartmann6 function
    #     "search_space": [
    #         [0] * 6,  # Lower bounds for each dimension
    #         [1] * 6,  # Upper bounds for each dimension
    #     ],
    #     "normalize": True,  # Enable normalization
    #     "standardize": False,  # Disable standardization (can change to True if needed)
    #     "memo": "",
    # }
    settings = {
        "name": name,
        "device": device,
        "dtype": torch.double,
        "seed": 42,  # Set a seed for reproducibility
        "initial_data_size": 5,
        "budget": 10,
        "objective": branin_hoo,  # Use the provided Hartmann6 function
        "search_space": [
            [-10] * 2,  # Lower bounds for each dimension
            [10] * 2,  # Upper bounds for each dimension
        ],
        "normalize": True,  # Enable normalization
        "standardize": False,  # Disable standardization (can change to True if needed)
        "memo": "",
    }

    set_logger(settings["name"], LOG_DIR)

    # Call run_bo with the specific train_gp and optimize_acquisition_function implementations
    run_bo(settings, train_tp, optimize_acquisition_function)
