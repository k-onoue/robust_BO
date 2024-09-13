import datetime
import logging
import os
# import re
import sys

# import pandas as pd
# import torch


def set_logger(log_filename_base, save_dir):
    # ログの設定
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}_{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )


def search_log_files(
    log_dir: str, keywords: list[str], logic: str = "and"
) -> list[str]:
    if logic not in ["or", "and"]:
        raise ValueError("The logic parameter must be 'or' or 'and'.")

    res_files = sorted(os.listdir(log_dir))

    if logic == "and":
        res_files_filtered = [
            f for f in res_files if all(keyword in f for keyword in keywords)
        ]
    elif logic == "or":
        res_files_filtered = [
            f for f in res_files if any(keyword in f for keyword in keywords)
        ]

    return res_files_filtered


# class OptimLogParser_v1:
#     def __init__(self, log_file):
#         self.log_file = log_file
#         self.settings = {}
#         self.initial_data = {
#             "candidate": [],
#             "function_value": [],
#             "final_training_loss": [],
#         }
#         self.bo_data = {
#             "iteration": [],
#             "candidate": [],
#             "acquisition_value": [],
#             "function_value": [],
#             "final_training_loss": [],
#             "iteration_time": [],
#         }

#     def combine_log_entries(self):
#         with open(self.log_file, "r") as file:
#             lines = file.readlines()

#         timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "

#         combined_lines = []
#         current_entry = ""

#         for line in lines:
#             if re.match(timestamp_pattern, line):
#                 if current_entry:
#                     combined_lines.append(current_entry.strip())
#                 current_entry = line.strip()
#             else:
#                 current_entry += " " + line.strip()

#         if current_entry:
#             combined_lines.append(current_entry.strip())

#         return combined_lines

#     def parse_log_file(self):
#         combined_lines = self.combine_log_entries()

#         mode = None

#         for line in combined_lines:
#             if "Running optimization with settings:" in line:
#                 mode = "settings"
#                 self._parse_settings(line)
#             elif "Initial data points" in line:
#                 mode = "init"
#             elif "Iteration" in line:
#                 mode = "bo_loop"
#             elif "Optimization completed." in line:
#                 break  

#             if mode == "init":
#                 self._parse_init_data(line)
#             elif mode == "bo_loop":
#                 self._parse_bo_data(line)

#         # Fill in the final training loss for the initial data
#         val = self.initial_data["final_training_loss"][-1]
#         length = len(self.initial_data["candidate"])
#         self.initial_data["final_training_loss"] = [val] * length

#         final_iter = self.bo_data["iteration"][-1] if self.bo_data["iteration"] else 0
#         for column in self.bo_data:
#             while len(self.bo_data[column]) < final_iter:
#                 self.bo_data[column].append(None)

#         self.initial_data = pd.DataFrame(self.initial_data)
#         self.bo_data = pd.DataFrame(self.bo_data)

#     def _parse_settings(self, line):
#         settings_str = line.split("settings:")[1].strip()
#         settings_str = re.sub(r"device\(type='[^']+'\)", "'cpu'", settings_str)
#         settings_str = re.sub(r"device\(type=\"[^\"]+\"\)", "'cpu'", settings_str)
#         try:
#             self.settings = eval(settings_str)
#         except SyntaxError as e:
#             print(f"Failed to parse settings: {e}")
#             self.settings = {}

#     def _parse_init_data(self, line):
#         candidate_match = re.search(r"Candidate: (.*?) Function Value:", line)
#         function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
#         final_training_loss_match = re.search(
#             r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
#         )

#         if candidate_match:
#             self.initial_data["candidate"].append(candidate_match.group(1).strip())
#         if function_value_match:
#             self.initial_data["function_value"].append(
#                 float(function_value_match.group(1))
#             )
#         if final_training_loss_match:
#             self.initial_data["final_training_loss"].append(
#                 float(final_training_loss_match.group(1))
#             )

#     def _parse_bo_data(self, line):
#         iteration_match = re.search(r"Iteration (\d+)/", line)
#         candidate_match = re.search(r"Candidate: (\[.*?\])", line)
#         acquisition_value_match = re.search(
#             r"Acquisition Value: ([-+]?\d*\.\d+|\d+)", line
#         )
#         function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
#         final_training_loss_match = re.search(
#             r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
#         )
#         iteration_time_match = re.search(r"Iteration time: ([-+]?\d*\.\d+)", line)

#         if iteration_match:
#             self.bo_data["iteration"].append(int(iteration_match.group(1)))
#         if candidate_match:
#             self.bo_data["candidate"].append(candidate_match.group(1).strip())
#         if acquisition_value_match:
#             self.bo_data["acquisition_value"].append(
#                 float(acquisition_value_match.group(1))
#             )
#         if function_value_match:
#             self.bo_data["function_value"].append(float(function_value_match.group(1)))
#         if final_training_loss_match:
#             self.bo_data["final_training_loss"].append(
#                 float(final_training_loss_match.group(1))
#             )
#         if iteration_time_match:
#             self.bo_data["iteration_time"].append(float(iteration_time_match.group(1)))


# class OptimLogParser:
#     def __init__(self, log_file):
#         self.log_file = log_file
#         self.settings = {}
#         self.initial_data = {
#             "candidate": [],
#             "function_value": [],
#             "final_training_loss": [],
#         }
#         self.bo_data = {
#             "iteration": [],
#             "candidate": [],
#             "acquisition_value": [],
#             "beta": [],
#             "function_value": [],
#             "final_training_loss": [],
#             "iteration_time": [],
#         }

#     def combine_log_entries(self):
#         with open(self.log_file, "r") as file:
#             lines = file.readlines()

#         timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "

#         combined_lines = []
#         current_entry = ""

#         for line in lines:
#             if re.match(timestamp_pattern, line):
#                 if current_entry:
#                     combined_lines.append(current_entry.strip())
#                 current_entry = line.strip()
#             else:
#                 current_entry += " " + line.strip()

#         if current_entry:
#             combined_lines.append(current_entry.strip())

#         return combined_lines

#     def parse_log_file(self):
#         combined_lines = self.combine_log_entries()

#         mode = None

#         for line in combined_lines:
#             if "Running optimization with settings:" in line:
#                 mode = "settings"
#                 self._parse_settings(line)
#             elif "Initial data points" in line:
#                 mode = "init"
#             elif "Iteration" in line:
#                 mode = "bo_loop"
#             elif "Optimization completed." in line:
#                 break  # 終了条件

#             if mode == "init":
#                 self._parse_init_data(line)
#             elif mode == "bo_loop":
#                 self._parse_bo_data(line)

#         # Fill in the final training loss for the initial data
#         val = self.initial_data["final_training_loss"][-1]
#         length = len(self.initial_data["candidate"])
#         self.initial_data["final_training_loss"] = [val] * length

#         final_iter = self.bo_data["iteration"][-1] if self.bo_data["iteration"] else 0
#         for column in self.bo_data:
#             while len(self.bo_data[column]) < final_iter:
#                 self.bo_data[column].append(None)

#         self.initial_data = pd.DataFrame(self.initial_data)
#         self.bo_data = pd.DataFrame(self.bo_data)

#     def _parse_settings(self, line):
#         from torch.nn import Tanh, ReLU, LeakyReLU
#         settings_str = line.split("settings:")[1].strip()
#         settings_str = re.sub(r"device\(type='[^']+'\)", "'cpu'", settings_str)
#         settings_str = re.sub(r"device\(type=\"[^\"]+\"\)", "'cpu'", settings_str)
#         try:
#             self.settings = eval(settings_str)
#         except SyntaxError as e:
#             print(f"Failed to parse settings: {e}")
#             self.settings = {}

#     def _parse_init_data(self, line):
#         candidate_match = re.search(r"Candidate: (\[.*?\])", line)
#         function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
#         final_training_loss_match = re.search(
#             r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
#         )

#         if candidate_match:
#             candidate = candidate_match.group(1).strip()
#             self.initial_data["candidate"].append(candidate)

#         if function_value_match:
#             function_value = float(function_value_match.group(1))
#             # Ensure we have a candidate entry before appending the function value
#             if len(self.initial_data["candidate"]) > len(self.initial_data["function_value"]):
#                 self.initial_data["function_value"].append(function_value)

#         if final_training_loss_match:
#             self.initial_data["final_training_loss"].append(
#                 float(final_training_loss_match.group(1))
#             )

#     def _parse_bo_data(self, line):
#         iteration_match = re.search(r"Iteration (\d+)/", line)
#         candidate_match = re.search(r"Candidate: (\[.*?\])", line)
#         acquisition_value_match = re.search(
#             r"Acquisition Value: ([-+]?\d*\.\d+|\d+)", line
#         )
#         function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
#         final_training_loss_match = re.search(
#             r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
#         )
#         iteration_time_match = re.search(r"Iteration time: ([-+]?\d*\.\d+)", line)
#         surrogate_mean_match = re.search(r"Suroggate Mean: ([-+]?\d*\.\d+|\d+)", line)
#         surrogate_covariance_match = re.search(r"Suroggate Covariance: ([-+]?\d*\.\d+|\d+)", line)
#         beta_match = re.search(r"Beta: ([-+]?\d*\.\d+|\d+)", line)

#         if iteration_match:
#             self.bo_data["iteration"].append(int(iteration_match.group(1)))
#         if candidate_match:
#             self.bo_data["candidate"].append(candidate_match.group(1).strip())
#         if acquisition_value_match:
#             self.bo_data["acquisition_value"].append(
#                 float(acquisition_value_match.group(1))
#             )
#         if function_value_match:
#             self.bo_data["function_value"].append(float(function_value_match.group(1)))
#         if final_training_loss_match:
#             self.bo_data["final_training_loss"].append(
#                 float(final_training_loss_match.group(1))
#             )
#         if iteration_time_match:
#             self.bo_data["iteration_time"].append(float(iteration_time_match.group(1)))
#         if surrogate_mean_match:
#             if "surrogate_mean" not in self.bo_data:
#                 self.bo_data["surrogate_mean"] = []
#             self.bo_data["surrogate_mean"].append(float(surrogate_mean_match.group(1)))
#         if surrogate_covariance_match:
#             if "surrogate_covariance" not in self.bo_data:
#                 self.bo_data["surrogate_covariance"] = []
#             self.bo_data["surrogate_covariance"].append(float(surrogate_covariance_match.group(1)))
#         if beta_match:
#             if "beta" not in self.bo_data:
#                 self.bo_data["beta"] = []
#             self.bo_data["beta"].append(float(beta_match.group(1)))

# if __name__ == "__main__":

#     # log_file = "logs/2024-08-23_16-24-48_Warcraft_3x4_architecture-search_4.log"
#     log_file = "/Users/keisukeonoue/ws/constrained_BO/logs/2024-08-23_22-28-52_Warcraft_3x4_unconstrained.log"

#     parser = OptimLogParser(log_file)
#     parser.parse_log_file()

#     print("Experimental settings:")
#     print(parser.settings)

#     print("Initial data:")
#     print(parser.initial_data)

#     print("BO data:")
#     print(parser.bo_data)