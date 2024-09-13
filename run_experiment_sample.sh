#!/bin/bash

# Resource configuration
CPUS_PER_TASK=4  # Adjust the number of CPUs per task
PARTITION="gpu_short"  # Partition name
TIME="4:00:00"  # Maximum execution time

# Create results directory if it doesn't exist
mkdir -p results/

# Create logs directory if it doesn't exist
mkdir -p logs

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/rubust_BO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/logs"

# Overwrite config.ini file
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Run Simple functions
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_ReLU.py"
       
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_ReLU_2.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_ReLU_3.py"
       
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_tanh.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_tanh_2.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_constrained3_d_tanh_3.py"
       
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_unconstrained_d_ReLU.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-09-03_botorch_simple/Simple_5d_unconstrained_d_tanh.py"
