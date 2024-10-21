#!/bin/bash
#SBATCH --job-name=mtt_heatmap
#SBATCH --output=mtt_heatmap_%j.out
#SBATCH --error=mtt_heatmap_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load any necessary modules (adjust these based on your cluster setup)
module load cuda/11.3
module load python/3.8

# Activate your virtual environment if you're using one
source /path/to/your/venv/bin/activate

# Navigate to the directory containing your script
cd /path/to/your/project/directory

# Run the script
python src/mtt_heatmap.py

# Deactivate the virtual environment
deactivate
