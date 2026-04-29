#!/bin/bash
#SBATCH --job-name=odelia_swin
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=Log_and_helper_files/slurm_cv_%j.log
#SBATCH --mail-user=thouansimon@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Stop the script if a command fails
set -e

# Managing the session name using the $1 argument
# If no argument is provided, the default date is used
SESSION_NAME=${1:-"training_$(date +%d_%b_%Hh%M)"}
OUTPUT_PATH="Training_results/$SESSION_NAME"

echo "Nom de la session : $SESSION_NAME"
echo "Dossier de sortie : $OUTPUT_PATH"

# Creating the necessary folders
mkdir -p Log_and_helper_files
mkdir -p "$OUTPUT_PATH"

# Preparing the environment
module load Anaconda3/2023.09-0
conda activate odelia

# Cross-Validation loop
for f in 0 1 2 3 4
do
   echo "[$(date)] START FOLD $f"
   
   python Python/train_odelia.py --fold $f --out_dir "$OUTPUT_PATH"
   python Python/generate_predictions.py --fold $f --out_dir "$OUTPUT_PATH"
done

# Final ensembling step
echo "[$(date)] START ENSEMBLING"
python Python/ensemble_predictions.py "$OUTPUT_PATH"
