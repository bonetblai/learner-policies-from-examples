#!/bin/bash
# xxx #SBATCH --mail-user=bonetblai@gmail.com
# xxx #SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
# xxx1 #SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --time=8:00:00
# xxx #SBATCH --account=blai.bonet
#SBATCH --partition=rleap_cpu
#SBATCH --output=slurm/logs/testing/%x.output.%A
#SBATCH --error=slurm/logs/testing/%x.error.%A

SCRIPT_NAME=$SLURM_JOB_NAME
DOMAIN=$1
SKETCHES=$2
OUTPUT=$3

. ~/.bashrc
conda activate sketches
python test_sketch.py --domain $DOMAIN --sketches $SKETCHES --output $OUTPUT --no_filtering

