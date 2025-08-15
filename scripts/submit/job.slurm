#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/default/slurm.output.%A
#SBATCH --error=slurm/logs/default/slurm.error.%A

mkdir -p slurm/logs/default
ulimit -a

DOMAIN=$1
NAME=`basename $DOMAIN`
SUFFIX=$2
shift 2
ARGS="$*"

REQUIRED_ARGS="--domain_filepath $DOMAIN/domain.pddl --problems_directory $DOMAIN/training/easy --workspace $DOMAIN/workspace"
PLANNER=`python scripts/submit/parse_option.py --planner "$ARGS"`
STRATEGY=`python scripts/submit/parse_option.py --instance_selection "$ARGS"`

DATEFMT=`date +"%F"`
if [ "$SUFFIX" == "<hour>" ]; then
    DATEFMT+=`date +".%Hh"`
else
    DATEFMT+=$SUFFIX
fi
LOG_FOLDER=slurm/logs/${DATEFMT}/${PLANNER}.${STRATEGY}
mkdir -p ${LOG_FOLDER}

LOGFILE_OUTPUT=${LOG_FOLDER}/${NAME}.output.${SLURM_JOB_ID}
SLURM_OUTPUT=slurm/logs/default/slurm.output.${SLURM_JOB_ID}
echo ln -f ${SLURM_OUTPUT} ${LOGFILE_OUTPUT}
ln -f ${SLURM_OUTPUT} ${LOGFILE_OUTPUT}

LOGFILE_ERROR=${LOG_FOLDER}/${NAME}.error.${SLURM_JOB_ID}
SLURM_ERROR=slurm/logs/default/slurm.error.${SLURM_JOB_ID}
echo ln -f ${SLURM_ERROR} ${LOGFILE_ERROR}
ln -f ${SLURM_ERROR} ${LOGFILE_ERROR}

. ~/.bashrc
conda activate sketches
python learning/main.py $REQUIRED_ARGS $ARGS

