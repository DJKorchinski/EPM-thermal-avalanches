#!/bin/sh -x
#SBATCH --partition=nodes
#SBATCH --nodes 1
#SBATCH --ntasks-per-core 1

# module load anaconda3 
# source activate fenics05
# conda activate fenics05

SPACK_ENV_DIR=/spack/var/spack/environments/fenics-test
source ${SPACK_ENV_DIR}/load_modules.sh

DATA_DIRECTORY=data_01
LOGS_DIRECTORY=logs_01
SCRATCH_HOME=/scratch/djkorchi/2024_01_16_av_phase_dia
PYTHON_FILES="run_01_prod.py sim_params_01.py"
MAIN_PYTHON="run_01_prod.py"
START_DIR="$PWD"

#making the directories in the home
mkdir -p ${DATA_DIRECTORY}
mkdir -p ${LOGS_DIRECTORY}

#copying over code files
mkdir -p ${SCRATCH_HOME}
cp -r dklib ${SCRATCH_HOME}/
cp ${PYTHON_FILES} ${SCRATCH_HOME}/

#setting up simulation numbers: 
repno_padded=`printf %04d ${SLURM_ARRAY_TASK_ID}`
L_padded=`printf %04d $L`
SIM_ID=${L_padded}_${repno_padded}_${KICK_SOURCE}

#making output directories
cd ${SCRATCH_HOME}
DATA_DIRECTORY_LOCAL=${DATA_DIRECTORY}_${SIM_ID}
LOGS_DIRECTORY_LOCAL=${LOGS_DIRECTORY}_${SIM_ID}
mkdir -p ${DATA_DIRECTORY_LOCAL}
mkdir -p ${LOGS_DIRECTORY_LOCAL}


export OMP_NUM_THREADS=1
mpirun --bind-to core:overload-allowed python3 $MAIN_PYTHON -kicksource $KICK_SOURCE --arrhenius -data_folder ${DATA_DIRECTORY_LOCAL}/ -runno $SLURM_ARRAY_TASK_ID -L $L > ${LOGS_DIRECTORY_LOCAL}/${SIM_ID}.log 

mv ${DATA_DIRECTORY_LOCAL}/* ${START_DIR}/${DATA_DIRECTORY}/ && rm -r ${DATA_DIRECTORY_LOCAL}
mv ${LOGS_DIRECTORY_LOCAL}/* ${START_DIR}/${LOGS_DIRECTORY}/ && rm -r ${LOGS_DIRECTORY_LOCAL}

#call this script as: 
#sbatch -J GAUSSIAN -n 1 --array=0-17 --export=L=128,KICK_SOURCE=gaussian queue-2d-01.sh 