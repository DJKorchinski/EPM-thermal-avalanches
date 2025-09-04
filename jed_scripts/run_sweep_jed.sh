#!/bin/sh
#SBATCH --nodes 1



# module load anaconda3 
source /home/korchins/.bashrc
conda activate fenics05



START_DIR="$PWD"
echo Start directory: $START_DIR
echo Output directory: $DATA_DIRECTORY
echo Base scratch working directory: ${SCRATCH_WORKING_DIRECTORY}

#Setting up the slurm worker directory. 
REPNO_PADDED=$(printf %04d ${SLURM_ARRAY_TASK_ID})
WORKER_DIRECTORY=${SCRATCH_WORKING_DIRECTORY}/${REPNO_PADDED}
mkdir -p $WORKER_DIRECTORY
#making sure we clean out any old files that might be present: 
rm -f ${WORKER_DIRECTORY}/*
LOGFILE=${WORKER_DIRECTORY}/${REPNO_PADDED}.log
echo worker directory: $WORKER_DIRECTORY
echo logfile: $LOGFILE
echo Python script flags: ${PYTHON_SCRIPT_FLAGS}


export OMP_NUM_THREADS=1

# Debug information
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# OpenMPI environment variables to handle binding issues
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_mpi_warn_on_fork=0

echo "Running: mpirun -np $SLURM_CPUS_PER_TASK --bind-to none --oversubscribe python3..."
mpirun -np $SLURM_CPUS_PER_TASK --bind-to none --oversubscribe python3 $PYTHON_MAIN \
    -data_folder ${WORKER_DIRECTORY}/ -runno $SLURM_ARRAY_TASK_ID \
    ${PYTHON_SCRIPT_FLAGS} \
    | tee ${LOGFILE}

cp -r ${WORKER_DIRECTORY}/* ${DATA_DIRECTORY}/results/

#call this script as: 
#sbatch -J GAUSSIAN -n 1 --array=0-17 --export=L=128,KICK_SOURCE=gaussian queue-2d-01.sh 