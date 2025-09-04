#!/bin/sh -x
#SBATCH --nodes 1
#SBATCH --ntasks-per-core 1



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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Multiple fallback options for MPI execution
echo "Attempting MPI execution with various binding options..."

# Option 1: No binding
echo "Trying: mpirun --bind-to none"
mpirun --bind-to none --oversubscribe python3 $PYTHON_MAIN \
    -data_folder ${WORKER_DIRECTORY}/ -runno $SLURM_ARRAY_TASK_ID \
    ${PYTHON_SCRIPT_FLAGS} \
    2>&1 | tee ${LOGFILE}

# Check if the above succeeded
if [ $? -ne 0 ]; then
    echo "First attempt failed, trying without binding flags..."
    
    # Option 2: Minimal mpirun
    mpirun -np $SLURM_CPUS_PER_TASK python3 $PYTHON_MAIN \
        -data_folder ${WORKER_DIRECTORY}/ -runno $SLURM_ARRAY_TASK_ID \
        ${PYTHON_SCRIPT_FLAGS} \
        2>&1 | tee -a ${LOGFILE}
    
    # Check if that failed too
    if [ $? -ne 0 ]; then
        echo "MPI attempts failed, trying serial execution..."
        
        # Option 3: Serial execution as fallback
        python3 $PYTHON_MAIN \
            -data_folder ${WORKER_DIRECTORY}/ -runno $SLURM_ARRAY_TASK_ID \
            ${PYTHON_SCRIPT_FLAGS} \
            2>&1 | tee -a ${LOGFILE}
    fi
fi

cp -r ${WORKER_DIRECTORY}/* ${DATA_DIRECTORY}/results/

#call this script as: 
#sbatch -J GAUSSIAN -n 1 --array=0-17 --export=L=128,KICK_SOURCE=gaussian queue-2d-01.sh
