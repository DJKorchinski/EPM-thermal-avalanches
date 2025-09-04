#!/bin/sh -x
PARTITION=standard #l40s or h100
QOS=debug
SIMULATION_IDENTIFIER=test_run_01
PYTHON_MAIN=run_01_prod.py
# SCRATCH_HOME=/scratch/korchins # I omit this, because it is defined in an environment variable set in my initrc. 
PYTHON_SCRIPT_FLAGS="\
--arrhenius \
-L 16 \
-kicksource 2d \
"
#setting up the sweep parameters. These may overwrite the commands issued above.
PYTHON_SCRIPT_FLAGS+=" \
"

export PYTHON_SCRIPT_FLAGS

#compute over a slurm job array
ARRAY_MIN=0
ARRAY_MAX=0
MAXTIME="0-00:05:00"
MAXMEM="8G"
MAXCPU=4
PCSL_DATA_DIRECTORY=/work/pcsl/korchins/projects/epm/cyclic_epm_data
SCRIPTS_FOLDER=jed_scripts
SUB_SCRIPT=${SCRIPTS_FOLDER}/run_sweep_jed_robust.sh #TODO: update this submission script.
# POSTPROCESS_SCRIPT=run_scripts/run_sweep_copy_data.sh

DATA_DIRECTORY=cyclic_epm_data/${SIMULATION_IDENTIFIER}
SLURMOUT_DIR=${DATA_DIRECTORY}/slurmout_${SIMULATION_IDENTIFIER}
WD_NAME=${SIMULATION_IDENTIFIER}_wd
QUEUE_SCRIPT=$(realpath "$0")

#I want to store the data for this simulation on the work/pcsl directory, so we create a symlink to the data directory.
mkdir -p ${PCSL_DATA_DIRECTORY}
ln -s -f ${PCSL_DATA_DIRECTORY}
mkdir -p ${DATA_DIRECTORY}
mkdir -p ${DATA_DIRECTORY}/code
mkdir -p ${DATA_DIRECTORY}/results
mkdir -p ${DATA_DIRECTORY}/${SCRIPTS_FOLDER}
mkdir -p $SLURMOUT_DIR

#copying all input files to the output directory, for full(er) reproducibility.
rsync -aqu dklib ${DATA_DIRECTORY}/
cp *.py ${DATA_DIRECTORY}/code/
cp ${SUB_SCRIPT} ${DATA_DIRECTORY}/${SCRIPTS_FOLDER}/
cp ${QUEUE_SCRIPT} ${DATA_DIRECTORY}/${SCRIPTS_FOLDER}/
# cp ${POSTPROCESS_SCRIPT} ${DATA_DIRECTORY}/${SCRIPTS_FOLDER}/


#making the working directory: 
SCRATCH_WORKING_DIRECTORY=${SCRATCH_HOME}/${WD_NAME}
mkdir -p ${SCRATCH_WORKING_DIRECTORY}

# -q debug \ for the debug queue
MAIN_JOBID=$(sbatch --parsable \
 --array=${ARRAY_MIN}-${ARRAY_MAX} \
 -o ${SLURMOUT_DIR}/slurm-%A_%a.out \
 --time=${MAXTIME} \
 --mem=${MAXMEM} \
 --cpus-per-task=${MAXCPU} \
 -p ${PARTITION} \
 -q ${QOS} \
 --job-name=${SIMULATION_IDENTIFIER} \
 --export=DATA_DIRECTORY=${DATA_DIRECTORY},\
SCRATCH_WORKING_DIRECTORY=${SCRATCH_WORKING_DIRECTORY},\
PYTHON_MAIN=${PYTHON_MAIN},\
TQDM_DISABLE=1,\
PYTHON_SCRIPT_FLAGS \
 $SUB_SCRIPT )
echo "Main job id: $MAIN_JOBID"
