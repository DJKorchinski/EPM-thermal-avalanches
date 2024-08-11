SLURMOUT_DIR=slurmout_01
SUB_SCRIPT=queue-01.sh
mkdir -p $SLURMOUT_DIR

sbatch -J low-V-cross-L032-2d -n 1   --array=0-95      --export=L=32,KICK_SOURCE=2d  --mem=8G    --time=3-13:00:00 -o $SLURMOUT_DIR/slurm-%A_%a.out $SUB_SCRIPT 
sbatch -J low-V-cross-L064-2d -n 4   --array=0-95      --export=L=64,KICK_SOURCE=2d  --mem=12G   --time=3-13:00:00 -o $SLURMOUT_DIR/slurm-%A_%a.out $SUB_SCRIPT 
sbatch -J low-V-cross-L128-2d -n 7   --array=0-95     --export=L=128,KICK_SOURCE=2d  --mem=20G   --time=5-13:00:00 -o $SLURMOUT_DIR/slurm-%A_%a.out $SUB_SCRIPT 
sbatch -J low-V-cross-L256-2d -n 14  --array=0-95     --export=L=256,KICK_SOURCE=2d  --mem=55G   --time=7-13:00:00 -o $SLURMOUT_DIR/slurm-%A_%a.out $SUB_SCRIPT 
sbatch -J low-V-cross-L512-2d -n 28  --array=0-95%3   --export=L=512,KICK_SOURCE=2d  --mem=112G  --time=7-13:00:00 -o $SLURMOUT_DIR/slurm-%A_%a.out $SUB_SCRIPT 
