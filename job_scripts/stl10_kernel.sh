#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_rtx2080ti_shared
#SBATCH --gpus=1
#SBATCH -t 10:00:00

#Loading modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0




N_CPUS=$(nproc)
echo "Running on $N_CPUS cores"
echo "Run number $SLURM_ARRAY_TASK_ID"

# Copy input data to scratch and create output directory
cp -r data "$TMPDIR"
echo "$TMPDIR"

poetry run python main.py data.dir="$TMPDIR/data" dir.output_base="$TMPDIR" \
    dir.run="stl10_kernel/$SLURM_ARRAY_TASK_ID" \
    +group="stl10_kernel" \
    +name="stl10 kernel seed $SLURM_ARRAY_TASK_ID" \
    +experiment=stl10_repro \
    seed=$SLURM_ARRAY_TASK_ID \
    +trainer.gpus=1 \
    +trainer.progress_bar_refresh_rate=0 \
    data.num_workers="$N_CPUS"

# Copy output data from scratch to home
# the -T flag ensures that we copy onto rather than into the logs directory
# (i.e. not to logs/logs/...)
cp -rT "$TMPDIR"/logs logs
