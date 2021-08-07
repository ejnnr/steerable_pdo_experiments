#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_rtx2080ti_shared
#SBATCH --gpus=1
#SBATCH -t 00:30:00

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
    dir.run="diffop_5x5/gauss/$SLURM_ARRAY_TASK_ID" \
    +experiment=diffop_5x5 \
    +group="diffop_5x5 gauss" \
    +name="diffop_5x5 gauss seed $SLURM_ARRAY_TASK_ID" \
    +model.smoothing=1.3 \
    seed="$SLURM_ARRAY_TASK_ID" \
    +trainer.gpus=1 \
    +trainer.progress_bar_refresh_rate=0 \
    data.num_workers="$N_CPUS"

# Copy output data from scratch to home
# the -T flag ensures that we copy onto rather than into the logs directory
# (i.e. not to logs/logs/...)
cp -rT "$TMPDIR"/logs logs
