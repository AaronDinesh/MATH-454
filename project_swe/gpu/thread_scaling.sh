#!/bin/bash

# Thread block configurations to test
CONFIGS=(
    "2 2"
    "4 4"
    "8 8"
    "16 16"
    "32 32"
    "64 64"
    "128 128"
    "256 256"
    "512 512"
    "1024 1024"
)

# Grid sizes for test case 1
SIZES=(
    "256 256"
    "512 512"
    "1024 1024"
    "2048 2048"
    "4096 4096"
)

mkdir -p logs
mkdir -p slurm_jobs

MAX_JOBS=5  # Set this to your SLURM submission quota
CHECK_INTERVAL=5  # seconds to wait before checking again

# Output CSV
OUTPUT_CSV="scaling_results.csv"
echo "TestCase,GridSize,ThreadsX,ThreadsY,TotalThreads,TimeSeconds" > $OUTPUT_CSV

# Submit batch jobs
for size in "${SIZES[@]}"; do
    read -r nx ny <<< "$size"
    for config in "${CONFIGS[@]}"; do
        read -r tx ty <<< "$config"

        # Skip invalid thread configurations
        if (( tx > nx || ty > ny )); then
            echo "Skipping invalid config: threads=($tx,$ty) for grid=($nx,$ny)"
            continue
        fi

        total_threads=$((tx * ty))
        job_name="tc1_${nx}x${ny}_${tx}x${ty}"
        slurm_file="slurm_jobs/${job_name}.slurm"
        cat <<EOF > slurm_jobs/${job_name}.slurm
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.out
#SBATCH --error=logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --account=math-454
#SBATCH --qos=math-454
#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00

module load cuda  # Modify as needed
total_time=0
for run in {1..3}; do
    t=\$(./swe_gpu 1 $nx $ny 0 $tx $ty 2>&1 | grep "Time taken:" | awk '{print \$3}')
    total_time=\$(echo "\$total_time + \$t" | bc -l)
done
avg_time=\$(echo "scale=9; \$total_time / 3" | bc -l)
echo "1,${nx}x${ny},$tx,$ty,$total_threads,\$avg_time" >> ${OUTPUT_CSV}
EOF

        # Wait until job count is under limit
        while true; do
            current_jobs=$(squeue -u $USER | tail -n +2 | wc -l)
            if (( current_jobs < MAX_JOBS )); then
                sbatch $slurm_file
                break
            else
                echo "Waiting: $current_jobs jobs in queue (limit: $MAX_JOBS)..."
                sleep $CHECK_INTERVAL
            fi
        done
    done
done
