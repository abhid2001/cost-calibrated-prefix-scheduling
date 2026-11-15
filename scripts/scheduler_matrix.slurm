#!/usr/bin/env bash
#SBATCH --job-name=prefix-matrix
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=matrix-%A_%a.out
set -euo pipefail
# Deliberately fail closed: the installed engine does not yet support the
# calibrated or k-LPM policy, or GPU prefill telemetry. Do not label FCFS as
# another policy. Replace this gate only after implementing and validating both.
echo 'BLOCKED: live FCFS/k-LPM/calibrated admission integration and calibrated GPU prefill telemetry are not implemented. No valid scheduler matrix can be run.' >&2
exit 1
