#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 -m MODEL -s SHARD -l LENGTH -b BATCH"
  exit 1
}

# Parse script parameters
while getopts "m:s:l:b:" opt; do
  case ${opt} in
    m) MODEL=${OPTARG} ;;  # 모델 이름
    s) SHARD=${OPTARG} ;;  # shard 수
    l) LENGTH=${OPTARG} ;; # sample length
    b) BATCH=${OPTARG} ;;
    *) usage ;;
  esac
done

# Ensure required params are set
if [[ -z "${MODEL:-}" || -z "${SHARD:-}" || -z "${LENGTH:-}" || -z "${BATCH:-}" ]]; then
  usage
fi

# Directory to store logs
LOGDIR="/home/deepspeed/output/logs"
mkdir -p "${LOGDIR}"

# Define cases: offload(o), recompute(r), prefetch(p), and filename suffix
cases=(
  # o r p q suffix
  # "0 0 0 0 plain"
  "1 0 0 0 o"
  "1 0 0 1 oq"
  "1 1 0 1 orq"
  "1 0 2 1 opq"
  "1 1 2 1 orpq"
)

# Loop through each configuration
for case in "${cases[@]}"; do
  read -r O R P Q SUFFIX <<< "${case}"
  LOGFILE="${LOGDIR}/${MODEL}_${SUFFIX}_${BATCH}.log"
  echo "[RUNNING] MODEL=${MODEL} SHARD=${SHARD} LENGTH=${LENGTH} BATCH=${BATCH} -o ${O} -r ${R} -p ${P} -q ${Q} -> ${LOGFILE}"

  deepspeed --no_python \
    --hostfile=/home/deepspeed/hostfile.txt \
    --master_addr $MANAGER_IP \
    --master_port 29500 \
    --num_nodes=5 \
    --num_gpus=1 \
    /home/deepspeed/nsys_profile.sh \
    /home/deepspeed/AO_DS_test_m.py \
    -m "${MODEL}" \
    -s "${SHARD}" \
    -o "${O}" \
    -r "${R}" \
    -p "${P}" \
    -q "${Q}" \
    -l "${LENGTH}" \
    -d 0 \
    -n 1 \
    2>&1 | tee "${LOGFILE}"
done
