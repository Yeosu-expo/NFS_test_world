#!/usr/bin/env bash
# --capture-range=cudaProfilerApi \
# --capture-range-end=stop \
# --------- Nsight-Systems wrapper for DeepSpeed ------------
set -euo pipefail
trap 'echo "[nsys_profile.sh ERROR] $BASH_SOURCE:$LINENO – $BASH_COMMAND"; exit 1' ERR

NSYS="/opt/nvidia/nsight-systems/2025.3.1/bin/nsys"
export LD_LIBRARY_PATH="$NSYS/host-linux-x64:${LD_LIBRARY_PATH}"

# DeepSpeed 런처가 첫 번째 인자로 넘겨주는 --local_rank=* 는
# 단지 파이썬 스크립트용 플래그이므로 Python 쪽엔 **전달하지 않고 폐기**한다.
if [[ $1 == --local_rank=* ]]; then
    shift           # 버린다
fi

SCRIPT=$1           # 학습 스크립트 경로
shift               # 나머지 인수(스크립트 옵션)

exec "$NSYS" profile \
     -t cuda,nvtx,cublas,cudnn,mpi \
     -s none \
     --gpu-metrics-frequency='1000' \
     --capture-range=cudaProfilerApi \
     --capture-range-end=stop \
     --gpu-metrics-devices=cuda-visible \
     --cuda-memory-usage=true \
     --cuda-event-trace=true \
	 --pytorch=autograd-nvtx \
     -o /home/deepspeed/output/nsys_report/nsys_report.%n \
     -- \
     /usr/local/bin/python "$SCRIPT" "$@"
