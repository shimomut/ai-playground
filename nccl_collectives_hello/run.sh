#!/usr/bin/env bash
# Launch WORLD_SIZE rank processes on this node, one process per rank.
#
# This mirrors how a real multi-GPU / multi-node job is launched (one process
# per GPU, each given RANK/WORLD_SIZE/LOCAL_RANK and a shared bootstrap address).
# Here the bootstrap is a unique-id file that rank 0 writes and the others read.
#
# Env knobs:
#   WORLD_SIZE  number of ranks (default 2)
#   COUNT       elements per buffer (default 1<<20)
#   NCCL_DEBUG  set to INFO to watch topology + transport selection
set -euo pipefail

cd "$(dirname "$0")"

BIN=build/nccl_collectives
if [[ ! -x "$BIN" ]]; then
  echo "binary not found; run 'make' first" >&2
  exit 1
fi

WORLD_SIZE=${WORLD_SIZE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
COUNT=${COUNT:-1048576}
ID_FILE="$(mktemp -u "${TMPDIR:-/tmp}/nccl_id.XXXXXX")"
rm -f "$ID_FILE"

echo "launching $WORLD_SIZE ranks (count=$COUNT, NCCL_DEBUG=${NCCL_DEBUG:-WARN})..."

pids=()
for ((r = 0; r < WORLD_SIZE; r++)); do
  RANK=$r \
  WORLD_SIZE=$WORLD_SIZE \
  LOCAL_RANK=$r \
  COUNT=$COUNT \
  NCCL_ID_FILE="$ID_FILE" \
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    "$BIN" &
  pids+=($!)
done

status=0
for p in "${pids[@]}"; do
  wait "$p" || status=1
done

rm -f "$ID_FILE"
exit $status
