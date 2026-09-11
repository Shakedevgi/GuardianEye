#!/bin/bash
# GuardianEye - start everything with one command.
#
# Replaces the three-terminal ritual (camera+backend, then UI, then remember
# which URL works). Shaked, 2026-08-30: "we can't each time do like 3 terminal
# commands and run it like devs."
#
#   ./start.sh              camera + web UI, Mac only (this is the normal way)
#   ./start.sh --desktop    also open the desktop window
#
# Mobile/tablet access is parked for now - see docs/decision-log.md,
# 2026-08-30 "iPad/mobile access reverted" - so this only ever serves
# 127.0.0.1 (this Mac). No certificates, no setup, nothing to install.
#
# PIN comes from GUARDIANEYE_PIN, or defaults to 4821.
# Ctrl-C stops everything.

set -euo pipefail
cd "$(dirname "$0")"

PIN="${GUARDIANEYE_PIN:-4821}"
CAMERA="${GUARDIANEYE_CAMERA:-Arducam}"
UI_PORT=8551
BACKEND_PORT=8000
DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --desktop) DESKTOP=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

PIDS=()
cleanup() {
  echo ""
  echo "stopping..."
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "stopped."
}
trap cleanup EXIT INT TERM

# Anything left over from a previous run will hold the ports and the camera,
# and the failure ("address already in use") is buried in a log nobody reads.
for port in "$BACKEND_PORT" "$UI_PORT"; do
  lsof -ti tcp:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 1

echo "1/2  starting camera + backend..."
( cd cv && exec ../.venv/bin/python3 risk_engine.py --name "$CAMERA" --serve ) &
PIDS+=($!)

# Wait for the camera loop to actually publish a frame. Starting the UI first
# just means it shows "server unreachable" until the model finishes loading,
# which reads as a broken app rather than a slow start.
printf "     waiting for the camera"
for i in $(seq 1 90); do
  if curl -sf -m 2 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo " - ready."
    break
  fi
  printf "."
  sleep 1
  if [ "$i" -eq 90 ]; then
    echo ""
    echo "     camera did not come up in 90s. Is it plugged in? Is the name '$CAMERA' right?"
    echo "     (set a different one with GUARDIANEYE_CAMERA=...)"
    exit 1
  fi
done

echo "2/2  starting the app..."
( cd ui && exec ../.venv/bin/python3 app.py --mode web --pin "$PIN" --web-port "$UI_PORT" ) &
PIDS+=($!)

if [ "$DESKTOP" = "1" ]; then
  sleep 3
  ( cd ui && exec ../.venv/bin/python3 app.py --pin "$PIN" ) &
  PIDS+=($!)
fi

echo ""
echo "PIN: $PIN     (Ctrl-C to stop everything)"
wait
