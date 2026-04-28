#!/bin/bash
set -e
rm -rf /tmp/bridge
mkdir -p /tmp/bridge/req /tmp/bridge/resp

echo "[orch] starting N1.6 inference server..."
/home/ray/anaconda3/envs/groot-n16/bin/python -u /home/ray/groot_demo/n16_inference_server.py \
    > /tmp/bridge/server.log 2>&1 &
SERVER_PID=$!
echo "[orch] server PID=$SERVER_PID"

echo "[orch] starting sim runner v2 (real joint state + 128 steps)..."
/home/ray/anaconda3/bin/python -u /home/ray/groot_demo/sim_runner_n16_v2.py \
    > /tmp/bridge/sim.log 2>&1
SIM_RC=$?

echo "[orch] sim finished rc=$SIM_RC"
touch /tmp/bridge/STOP
wait $SERVER_PID 2>/dev/null || true

echo "--- sim.log tail ---"
tail -50 /tmp/bridge/sim.log
echo
echo "--- server.log tail ---"
tail -10 /tmp/bridge/server.log

exit $SIM_RC
