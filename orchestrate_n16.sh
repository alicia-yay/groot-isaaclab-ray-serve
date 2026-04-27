#!/bin/bash
# Orchestrator: launches inference server (groot-n16 env) + sim runner (base env)
# both on the same worker, communicating via /tmp/bridge/

set -e

# Clean bridge dir
rm -rf /tmp/bridge
mkdir -p /tmp/bridge/req /tmp/bridge/resp

# Launch inference server in groot-n16 env in background
echo "[orch] starting N1.6 inference server in groot-n16 env..."
/home/ray/anaconda3/envs/groot-n16/bin/python -u /home/ray/groot_demo/n16_inference_server.py \
    > /tmp/bridge/server.log 2>&1 &
SERVER_PID=$!
echo "[orch] server PID=$SERVER_PID (logs: /tmp/bridge/server.log)"

# Launch sim runner in base env (foreground)
echo "[orch] starting Isaac Lab sim runner in base env..."
/home/ray/anaconda3/bin/python -u /home/ray/groot_demo/sim_runner_n16.py \
    > /tmp/bridge/sim.log 2>&1
SIM_RC=$?

echo "[orch] sim runner finished (rc=$SIM_RC)"

# Stop server
echo "[orch] stopping server..."
touch /tmp/bridge/STOP
wait $SERVER_PID 2>/dev/null || true

echo "--- sim.log tail ---"
tail -40 /tmp/bridge/sim.log
echo
echo "--- server.log tail ---"
tail -20 /tmp/bridge/server.log

exit $SIM_RC
