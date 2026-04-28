#!/bin/bash
# Reorganize the repo into demos/, path_a_ray_serve/, path_b_file_bridge/, tools/
# Run from the repo root: bash reorganize.sh

set -e
cd "$(dirname "$0")"

echo "Creating directories..."
mkdir -p demos path_a_ray_serve path_b_file_bridge tools

echo "Moving GIFs..."
[ -f g1_groot_n17_zeroshot.gif ] && git mv g1_groot_n17_zeroshot.gif demos/
[ -f g1_groot_n16_g1pnp.gif ] && git mv g1_groot_n16_g1pnp.gif demos/
[ -f g1_groot_n16_polished.gif ] && git mv g1_groot_n16_polished.gif demos/

echo "Moving Path A files (Ray Serve, N1.7)..."
[ -f g1_env.py ] && git mv g1_env.py path_a_ray_serve/
[ -f policy_server.py ] && git mv policy_server.py path_a_ray_serve/
[ -f sim_worker.py ] && git mv sim_worker.py path_a_ray_serve/
[ -f run_demo.py ] && git mv run_demo.py path_a_ray_serve/
[ -f single_shot.py ] && git mv single_shot.py path_a_ray_serve/

echo "Moving Path B files (file-bridge, N1.6)..."
[ -f n16_inference_server.py ] && git mv n16_inference_server.py path_b_file_bridge/
[ -f sim_runner_n16.py ] && git mv sim_runner_n16.py path_b_file_bridge/
[ -f orchestrate_n16.sh ] && git mv orchestrate_n16.sh path_b_file_bridge/

echo "Moving tools..."
[ -f polish_gif.py ] && git mv polish_gif.py tools/
[ -f set_token.py ] && git mv set_token.py tools/
[ -f setup_workers.sh ] && git mv setup_workers.sh tools/
[ -f test_g1_sim.py ] && git mv test_g1_sim.py tools/
[ -f test_groot_standalone.py ] && git mv test_groot_standalone.py tools/

echo "Removing draft v2/v3 files (kept in git history)..."
[ -f sim_runner_n16_v2.py ] && git rm sim_runner_n16_v2.py
[ -f sim_runner_n16_v3.py ] && git rm sim_runner_n16_v3.py
[ -f orchestrate_n16_v2.sh ] && git rm orchestrate_n16_v2.sh
[ -f orchestrate_n16_v3.sh ] && git rm orchestrate_n16_v3.sh

echo "Removing CLI tarball if present..."
[ -f gh_2.59.0_linux_amd64.tar.gz ] && git rm gh_2.59.0_linux_amd64.tar.gz

echo "Updating .gitignore..."
cat >> .gitignore << GIGNORE

# CLI binaries and archives
gh_*.tar.gz
gh_*_linux_*/
GIGNORE

echo
echo "Done. Review with:"
echo "  git status"
echo
echo "Then commit:"
echo '  git commit -m "Reorganize repo into Path A (Ray Serve), Path B (file-bridge), tools"'
