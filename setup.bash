uv venv --python 3.10
source .venv/bin/activate
uv sync

git submodule update --init --recursive
uv pip install -e pyroki
# uv pip install -e lerobot

# Run from repo root
python3 bimanual-e2e/server.py
python3 bimanual-e2e/sim.py