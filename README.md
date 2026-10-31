# ECE 525 SmolVLA Dashboard

This dashboard runs the LIBERO simulator on PC1 and sends real observations to SmolVLA running on either the Jetson AGX Orin or PC2 RTX 4090.

It displays:

- two LIBERO camera views;
- the 8D robot state and task instruction;
- returned 7D actions and inference latency;
- a video for each inference chunk;
- one combined video for a full rollout.

## Start the dashboard

Run from the project root on PC1 (the directory containing both `midterm_dashboard/` and `lerobot/`):

```bash
MUJOCO_GL=egl \
midterm_dashboard/.venv/bin/python \
-m midterm_dashboard.app
```

Open:

```text
http://localhost:7860/
```

The server listens only on `127.0.0.1`. When using VS Code Remote SSH, forward port `7860` as a private port.

If port `7860` is already in use, check whether the dashboard is already running:

```bash
pgrep -af midterm_dashboard.app
ss -ltnp | grep ':7860 '
```

## Use the dashboard

1. Select a LIBERO task.
2. Select **Jetson AGX Orin** or **PC2 RTX 4090**.
3. Click **Start** to reset the task and capture the first observation.
4. Click **Single Run & Record** to request and execute one 50-action chunk.
5. Use **Run Full Task & Record** to repeat inference until success, environment termination, or the selected chunk limit.

SmolVLA returns up to 50 actions per request. At the 20 Hz LIBERO control rate, one complete chunk represents up to 2.5 seconds of movement.

## Jetson server and tunnel

Start the server on the Jetson:

```bash
ssh <JETSON_USER>@<JETSON_IP>

source smolvla/smolvla/bin/activate
cd smolvla/lerobot

lerobot-policy-server --config_path=libero_server.yaml
```

Wait for `Policy server ready`, then leave the terminal open.

Start the tunnel from PC1 in another terminal:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:7447:127.0.0.1:7447 \
  <JETSON_USER>@<JETSON_IP>
```

Jetson dashboard configuration:

```text
Endpoint:   tcp/127.0.0.1:7447
Deployment: libero_jetson
Semantics:  libero-relative-eef-v1
Zenoh mode: peer
```

## PC2 server and tunnel

On PC2, enter WSL and start the server:

```bash
conda activate smolvla
cd ece525-smolvla/lerobot

lerobot-policy-server \
  --config_path=../configs/libero_server.yaml
```

Verify the listener on PC2:

```bash
ss -ltnp | grep ':7447 '
```

Start the tunnel from PC1:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:17447:127.0.0.1:7447 \
  <PC2_USER>@<PC2_IP>
```

PC2 dashboard configuration:

```text
Endpoint:   tcp/127.0.0.1:17447
Deployment: smolvla_libero
Semantics:  libero-relative-eef-v1
Zenoh mode: client
```

If the tunnel reports `Connection refused`, the remote policy server is not reachable at its destination. Check the server terminal and listener before changing the dashboard.

## Environment

The dashboard overlay is:

```text
midterm_dashboard/.venv
```

It reuses the existing Conda environment through system site packages. Do not reinstall CUDA, PyTorch, LeRobot, or LIBERO for the dashboard.

UI dependency versions are recorded in:

```text
midterm_dashboard/requirements.txt
midterm_dashboard/requirements-ui-lock.txt
```

## Tests

Run from the project root:

```bash
midterm_dashboard/.venv/bin/python -m unittest discover \
  -s midterm_dashboard/tests -v

midterm_dashboard/.venv/bin/python -m ruff check midterm_dashboard
midterm_dashboard/.venv/bin/python -m ruff format --check midterm_dashboard
```

The unit tests do not prove that a remote server or SSH tunnel is currently available.
