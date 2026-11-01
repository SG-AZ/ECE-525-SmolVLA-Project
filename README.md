# ECE 525 SmolVLA Dashboard

This project runs LIBERO on a simulator PC and sends real observations to a remote SmolVLA inference server. The dashboard displays two camera views, the 8D robot state, task instruction, returned 7D actions, round-trip latency, and recorded rollout videos.

## Clone

LeRobot is included as a pinned Git submodule, so clone with submodules enabled:

```bash
git clone --recurse-submodules \
  https://github.com/SG-AZ/ECE-525-SmolVLA-Project.git

cd ECE-525-SmolVLA-Project
```

If the repository was cloned without the submodule:

```bash
git submodule update --init --recursive
```

## Environment setup

The dashboard requires Linux, Python 3.12, a CUDA-compatible PyTorch installation, and the LIBERO assets used by LeRobot. The following creates a new Conda environment without modifying an existing CUDA installation:

```bash
conda create -n smolvla python=3.12 -y
conda activate smolvla

python -m pip install -e "./lerobot[libero,remote,smolvla]"

python -m venv --system-site-packages midterm_dashboard/.venv
midterm_dashboard/.venv/bin/python -m pip install \
  -r midterm_dashboard/requirements-ui-lock.txt
```

The UI overlay reuses PyTorch, LeRobot, and LIBERO from the Conda environment. Do not reinstall CUDA or GPU PyTorch inside `midterm_dashboard/.venv`.

## Inference server configuration

Server files are machine-specific and may contain local paths, so they are not stored in this repository. Create a YAML file on each inference machine using this template:

```yaml
deployment: <DEPLOYMENT_NAME>

model:
  repo_or_path: lerobot/smolvla_libero
  device: cuda

execution:
  supported_modes: [chunk]
  n_action_steps: 50
  action_fps: 20.0
  action_deadline_s: 10.0
  warmup_calls: 2

language:
  enabled: false

zenoh:
  mode: <ZENOH_MODE>
  listen_endpoints:
    - tcp/127.0.0.1:7447

semantics: libero-relative-eef-v1
robot_type: libero_sim

features:
  - name: observation.state
    shape: [8]
    dtype: float32
    kind: tensor
    semantics: libero-state-v1
  - name: observation.images.camera1
    shape: [256, 256, 3]
    dtype: uint8
    kind: rgb
    semantics: libero-agentview-rgb-v1
  - name: observation.images.camera2
    shape: [256, 256, 3]
    dtype: uint8
    kind: rgb
    semantics: libero-wrist-rgb-v1

action_feature:
  name: action
  shape: [7]
  dtype: float32
  kind: tensor
  names: [delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper]
  semantics: libero-relative-eef-v1

log_level: INFO
```

Start the server from its LeRobot checkout:

```bash
conda activate smolvla
cd <LEROBOT_CHECKOUT>
lerobot-policy-server --config_path=<SERVER_CONFIG_PATH>
```

Wait for `Policy server ready` before starting a tunnel.

## SSH tunnel templates

Jetson tunnel on local port `7447`:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:7447:127.0.0.1:7447 \
  <JETSON_USER>@<JETSON_IP>
```

PC2 tunnel on local port `17447`:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:17447:127.0.0.1:7447 \
  <PC2_USER>@<PC2_IP>
```

The SSH command stays open without printing output while the tunnel is active. `Connection refused` means the policy server is not listening at the remote destination.

## Start the dashboard

Run from the repository root:

```bash
MUJOCO_GL=egl \
midterm_dashboard/.venv/bin/python \
-m midterm_dashboard.app
```

Open `http://127.0.0.1:7860/`. The dashboard binds only to localhost; use private VS Code port forwarding when accessing it through Remote SSH.

If port `7860` is already occupied:

```bash
pgrep -af midterm_dashboard.app
ss -ltnp | grep ':7860 '
```

## Dashboard workflow

1. Select a `libero_object` task.
2. Select the Jetson or PC2 inference target.
3. Click **Start** to reset LIBERO and capture the first observation.
4. Click **Single Run & Record** to infer and execute one action chunk.
5. Use **Run Full Task & Record** for repeated observe-infer-execute cycles up to the selected safety limit.

SmolVLA returns up to 50 actions per request. At the 20 Hz LIBERO control rate, a complete chunk represents up to 2.5 seconds of movement.

## Checks

These committed smoke scripts require the Linux simulator environment:

```bash
# Real reset and observation formatting; no remote inference
MUJOCO_GL=egl midterm_dashboard/.venv/bin/python \
  -m midterm_dashboard.smoke

# Local Gradio HTTP integration on port 7861
MUJOCO_GL=egl midterm_dashboard/.venv/bin/python \
  -m midterm_dashboard.http_smoke

# Static checks
midterm_dashboard/.venv/bin/ruff check midterm_dashboard
midterm_dashboard/.venv/bin/ruff format --check midterm_dashboard
```

The smoke checks do not prove that a remote inference server or SSH tunnel is currently available.

## Current scope

The dashboard supports real LIBERO observations, remote inference, action execution, latency display, and video recording. Formal quantization comparisons, task-success rates, power measurements, and repeated hardware benchmarks are separate evaluation work and are not claimed complete here.
