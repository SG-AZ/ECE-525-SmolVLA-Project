"""using real libero observations for remote inference."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JETSON_TARGET_NAME = "Jetson AGX Orin"
PC2_TARGET_NAME = "PC2 RTX 4090"


@dataclass(frozen=True)
class Target:
    endpoint: str = "tcp/127.0.0.1:7447"
    deployment: str = "libero_jetson"
    semantics: str = "libero-relative-eef-v1"
    zenoh_mode: str = "peer"
    language_enabled: bool = False


def format_observation(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """converting the processed libero tensors into the remote format."""
    state = obs["observation.state"].detach().cpu().numpy()
    if state.shape != (1, 8) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite batched 8D state; got {state.shape}")
    features = {"observation.state": np.ascontiguousarray(state[0], dtype=np.float32)}
    for camera in ("camera1", "camera2"):
        key = f"observation.images.{camera}"
        image = obs[key].detach().cpu().numpy()
        if image.shape != (1, 3, 256, 256) or not np.isfinite(image).all():
            raise ValueError(f"Expected finite {key} tensor (1,3,256,256); got {image.shape}")
        if image.min() < 0 or image.max() > 1:
            raise ValueError(f"{key} must be normalized to [0,1]")
        features[key] = np.ascontiguousarray(np.rint(image[0].transpose(1, 2, 0) * 255), dtype=np.uint8)
    return features


def action_array(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim != 2 or array.shape[1] != 7 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite canonical action chunk (steps,7); got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def camera_frame(features: dict[str, np.ndarray]) -> np.ndarray:
    """joinig the two procrssed rgb view into one contiguous video."""
    camera1 = features["observation.images.camera1"]
    camera2 = features["observation.images.camera2"]
    if camera1.shape != (256, 256, 3) or camera2.shape != (256, 256, 3):
        raise ValueError("Video cameras must both have shape (256,256,3)")
    return np.ascontiguousarray(np.concatenate((camera1, camera2), axis=1), dtype=np.uint8)


class Backend:
    """running all the simulator and transport calls on one worker thread."""

    def __init__(self, targets: dict[str, Target] | None = None) -> None:
        self.targets = targets or {
            JETSON_TARGET_NAME: Target(),
            PC2_TARGET_NAME: Target(
                endpoint="tcp/127.0.0.1:17447",
                deployment="smolvla_libero",
                semantics="libero-relative-eef-v1",
                zenoh_mode="client",
            ),
        }
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-owner")
        self.env: Any = None
        self.processor: Any = None
        self.raw: Any = None
        self.task = ""
        self.features: dict[str, np.ndarray] | None = None
        self.capture_time = 0.0
        self.observation_id = ""
        self.task_id: int | None = None
        self.chunk: np.ndarray | None = None
        self.action_history: list[np.ndarray] = []
        self.action_chunks: list[np.ndarray] = []
        self.chunk_videos: list[str] = []
        self.latency: float | None = None
        self.server_durations = "No remote inference yet"
        self.video: str | None = None
        self.last_inference_target: Target | None = None
        self.selected_target_name = next(iter(self.targets))
        self.status = "No live data yet. Click Start to initialize the simulator."

    def dispatch(
        self, operation: str, task_id: int, target: str, max_chunks: int | None = None
    ) -> dict[str, Any]:
        return self.worker.submit(self._dispatch, operation, task_id, target, max_chunks).result()

    def _dispatch(self, operation: str, task_id: int, target: str, max_chunks: int | None) -> dict[str, Any]:
        try:
            if operation == "reset":
                self.features, self.raw, self.chunk, self.latency, self.video = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
                self.last_inference_target = None
                self._reset(task_id)
                self._capture()
                self.status = (
                    "LIVE simulator reset and observation captured. Inference target not contacted yet."
                )
            elif operation == "capture":
                self._check_task(task_id)
                self._capture()
                self.chunk, self.latency, self.video, self.last_inference_target = None, None, None, None
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
                self.status = (
                    "LIVE observation captured from current simulator state; simulator did not advance."
                )
            elif operation == "infer":
                self.selected_target_name = target
                self.chunk, self.latency, self.video, self.last_inference_target = None, None, None, None
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
                self._check_task(task_id)
                self._infer(self.targets[target])
            elif operation == "infer_execute":
                self.selected_target_name = target
                self.chunk, self.latency, self.video, self.last_inference_target = None, None, None, None
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
                self._check_task(task_id)
                self._infer(self.targets[target])
                self._execute()
            elif operation == "execute":
                self._check_task(task_id)
                self._execute()
            elif operation == "rollout":
                self.selected_target_name = target
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
                self._check_task(task_id)
                self._rollout(self.targets[target], max_chunks)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as exc:
            if operation in {"infer", "infer_execute"}:
                self.chunk, self.latency = None, None
                self.action_history = []
                self.action_chunks = []
                self.chunk_videos = []
            if operation == "execute":
                self.video = None
            self.status = f"ERROR [{operation}] {type(exc).__name__}: {exc}"
            if operation == "infer":
                self.status += "\nNo action executed. Check the existing SSH tunnel, server readiness, deployment, and Zenoh mode."
            elif operation == "infer_execute":
                self.status += (
                    "\nInference/execution stopped; no action is executed after an inference failure."
                )
            elif operation == "execute":
                self.status += "\nRecorded rollout stopped. Click Start before requesting another inference."
            elif operation == "rollout":
                self.status += "\nClosed-loop rollout stopped; no task success is claimed."
        return self.view()

    def _check_task(self, task_id: int) -> None:
        if self.raw is None or self.task_id != task_id:
            raise ValueError("Reset the selected task before capturing or running inference.")

    def _reset(self, task_id: int) -> None:
        from lerobot.envs import make_env
        from lerobot.envs.configs import LiberoEnv

        if task_id not in range(10):
            raise ValueError("libero_object task ID must be 0–9")
        if self.env is not None:
            self.env.close()
            self.env = None
        cfg = LiberoEnv(
            task="libero_object",
            task_ids=[task_id],
            control_mode="relative",
            observation_height=256,
            observation_width=256,
            camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"},
        )
        self.env = make_env(cfg, n_envs=1, use_async_envs=False)["libero_object"][task_id]
        self.processor, _ = cfg.get_env_processors()
        self.raw, _ = self.env.reset(seed=[42])
        self.task = list(self.env.call("task_description"))[0]
        self.task_id = task_id

    def _capture(self) -> None:
        from lerobot.envs import preprocess_observation

        if self.raw is None:
            raise ValueError("Reset the simulator first")
        obs = preprocess_observation(self.raw)
        obs["task"] = [self.task]
        self.features = format_observation(self.processor(obs))
        self.capture_time = time.monotonic()
        self.observation_id = uuid.uuid4().hex

    def _infer(self, target: Target) -> None:
        from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
        from lerobot.inference.execution import QueueSnapshot
        from lerobot.remote_inference.client import RemoteClient

        if self.features is None:
            raise ValueError("Capture a real observation first")
        config = RemoteInferenceConfig(
            endpoint=target.endpoint,
            deployment=target.deployment,
            semantics=target.semantics,
            zenoh_mode=target.zenoh_mode,
            mode="chunk",
            chunk_merge="append",
            handshake_timeout_s=15,
            action_timeout_s=30,
        )
        client = None
        try:
            client = RemoteClient.connect(config)
            caps = client.capabilities
            actual = self.features
            if set(actual) != {f.name for f in caps.features}:
                raise ValueError("Server camera/state names differ from captured LIBERO features")
            for spec in caps.features:
                value = actual[spec.name]
                if tuple(spec.shape) != value.shape or spec.dtype != value.dtype.name:
                    raise ValueError(f"Server schema differs for {spec.name}: {spec}")
                expected_kind = "rgb" if "images" in spec.name else "tensor"
                if expected_kind == "rgb" and spec.kind != "rgb":
                    raise ValueError(f"Server must declare {spec.name} as rgb")
            if tuple(caps.action_feature.shape) != (7,):
                raise ValueError("Server does not advertise seven action dimensions")
            client.admit(
                features=caps.features,
                action_feature=caps.action_feature,
                semantics=target.semantics,
                action_interval=0.05,
                mode="chunk",
            )
            observation = ObservationSnapshot(
                features=actual,
                capture_time=self.capture_time,
                task=self.task,
                observation_id=self.observation_id,
            )
            request = ChunkRequest(
                observation=observation,
                continuation=QueueSnapshot(
                    generation=0, cursor=0, index=0, model_actions=None, canonical_actions=None, provenance=()
                ),
                request_id=uuid.uuid4().hex,
                generation=0,
                mode=ExecutionMode.CHUNK,
                delay=0,
                submitted_at=time.monotonic(),
            )
            start = time.perf_counter()
            result = client.infer(request)
            elapsed = (time.perf_counter() - start) * 1000
            self.chunk = action_array(result.canonical_actions)
            self.action_history = [self.chunk[0].copy()]
            self.action_chunks = [self.chunk.copy()]
            self.latency = elapsed
            self.server_durations = str(result.server_durations)
            self.last_inference_target = target
            self.status = (
                f"LIVE remote inference succeeded: {target.deployment} at {target.endpoint}\n"
                f"Observation {self.observation_id}; chunk {self.chunk.shape}; first action shown.\n"
                f"Artifact: {client.artifact_identity}\nServer durations (seconds): {result.server_durations}\n"
                "No simulator action executed. This is not task-success evidence."
            )
        finally:
            if client is not None:
                client.close()

    def _execute(self) -> None:
        """executing one returned action chunk and recording both camera views."""
        if self.env is None or self.raw is None or self.features is None:
            raise ValueError("Click Start and run inference before execution")
        if self.chunk is None:
            raise ValueError("Run Inference successfully before execution")
        if (
            self.last_inference_target is None
            or self.last_inference_target.semantics != "libero-relative-eef-v1"
        ):
            raise ValueError(
                "Execution requires compatible libero-relative-eef-v1 output; "
                "the selected target remains inference-only"
            )

        frames = [camera_frame(self.features)]
        executed, last_reward, terminated, truncated, success = self._execute_chunk(frames)
        video_path = self._write_video(frames, "chunk")
        self.chunk_videos = [str(video_path)]
        self.status = (
            f"LIVE simulator rollout recorded from the returned canonical chunk: {executed} steps at 20 Hz.\n"
            f"Final reward={last_reward:g}; terminated={terminated}; truncated={truncated}; success={success}.\n"
            f"Video: {video_path}\n"
            "This is simulator execution evidence; inspect the video before making task-success claims."
        )

    def _execute_chunk(self, frames: list[np.ndarray]) -> tuple[int, float, bool, bool, bool]:
        """executing the current action chunk and adding the two camera frames."""
        from lerobot.envs import preprocess_observation

        if self.chunk is None:
            raise ValueError("No inferred action chunk is available")
        executed = 0
        last_reward = 0.0
        terminated = False
        truncated = False
        success = False
        for action in self.chunk:
            self.raw, reward, term, trunc, info = self.env.step(action[np.newaxis, :])
            executed += 1
            last_reward = float(np.asarray(reward).reshape(-1)[0])
            terminated = bool(np.asarray(term).reshape(-1)[0])
            truncated = bool(np.asarray(trunc).reshape(-1)[0])
            if isinstance(info, dict):
                success_values = info.get("is_success", False)
                success = bool(np.asarray(success_values).reshape(-1)[0])

            obs = preprocess_observation(self.raw)
            obs["task"] = [self.task]
            step_features = format_observation(self.processor(obs))
            frames.append(camera_frame(step_features))
            if terminated or truncated or success:
                break
        self._capture()
        return executed, last_reward, terminated, truncated, success

    def _write_video(self, frames: list[np.ndarray], prefix: str, *, set_current: bool = True) -> Path:
        from lerobot.utils.io_utils import write_video

        output_dir = Path(__file__).resolve().parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{prefix}_{self.task_id}_{uuid.uuid4().hex}.mp4"
        write_video(video_path, frames, fps=20)
        if set_current:
            self.video = str(video_path)
        return video_path

    def _rollout(self, target: Target, max_chunks: int | None) -> None:
        """running repeated inference and execution and making one full video."""
        if self.env is None or self.raw is None or self.features is None:
            raise ValueError("Click Start before running a full task")
        if target.semantics != "libero-relative-eef-v1":
            raise ValueError("Full execution requires compatible libero-relative-eef-v1 semantics")
        if type(max_chunks) is not int or not 1 <= max_chunks <= 10:
            raise ValueError("Maximum chunks must be an integer from 1 to 10")

        frames: list[np.ndarray] = []
        total_steps = 0
        completed_chunks = 0
        last_reward = 0.0
        terminated = truncated = success = False
        latencies: list[float] = []
        rollout_actions: list[np.ndarray] = []
        rollout_chunks: list[np.ndarray] = []
        chunk_videos: list[str] = []
        for chunk_index in range(max_chunks):
            self._infer(target)
            if self.chunk is not None:
                rollout_actions.append(self.chunk[0].copy())
                rollout_chunks.append(self.chunk.copy())
            if self.latency is not None:
                latencies.append(self.latency)
            chunk_frames = [camera_frame(self.features)]
            executed, last_reward, terminated, truncated, success = self._execute_chunk(chunk_frames)
            frames.extend(chunk_frames if chunk_index == 0 else chunk_frames[1:])
            chunk_video = self._write_video(
                chunk_frames,
                f"rollout_chunk_{chunk_index + 1}",
                set_current=False,
            )
            chunk_videos.append(str(chunk_video))
            total_steps += executed
            completed_chunks += 1
            if terminated or truncated or success:
                break

        self.action_history = rollout_actions
        self.action_chunks = rollout_chunks
        self.chunk_videos = chunk_videos
        video_path = self._write_video(frames, "full_rollout")
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        self.status = (
            f"LIVE closed-loop rollout recorded: {completed_chunks} inference chunks, "
            f"{total_steps} simulator steps at 20 Hz.\n"
            f"Mean inference round trip={mean_latency:.2f} ms; final reward={last_reward:g}; "
            f"terminated={terminated}; truncated={truncated}; success={success}.\n"
            f"Video: {video_path}\n"
            + (
                "LIBERO reported task success."
                if success
                else "Stopped at the configured limit or environment termination; no task success is claimed."
            )
        )

    def view(self) -> dict[str, Any]:
        features = self.features or {}
        return {
            "camera1": features.get("observation.images.camera1"),
            "camera2": features.get("observation.images.camera2"),
            "state": features["observation.state"].tolist() if features else None,
            "task": self.task if features else "No live task captured",
            "action": self.chunk[0].tolist() if self.chunk is not None else None,
            "action_history": [action.tolist() for action in self.action_history],
            "action_chunks": [chunk.tolist() for chunk in self.action_chunks],
            "chunk_videos": self.chunk_videos.copy(),
            "latency": self.latency,
            "server_durations": self.server_durations,
            "video": self.video,
            "status": self.status,
            "selected_target_name": self.selected_target_name,
        }

    def snapshot(self) -> dict[str, Any]:
        """reading the current dashboard data from the simulator worker."""
        return self.worker.submit(self.view).result()

    def close(self) -> None:
        def cleanup() -> None:
            if self.env is not None:
                self.env.close()
                self.env = None

        self.worker.submit(cleanup).result()
        self.worker.shutdown()
