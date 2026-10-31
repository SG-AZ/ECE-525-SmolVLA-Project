"""running the task dashboard and the smolvla explainer."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from .backend import Backend, Target

TASK_DESCRIPTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}

APP_CSS = """
footer { display: none !important; }
.gradio-container {
    width: calc(100% - 48px) !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 24px 32px 48px !important;
    box-sizing: border-box !important;
}
.chunk-action-pair {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
}
.chunk-action-pair .action-row {
    grid-template-columns: 76px 94px minmax(90px, 1fr) !important;
    gap: 8px !important;
}
.chunk-result-row {
    display: grid !important;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr) !important;
    align-items: start !important;
    gap: 20px !important;
}
.chunk-result-row > .chunk-actions-column,
.chunk-result-row > .chunk-video-column,
.chunk-video {
    width: 100% !important;
    min-width: 0 !important;
}
.chunk-video video {
    width: 100% !important;
    min-height: 390px !important;
    max-height: 430px !important;
    object-fit: contain !important;
}
@media (max-width: 760px) {
    .gradio-container {
        width: calc(100% - 24px) !important;
        padding: 16px 12px 32px !important;
    }
    .chunk-action-pair { grid-template-columns: 1fr; }
    .chunk-result-row { grid-template-columns: 1fr !important; }
}
"""

ARM_ACTIONS = (
    ("ΔX", "translation", 0.05, "m"),
    ("ΔY", "translation", 0.05, "m"),
    ("ΔZ", "translation", 0.05, "m"),
    ("RotVec X", "axis-angle", 0.5, "rad"),
    ("RotVec Y", "axis-angle", 0.5, "rad"),
    ("RotVec Z", "axis-angle", 0.5, "rad"),
)


def selected_task_description(task_id: int) -> str:
    return TASK_DESCRIPTIONS[int(task_id)]


def task_selection(task_id: int) -> tuple[str, str]:
    return selected_task_description(task_id), ""


def action_chart(action: list[float] | None) -> str:
    if action is None:
        return "<p>No live action returned. Chart is empty.</p>"
    if len(action) != 7:
        return f"<p>Expected a 7D LIBERO action, received {len(action)} values.</p>"
    rows = []
    for (label, representation, output_scale, unit), value in zip(ARM_ACTIONS, action[:6], strict=True):
        width = min(abs(value), 1.0) * 50
        left = 50 - width if value < 0 else 50
        converted = value * output_scale
        rows.append(
            '<div class="action-row" style="display:grid;grid-template-columns:105px 120px minmax(220px,1fr);'
            f'gap:12px;align-items:center;margin:8px"><strong>{label}</strong>'
            f"<span>{value:+.5f}<br><small>{converted:+.4f} {unit}</small></span>"
            f'<div title="{representation}" style="position:relative;height:20px;background:#eee">'
            f'<div style="position:absolute;left:50%;height:20px;border-left:1px solid #777"></div>'
            f'<div style="position:absolute;left:{left}%;width:{width}%;height:20px;'
            f'background:{"#d66" if value < 0 else "#368ac0"}"></div></div></div>'
        )
    gripper = action[6]
    gripper_state = "OPEN" if gripper < 0 else "CLOSE" if gripper > 0 else "HOLD"
    gripper_color = "#d66" if gripper < 0 else "#368ac0" if gripper > 0 else "#777"
    return (
        "".join(rows)
        + '<div style="margin:16px 8px;padding:12px;border:1px solid #666;border-radius:8px">'
        + f'<strong>Gripper</strong>: <span style="color:{gripper_color};font-weight:800">'
        + f"{gripper_state}</span> ({gripper:+.5f})"
        + "</div>"
    )


def action_history_chart(actions: list[list[float]]) -> str:
    if not actions:
        return action_chart(None)
    return "".join(
        f"<h4>Inference chunk {index}</h4>{action_chart(action)}" for index, action in enumerate(actions, 1)
    )


def action_chunk_chart(chunk: list[list[float]]) -> str:
    """showing the first and last action from one inference chunk."""
    if not chunk:
        return action_chart(None)
    step_count = len(chunk)
    return (
        '<div class="chunk-action-pair">'
        f"<section><h4>First action — step 1 of {step_count}</h4>{action_chart(chunk[0])}</section>"
        f"<section><h4>Last action — step {step_count} of {step_count}</h4>"
        f"{action_chart(chunk[-1])}</section></div>"
    )


def handle(
    backend: Backend,
    operation: str,
    task_id: int,
    target: str,
    max_chunks: int | None = None,
) -> tuple[Any, ...]:
    import gradio as gr

    view = backend.dispatch(operation, int(task_id), target, max_chunks)
    chunk_outputs: list[Any] = []
    chunks, videos = view["action_chunks"], view["chunk_videos"]
    for index in range(10):
        visible = index < len(chunks)
        chunk_outputs.extend(
            [
                gr.update(value=f"### Inference chunk {index + 1}", visible=visible),
                gr.update(
                    value=action_chunk_chart(chunks[index]) if visible else action_chart(None),
                    visible=visible,
                ),
                gr.update(value=videos[index] if index < len(videos) else None, visible=visible),
            ]
        )
    return (
        view["camera1"],
        view["camera2"],
        view["state"],
        view["task"],
        *chunk_outputs,
        view["video"],
        view["latency"],
        view["status"],
    )


def build_app(backend: Backend) -> Any:
    """building the task dashboard page."""
    import gradio as gr

    with gr.Blocks(title="ECE 525 Dashboard", css=APP_CSS) as app:
        gr.Markdown("# ECE 525 Dashboard\nSmolVLA workstation")
        with gr.Row():
            task = gr.Dropdown(
                choices=[(f"libero_object task {index}", index) for index in range(10)],
                value=0,
                label="Task",
            )
            target = gr.Dropdown(
                choices=list(backend.targets), value=next(iter(backend.targets)), label="Inference target"
            )
        with gr.Row():
            selected_description = gr.Textbox(
                value=selected_task_description(0), label="Task description", interactive=False
            )
            instruction = gr.Textbox(value="", label="Task instruction", interactive=False)
        max_chunks = gr.Slider(1, 10, value=4, step=1, label="Maximum inference chunks")
        task.change(
            task_selection,
            inputs=task,
            outputs=[selected_description, instruction],
            api_name="task_description",
        )
        with gr.Row():
            reset = gr.Button("Start", variant="primary")
            infer_execute = gr.Button("Single Run & Record")
            rollout = gr.Button("Run Full Task & Record", variant="secondary")
        with gr.Row():
            camera1 = gr.Image(label="Camera 1", interactive=False)
            camera2 = gr.Image(label="Camera 2", interactive=False)
            state = gr.JSON(label="Robot state — 8D")
        gr.Markdown("### Actions and videos by inference chunk")
        gr.Markdown(
            "Actions use LIBERO's normalized **−1 to +1** controller convention. "
            "Translation and rotation commands are relative to the current end-effector pose."
        )
        with gr.Accordion("Action convention", open=False):
            gr.Markdown(
                "- **ΔX, ΔY, ΔZ:** relative translation; ±1 maps to ±0.05 m.\n"
                "- **RotVec X/Y/Z:** scaled axis-angle vector, not roll/pitch/yaw; "
                "±1 maps to ±0.5 rad per component.\n"
                "- **Gripper:** negative opens, positive closes, and zero holds."
            )
        chunk_outputs = []
        for chunk_number in range(1, 11):
            chunk_heading = gr.Markdown(f"### Inference chunk {chunk_number}", visible=False)
            with gr.Row(elem_classes="chunk-result-row"):
                with gr.Column(scale=2, min_width=0, elem_classes="chunk-actions-column"):
                    chart = gr.HTML(action_chart(None), visible=False)
                with gr.Column(scale=1, min_width=0, elem_classes="chunk-video-column"):
                    chunk_video = gr.Video(
                        label=f"Inference chunk {chunk_number} video",
                        interactive=False,
                        visible=False,
                        elem_classes="chunk-video",
                    )
            chunk_outputs.extend([chunk_heading, chart, chunk_video])
        gr.Markdown("### Full rollout video")
        video = gr.Video(label="Combined recorded rollout", interactive=False)
        latency = gr.Number(label="Inference latency (ms)", interactive=False)
        status = gr.Textbox(value=backend.status, label="Status / errors", lines=6, interactive=False)
        outputs = [camera1, camera2, state, instruction, *chunk_outputs, video, latency, status]
        for button, operation in ((reset, "reset"), (infer_execute, "infer_execute")):
            button.click(
                partial(handle, backend, operation),
                inputs=[task, target],
                outputs=outputs,
                concurrency_id="simulator",
                concurrency_limit=1,
                api_name=operation,
            )
        rollout.click(
            partial(handle, backend, "rollout"),
            inputs=[task, target, max_chunks],
            outputs=outputs,
            concurrency_id="simulator",
            concurrency_limit=1,
            api_name="rollout",
        )
    return app.queue(max_size=8, default_concurrency_limit=1)


def main() -> None:
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI

    from .explainer import build_explainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, help="JSON mapping of display names to connection fields")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    targets = None
    if args.targets:
        targets = {name: Target(**fields) for name, fields in json.loads(args.targets.read_text()).items()}
        if not targets:
            parser.error("targets must not be empty")
    backend = Backend(targets)
    site = FastAPI()
    site = gr.mount_gradio_app(site, build_explainer(backend), path="/how-smolvla-works")
    site = gr.mount_gradio_app(site, build_app(backend), path="/")
    try:
        uvicorn.run(site, host="127.0.0.1", port=args.port)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
