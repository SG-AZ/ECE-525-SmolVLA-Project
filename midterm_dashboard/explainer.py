"""explaining one real smolvla inference in an interactive page."""

from __future__ import annotations

import html
from functools import partial
from typing import Any

from .backend import Backend

TASKS = {
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

STAGES = ("Language", "Vision", "Robot state", "VLM fusion", "Action expert", "Actions")

CSS = """
footer{display:none!important}
.vla-board{display:grid;grid-template-columns:1.25fr 70px 1fr 70px 1.05fr 70px 1fr;gap:10px;
align-items:center;padding:24px;border-radius:18px;background:linear-gradient(135deg,#f8fafc,#eef2ff);color:#172033;
overflow-x:auto;min-height:390px}.vla-inputs{display:grid;gap:14px}.vla-card{padding:15px;border:2px solid #475569;
color:#f8fafc!important;border-radius:12px;background:#1e293b;box-shadow:0 3px 10px rgba(15,23,42,.2)}
.vla-card *{color:#f8fafc!important}.vla-card code{color:#bae6fd!important;background:#0f172a!important;padding:5px 7px;border-radius:6px}
.vla-card small{color:#cbd5e1!important}.vla-card.active{border-color:#a78bfa;box-shadow:0 0 0 4px rgba(124,58,237,.2)}
.vla-card small,.vla-card code{display:block;margin-top:6px}
.vla-arrow{text-align:center;font-size:30px;color:#64748b}.vla-block{padding:30px 16px;border-radius:18px;text-align:center;
font-size:1.1rem;font-weight:800;background:#8096c8;color:#111827!important;min-width:145px}.vla-block *{color:#111827!important}.vla-block.policy{background:#36b718}
.vla-block.actions{background:#c7d2fe}.vla-block.active{outline:5px solid rgba(124,58,237,.3);transform:scale(1.03)}
.plus{display:inline-flex;width:23px;height:23px;border-radius:50%;align-items:center;justify-content:center;background:#ede9fe;
color:#6d28d9;font-weight:900}.detail{padding:22px;border:1px solid #334155;border-radius:14px;
background:linear-gradient(145deg,rgba(30,41,59,.9),rgba(15,23,42,.9));margin-top:14px}.detail h3{margin:10px 0}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;font-size:.72rem;font-weight:800}.live{color:#86efac;
background:rgba(22,101,52,.55)}.structure{color:#93c5fd;background:rgba(30,64,175,.45)}.concept{color:#fde68a;
background:rgba(146,64,14,.5)}.transform{display:flex;align-items:center;justify-content:center;gap:15px;flex-wrap:wrap;
margin:20px 0}.transform>div,.transform code{padding:14px;border:1px solid #475569;border-radius:10px;background:#0f172a}
.vector{display:inline-grid;grid-auto-flow:column;gap:2px;vertical-align:middle}.vector i{display:block;width:8px;height:34px;
background:linear-gradient(#60a5fa,#f472b6);opacity:.8}.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.metric-grid>div{padding:14px;text-align:center;border-radius:10px;background:#0f172a}.metric-grid strong,.metric-grid small{display:block}
.metric-grid strong{font-size:1.25rem;color:#7dd3fc}@media(max-width:900px){.vla-board{grid-template-columns:1fr}.vla-arrow{transform:rotate(90deg)}
.metric-grid{grid-template-columns:1fr}}
.flow-canvas{overflow-x:auto;padding:10px;border-radius:18px;background:#f8fafc;border:1px solid #dbeafe}
.flow-canvas svg{display:block;min-width:1180px;width:100%;height:auto}.flow-canvas text{font-family:Inter,Arial,sans-serif}
"""


def snapshot(backend: Backend) -> dict[str, Any]:
    view = backend.snapshot()
    target_name = view["selected_target_name"]
    target = backend.targets[target_name]
    camera1 = view["camera1"]
    return {
        "live": camera1 is not None,
        "target": target_name,
        "endpoint": target.endpoint,
        "deployment": target.deployment,
        "semantics": target.semantics,
        "language_enabled": target.language_enabled,
        "camera_shape": list(camera1.shape) if camera1 is not None else None,
        "camera_dtype": str(camera1.dtype) if camera1 is not None else None,
        "state": view["state"],
        "task": view["task"],
        "chunks": view["action_chunks"],
        "latency": view["latency"],
        "server_durations": view["server_durations"],
        "status": view["status"],
    }


def vector_icon(size: int = 8) -> str:
    return '<span class="vector">' + "".join("<i></i>" for _ in range(size)) + "</span>"


def architecture(data: dict[str, Any], active: str = "Overview") -> str:
    task = html.escape(str(data.get("task", "No observation yet")))
    camera = f"{data.get('camera_dtype')} {data.get('camera_shape')}" if data.get("live") else "not captured"
    state = "float32 [8]" if data.get("state") is not None else "not captured"
    chunks = data.get("chunks", [])
    first_action = chunks[0][0] if chunks else [0.0] * 7

    def opacity(*stages: str) -> str:
        return "1" if active == "Overview" or active in stages else ".14"

    def border(stage: str) -> str:
        return "#7c3aed" if active == stage else "#94a3b8"

    names = ("Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "grip")
    colors = ("#2563eb", "#3b82f6", "#06b6d4", "#8b5cf6", "#a855f7", "#ec4899", "#f97316")
    outputs = []
    for index, (name, color, value) in enumerate(zip(names, colors, first_action, strict=True)):
        y = 112 + index * 55
        outputs.append(
            f'<path d="M940 275 C990 275 995 {y} 1035 {y}" fill="none" stroke="{color}" '
            f'stroke-width="8" opacity="{opacity("Actions")}"/>'
            f'<rect x="1035" y="{y - 17}" width="135" height="34" rx="7" fill="white" stroke="{color}"/>'
            f'<text x="1048" y="{y + 5}" font-size="14" fill="#172033">{name}</text>'
            f'<text x="1158" y="{y + 5}" text-anchor="end" font-size="12" fill="#475569">{value:+.3f}</text>'
        )

    return f"""
    <div class="flow-canvas"><svg viewBox="0 0 1190 520" role="img" aria-label="SmolVLA computational flow">
      <defs>
        <linearGradient id="vlm" x1="0" x2="1"><stop stop-color="#c4b5fd"/><stop offset="1" stop-color="#93c5fd"/></linearGradient>
        <linearGradient id="policy" x1="0" x2="1"><stop stop-color="#86efac"/><stop offset="1" stop-color="#22c55e"/></linearGradient>
      </defs>
      <text x="20" y="30" font-size="17" font-weight="700" fill="#64748b">Robot observation</text>
      <text x="250" y="30" font-size="17" font-weight="700" fill="#64748b">Input representations</text>
      <text x="500" y="30" font-size="17" font-weight="700" fill="#64748b">Multimodal context</text>
      <text x="750" y="30" font-size="17" font-weight="700" fill="#64748b">Action generation</text>
      <text x="1025" y="30" font-size="17" font-weight="700" fill="#64748b">Action dimensions</text>

      <g opacity="{opacity("Language")}">
        <rect x="20" y="55" width="195" height="100" rx="12" fill="white" stroke="{border("Language")}" stroke-width="3"/>
        <text x="35" y="82" font-size="15" font-weight="700" fill="#172033">Language instruction  ⊕</text>
        <foreignObject x="35" y="91" width="165" height="40"><div xmlns="http://www.w3.org/1999/xhtml" style="font:12px Arial;color:#475569;line-height:1.25">{task}</div></foreignObject>
        <text x="35" y="143" font-size="12" fill="#64748b">request text</text>
        <path d="M215 105 C235 105 235 105 255 105" stroke="#8b5cf6" stroke-width="10" fill="none"/>
        <rect x="255" y="67" width="145" height="76" rx="10" fill="#f5f3ff" stroke="#8b5cf6"/>
        <text x="327" y="98" text-anchor="middle" font-size="14" fill="#172033">Tokenizer</text>
        <text x="327" y="122" text-anchor="middle" font-size="13" fill="#6d28d9">token embeddings</text>
      </g>

      <g opacity="{opacity("Vision")}">
        <rect x="20" y="205" width="195" height="100" rx="12" fill="white" stroke="{border("Vision")}" stroke-width="3"/>
        <text x="35" y="235" font-size="15" font-weight="700" fill="#172033">Two camera images  ⊕</text>
        <text x="35" y="262" font-size="13" fill="#475569">agent view + wrist view</text>
        <text x="35" y="287" font-size="12" fill="#64748b">{camera}</text>
        <path d="M215 255 C235 255 235 255 255 255" stroke="#2563eb" stroke-width="18" fill="none"/>
        <rect x="255" y="213" width="145" height="84" rx="10" fill="#eff6ff" stroke="#2563eb"/>
        <text x="327" y="247" text-anchor="middle" font-size="14" fill="#172033">Vision encoder</text>
        <text x="327" y="273" text-anchor="middle" font-size="13" fill="#1d4ed8">image embeddings</text>
      </g>

      <g opacity="{opacity("Robot state")}">
        <rect x="20" y="355" width="195" height="100" rx="12" fill="white" stroke="{border("Robot state")}" stroke-width="3"/>
        <text x="35" y="385" font-size="15" font-weight="700" fill="#172033">Robot state  ⊕</text>
        <text x="35" y="412" font-size="13" fill="#475569">position + rotation + gripper</text>
        <text x="35" y="437" font-size="12" fill="#64748b">{state}</text>
        <path d="M215 405 C235 405 235 405 255 405" stroke="#10b981" stroke-width="12" fill="none"/>
        <rect x="255" y="367" width="145" height="76" rx="10" fill="#ecfdf5" stroke="#10b981"/>
        <text x="327" y="398" text-anchor="middle" font-size="14" fill="#172033">Linear projection</text>
        <text x="327" y="422" text-anchor="middle" font-size="13" fill="#047857">state embedding</text>
      </g>

      <g opacity="{opacity("Language", "Vision", "Robot state", "VLM fusion")}">
        <path d="M400 105 C455 105 445 230 495 230" stroke="#8b5cf6" stroke-width="10" fill="none" opacity=".65"/>
        <path d="M400 255 C450 255 450 255 495 255" stroke="#2563eb" stroke-width="18" fill="none" opacity=".65"/>
        <path d="M400 405 C455 405 445 280 495 280" stroke="#10b981" stroke-width="12" fill="none" opacity=".65"/>
        <rect x="495" y="135" width="185" height="240" rx="22" fill="url(#vlm)" stroke="{border("VLM fusion")}" stroke-width="3"/>
        <rect x="510" y="158" width="155" height="38" rx="8" fill="white" opacity=".6"/>
        <rect x="510" y="207" width="155" height="38" rx="8" fill="white" opacity=".45"/>
        <rect x="510" y="256" width="155" height="38" rx="8" fill="white" opacity=".32"/>
        <rect x="510" y="305" width="155" height="38" rx="8" fill="white" opacity=".22"/>
        <text x="587" y="181" text-anchor="middle" font-size="15" font-weight="700" fill="#172033">SmolVLM layers  ⊕</text>
        <text x="587" y="230" text-anchor="middle" font-size="13" fill="#334155">multimodal attention</text>
        <text x="587" y="279" text-anchor="middle" font-size="13" fill="#334155">context mixing</text>
        <text x="587" y="328" text-anchor="middle" font-size="13" fill="#334155">context representation</text>
      </g>

      <g opacity="{opacity("VLM fusion", "Action expert")}">
        <path d="M680 255 C710 255 710 255 740 255" stroke="#6366f1" stroke-width="22" fill="none" opacity=".5"/>
        <rect x="740" y="125" width="200" height="260" rx="24" fill="url(#policy)" stroke="{border("Action expert")}" stroke-width="3"/>
        <text x="840" y="160" text-anchor="middle" font-size="16" font-weight="700" fill="#172033">Flow-matching expert  ⊕</text>
        <path d="M775 205 C810 170 870 240 905 205" stroke="white" stroke-width="5" fill="none" opacity=".8"/>
        <path d="M775 260 C810 225 870 295 905 260" stroke="white" stroke-width="5" fill="none" opacity=".8"/>
        <path d="M775 315 C810 280 870 350 905 315" stroke="white" stroke-width="5" fill="none" opacity=".8"/>
        <text x="840" y="360" text-anchor="middle" font-size="13" fill="#14532d">iterative trajectory refinement</text>
      </g>

      <g opacity="{opacity("Action expert", "Actions")}">
        <path d="M940 255 C975 255 985 275 1010 275" stroke="#f97316" stroke-width="24" fill="none" opacity=".45"/>
        {"".join(outputs)}
      </g>
      <text x="20" y="500" font-size="12" fill="#64748b">Inputs and final actions are live. Internal paths show verified structure; hidden representation values are not returned.</text>
    </svg></div>
    """


def explanation(data: dict[str, Any], stage: str) -> str:
    live = '<span class="badge live">LIVE DATA</span>'
    structure = '<span class="badge structure">MODEL STRUCTURE</span>'
    concept = '<span class="badge concept">CONCEPTUAL</span>'
    if not data["live"]:
        return '<div class="detail"><h3>Run an instruction first</h3><p>Select a task and inference target above. The page will capture a real robot observation and send it to SmolVLA.</p></div>'
    chunks = data["chunks"]
    state_values = html.escape(str(data["state"]))
    language_status = (
        "enabled: the server may tokenize the request instruction"
        if data["language_enabled"]
        else "disabled on this server: the instruction is carried in the request but is not claimed as a model input"
    )
    cards = {
        "Overview": f'<div class="detail">{structure}<h3>One VLA prediction</h3><p>The left side contains the robot observation. SmolVLM builds multimodal context, the action expert generates a trajectory, and the output is a sequence of robot controls. Click any <b>+</b> stage button below the diagram.</p></div>',
        "Language": f'<div class="detail">{live}<h3>Language instruction</h3><p><b>{html.escape(data["task"])}</b></p><div class="transform"><div>Instruction string</div><b>→ tokenizer →</b>{vector_icon()}<div>token embeddings</div></div><p><b>Current deployment:</b> language is {language_status}. Token IDs and embeddings are not returned by the remote protocol, so the transformation graphic is structural.</p></div>',
        "Vision": f'<div class="detail">{live}<h3>Vision input</h3><div class="transform"><div>2 × {data["camera_dtype"]}<br>{data["camera_shape"]}</div><b>→ vision encoder →</b>{vector_icon(12)}<div>image embeddings</div></div><p>The images and their wire types are live. Image embeddings are model structure; their values are not returned by the server.</p></div>',
        "Robot state": f'<div class="detail">{live}<h3>Robot-state input</h3><p><code>{state_values}</code></p><div class="transform"><div>float32 [8]</div><b>→ linear projection →</b>{vector_icon()}<div>state embedding</div></div><p>The eight live values are end-effector position (3), axis-angle rotation (3), and two gripper positions.</p></div>',
        "VLM fusion": f'<div class="detail">{structure}<h3>Multimodal context</h3><div class="transform"><div>image embeddings</div><span>+</span><div>language embeddings*</div><span>+</span><div>state embedding</div><b>→</b><div>context tokens</div></div><p>SmolVLA combines the encoded modalities before action generation. *Language depends on server configuration. Hidden context tensors are not exposed.</p></div>',
        "Action expert": f'<div class="detail">{concept}<h3>Flow-matching action expert</h3><div class="transform"><div>noisy candidate trajectory</div><b>→ repeated refinement →</b><div>structured action trajectory</div></div><p>This explains the algorithm, but it is not a live intermediate trace. The server returns only the completed action chunk.</p></div>',
        "Actions": f'<div class="detail">{live}<h3>Canonical action output</h3><div class="metric-grid"><div><strong>{len(chunks)}</strong><small>returned chunks</small></div><div><strong>{len(chunks[0]) if chunks else 0} × 7</strong><small>shape per chunk</small></div><div><strong>{data["latency"] or "—"} ms</strong><small>round trip</small></div></div><p>Each row is Δx, Δy, Δz, three rotation deltas, and gripper control. Inspect every value below.</p></div>',
    }
    return cards.get(stage, cards["Overview"])


def action_bar(action: list[float]) -> str:
    scale = max(max(abs(value) for value in action), 1e-6)
    rows = []
    for index, value in enumerate(action):
        width = abs(value) / scale * 45
        left = 50 - width if value < 0 else 50
        rows.append(
            f'<div style="display:flex;gap:12px;margin:8px"><span>a{index}: {value:.5f}</span>'
            f'<div style="position:relative;width:60%;height:20px;background:#eee">'
            f'<div style="position:absolute;left:50%;height:20px;border-left:1px solid #777"></div>'
            f'<div style="position:absolute;left:{left}%;width:{width}%;height:20px;'
            f'background:{"#d66" if value < 0 else "#368ac0"}"></div></div></div>'
        )
    return "".join(rows)


def action_visuals(chunks: list[list[list[float]]]) -> str:
    if not chunks:
        return "<p>No action tensor returned yet.</p>"
    labels = ("Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "grip")
    colors = ("#4cc9f0", "#4895ef", "#4361ee", "#7209b7", "#b5179e", "#f72585", "#f8961e")
    output = ["<h3>Live action trajectories</h3>"]
    for number, chunk in enumerate(chunks, 1):
        width, height, pad = 760, 230, 28
        scale = max(abs(value) for row in chunk for value in row) or 1.0
        paths = []
        for dimension, color in enumerate(colors):
            points = []
            for step, action in enumerate(chunk):
                x = pad + step * (width - 2 * pad) / max(len(chunk) - 1, 1)
                y = height / 2 - action[dimension] / scale * (height / 2 - pad)
                points.append(f"{x:.1f},{y:.1f}")
            paths.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        legend = " ".join(
            f'<span style="color:{color};margin-right:12px">● {label}</span>'
            for label, color in zip(labels, colors, strict=True)
        )
        output.append(
            f'<h4>Chunk {number}</h4><div>{legend}</div><svg viewBox="0 0 {width} {height}" style="width:100%;background:#111827;border-radius:8px"><line x1="{pad}" y1="{height / 2}" x2="{width - pad}" y2="{height / 2}" stroke="#64748b" stroke-dasharray="4"/>{"".join(paths)}</svg>'
        )
        output.append(
            f'<h4>Signed heatmap: {len(chunk)} × 7</h4><div style="overflow-x:auto"><table style="border-collapse:collapse">'
        )
        for dimension, label in enumerate(labels):
            output.append(f'<tr><th style="padding-right:8px;text-align:right">{label}</th>')
            for step, action in enumerate(chunk):
                value = action[dimension]
                rgb = "54,138,192" if value >= 0 else "214,102,102"
                alpha = 0.15 + 0.85 * min(abs(value) / scale, 1.0)
                output.append(
                    f'<td title="step {step}, {label}={value:.6f}" style="width:11px;min-width:11px;height:18px;background:rgba({rgb},{alpha:.3f});border:1px solid rgba(120,120,120,.2)"></td>'
                )
            output.append("</tr>")
        output.append("</table></div>")
    return "".join(output)


def run_inference(backend: Backend, task_id: int, target_name: str) -> tuple[Any, ...]:
    backend.dispatch("reset", int(task_id), target_name)
    view = backend.dispatch("infer", int(task_id), target_name)
    data = snapshot(backend)
    return (
        view["task"],
        view["camera1"],
        view["camera2"],
        view["state"],
        architecture(data),
        explanation(data, "Overview"),
        action_visuals(data["chunks"]),
        view["status"],
    )


def select_stage(backend: Backend, stage: str) -> tuple[str, str]:
    data = snapshot(backend)
    return architecture(data, stage), explanation(data, stage)


def show_action(backend: Backend, chunk_number: int, step: int) -> str:
    chunks = snapshot(backend)["chunks"]
    chunk_index, step_index = int(chunk_number) - 1, int(step)
    if chunk_index not in range(len(chunks)) or step_index not in range(len(chunks[chunk_index])):
        return "<p>Run inference, then choose an available chunk and action step.</p>"
    return f"<h3>Chunk {chunk_number}, action step {step_index}</h3>" + action_bar(
        chunks[chunk_index][step_index]
    )


def build_explainer(backend: Backend) -> Any:
    import gradio as gr

    empty = {"live": False, "task": "No observation yet", "state": None}
    with gr.Blocks(title="Interactive VLA Explainer", css=CSS) as page:
        gr.Markdown(
            "# Interactive VLA Explainer\n"
            "[← Run Task dashboard](../) · Choose an instruction and follow one actual SmolVLA request. "
            "LIBERO is used only to provide a repeatable robot observation."
        )
        with gr.Row():
            task = gr.Dropdown(
                choices=[(f"Task {index}: {text}", index) for index, text in TASKS.items()],
                value=0,
                label="Instruction",
            )
            target = gr.Dropdown(
                choices=list(backend.targets), value=next(iter(backend.targets)), label="Inference target"
            )
            run = gr.Button("Run SmolVLA inference", variant="primary")
        instruction = gr.Textbox(value=TASKS[0], label="Live instruction", interactive=False)
        with gr.Row():
            camera1 = gr.Image(label="Vision input: agent view", interactive=False)
            camera2 = gr.Image(label="Vision input: wrist view", interactive=False)
            state = gr.JSON(label="Robot-state input")
        gr.Markdown("### Click a + stage to inspect its transformation")
        with gr.Row():
            stage_buttons = [gr.Button(f"{stage}  +") for stage in STAGES]
        board = gr.HTML(architecture(empty))
        detail = gr.HTML(explanation(empty, "Overview"))
        gr.Markdown("## Action output explorer")
        with gr.Row():
            chunk_number = gr.Dropdown(choices=list(range(1, 11)), value=1, label="Action chunk")
            action_step = gr.Slider(0, 49, value=0, step=1, label="Action step")
        selected_action = gr.HTML("<p>Run inference to inspect one action.</p>")
        visuals = gr.HTML("<p>No action tensor returned yet.</p>")
        status = gr.Textbox(label="Inference status", lines=5, interactive=False)
        task.change(
            lambda task_id: TASKS[int(task_id)], inputs=task, outputs=instruction, api_name="instruction"
        )
        run.click(
            partial(run_inference, backend),
            inputs=[task, target],
            outputs=[instruction, camera1, camera2, state, board, detail, visuals, status],
            concurrency_id="simulator",
            concurrency_limit=1,
            api_name="run_explainer_inference",
        )
        for button, stage in zip(stage_buttons, STAGES, strict=True):
            button.click(
                partial(select_stage, backend, stage),
                outputs=[board, detail],
                api_name=f"stage_{stage.lower().replace(' ', '_')}",
            )
        for control in (chunk_number, action_step):
            control.change(
                partial(show_action, backend),
                inputs=[chunk_number, action_step],
                outputs=selected_action,
                api_name=False,
            )
    return page.queue(max_size=8, default_concurrency_limit=1)
