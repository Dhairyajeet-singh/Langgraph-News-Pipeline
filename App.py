"""
Flask frontend.

Routes:
  GET  /                  → main UI (form + log panel + draft preview)
  POST /run               → kicks off pipeline in a background thread
  GET  /logs/<job_id>     → Server-Sent Events stream of agent logs
  GET  /status/<job_id>   → JSON poll for status + draft (used by frontend)
  POST /resume/<job_id>   → HITL: user clicks Approve (with optional edits)

We run each pipeline in a background thread keyed by job_id, and pipe its
log lines into a per-job queue that SSE drains. This keeps the request/
response cycle simple while still giving a live UI.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

load_dotenv()  

from agent import tools
from agent.graph import build_graph
from agent.state import AgentState
from scraper import pipeline as scraper_pipeline

app = Flask(__name__)

JOBS: dict[str, dict] = {}


def _make_log_pipe(job_id: str):
    """Return a (level, msg) callback that pushes to this job's log queue."""
    q = JOBS[job_id]["log_queue"]
    def _cb(level: str, msg: str):
        q.put({"level": level, "msg": msg, "ts": time.strftime("%H:%M:%S")})
    return _cb


def _set_status(job_id: str, status: str, **extra):
    JOBS[job_id]["status"] = status
    JOBS[job_id].update(extra)
    JOBS[job_id]["log_queue"].put({"_status": status, **extra})


def _run_pipeline_thread(job_id: str, params: dict):
    """Background worker. Runs the graph and stores result in JOBS[job_id]."""
    try:
        log_cb = _make_log_pipe(job_id)
        tools.set_log_callback(log_cb)
        scraper_pipeline.set_log_callback(log_cb)

        mode = params["mode"]
        graph = build_graph(mode=mode)
        config = {"configurable": {"thread_id": job_id}}
        JOBS[job_id]["graph"] = graph
        JOBS[job_id]["config"] = config

        initial: AgentState = {
            "goal": params["goal"],
            "recipient_email": params["recipient_email"],
            "top_n": params["top_n"],
            "max_parallel": params["max_parallel"],
            "mode": mode,
            "iteration": 0,
        }

        if mode == "auto":
            _set_status(job_id, "running")
            final_state = graph.invoke(initial, config=config)
            _set_status(
                job_id, "done",
                draft=final_state.get("draft_markdown", ""),
                output_path=final_state.get("output_path", ""),
                email_sent=final_state.get("email_sent", False),
            )
        else:
            # HITL: invoke until interrupt, then expose draft for review
            _set_status(job_id, "running")
            graph.invoke(initial, config=config)
            # After the interrupt, fetch the current state
            snapshot = graph.get_state(config)
            draft = snapshot.values.get("draft_markdown", "")
            _set_status(job_id, "awaiting_human", draft=draft)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set_status(job_id, "error", error=str(exc))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_route():
    data = request.get_json() or {}
    goal = (data.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "log_queue": queue.Queue(),
        "status": "starting",
        "draft": "",
        "output_path": "",
        "email_sent": False,
    }

    params = {
        "goal": goal,
        "recipient_email": (data.get("email") or "").strip(),
        "top_n": int(data.get("top_n") or 7),
        "max_parallel": int(data.get("max_parallel") or 4),
        "mode": "hitl" if data.get("hitl") else "auto",
    }

    t = threading.Thread(target=_run_pipeline_thread, args=(job_id, params), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/logs/<job_id>")
def logs_stream(job_id: str):
    """Server-Sent Events stream of log lines for this job."""
    if job_id not in JOBS:
        return jsonify({"error": "unknown job"}), 404

    def gen():
        q = JOBS[job_id]["log_queue"]
        # Heartbeat so the browser doesn't close the connection
        last_beat = time.time()
        while True:
            try:
                item = q.get(timeout=2.0)
                yield f"data: {json.dumps(item)}\n\n"
                # Terminate stream when pipeline reaches a terminal state
                if item.get("_status") in ("done", "error", "awaiting_human"):
                    # Send one more event so client knows to stop polling
                    yield f"data: {json.dumps({'_end': True})}\n\n"
                    break
            except queue.Empty:
                if time.time() - last_beat > 15:
                    yield ": heartbeat\n\n"
                    last_beat = time.time()

    return Response(gen(), mimetype="text/event-stream")


@app.route("/status/<job_id>")
def status_route(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job.get("status"),
        "draft": job.get("draft", ""),
        "output_path": job.get("output_path", ""),
        "email_sent": job.get("email_sent", False),
        "error": job.get("error"),
    })


@app.route("/resume/<job_id>", methods=["POST"])
def resume_route(job_id: str):
    """HITL: user has reviewed the draft. Optionally accepts edited markdown."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "awaiting_human":
        return jsonify({"error": f"job not awaiting human (status={job['status']})"}), 400

    data = request.get_json() or {}
    edited_md = data.get("edited_markdown")

    graph = job["graph"]
    config = job["config"]

    # If user edited the draft, push the new value into the checkpointed state
    if edited_md:
        graph.update_state(config, {"draft_markdown": edited_md, "human_edits": edited_md})

    def _resume_thread():
        try:
            log_cb = _make_log_pipe(job_id)
            tools.set_log_callback(log_cb)
            scraper_pipeline.set_log_callback(log_cb)
            _set_status(job_id, "running")
            final_state = graph.invoke(None, config=config)  # None = resume
            _set_status(
                job_id, "done",
                draft=final_state.get("draft_markdown", ""),
                output_path=final_state.get("output_path", ""),
                email_sent=final_state.get("email_sent", False),
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _set_status(job_id, "error", error=str(exc))

    threading.Thread(target=_resume_thread, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    app.run(host="0.0.0.0", port=7860, debug=True, threaded=True, use_reloader=False)