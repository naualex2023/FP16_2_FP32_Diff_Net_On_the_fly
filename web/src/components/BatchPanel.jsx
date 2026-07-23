import React, { useState } from "react";
import JobProgress from "./JobProgress.jsx";
import { batch, downloadModel } from "../api.js";
import { DEFAULT_NEGATIVE, defaultModel } from "../presets.js";

/**
 * Batch generation: many prompts → 4-GPU multiprocess backend.
 * Prompts are entered one per line. Optional shared params (steps, size, etc.).
 */
export default function BatchPanel({ models, loras, onModelDownloaded }) {
  const [promptsText, setPromptsText] = useState("");
  const [shared, setShared] = useState({
    negative_prompt: DEFAULT_NEGATIVE,
    model_path: defaultModel("sdxl"),
    arch: "sdxl",
    steps: 25,
    width: 1024,
    height: 1024,
    base_seed: 42,
    guidance_scale: 7.5,
    scheduler: "default",
    lora_path: null,
    lora_scale: 1.0,
  });

  const [jobId, setJobId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [customRepo, setCustomRepo] = useState("");
  const [dlStatus, setDlStatus] = useState(null);

  const set = (key, value) => setShared((p) => ({ ...p, [key]: value }));

  const handleBatch = async () => {
    const prompts = promptsText
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (prompts.length === 0) {
      setError("Enter at least one prompt (one per line).");
      return;
    }
    setError(null);
    setBusy(true);
    setJobId(null);
    try {
      const { job_id } = await batch({ prompts, ...shared });
      setJobId(job_id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  const handleDownload = async () => {
    const repo = customRepo.trim();
    if (!repo || !repo.includes("/")) {
      setDlStatus({ state: "error", msg: "Enter a repo ID like org/name" });
      return;
    }
    setDlStatus({ state: "loading", msg: `Downloading ${repo}…` });
    try {
      const r = await downloadModel(repo);
      setDlStatus({ state: "ok", msg: `Saved ${repo} → ${r.path} (arch=${r.arch})` });
      setCustomRepo("");
      set("model_path", r.path);
      if (r.arch) set("arch", r.arch);
      if (onModelDownloaded) onModelDownloaded();
    } catch (e) {
      setDlStatus({ state: "error", msg: e.message });
    }
  };

  const promptCount = promptsText.split("\n").filter((l) => l.trim()).length;

  return (
    <div className="panel batch-panel">
      <div className="panel-layout">
        <div className="panel-left">
          <h2>Batch (4 GPUs · 1 model per GPU, no split)</h2>
          <p className="panel-desc">
            Run a batch of <strong>different prompts</strong> across all 4 GPUs — one complete (un-split)
            model per GPU (no pipeline split). Each GPU keeps its model cached for the whole batch.
          </p>

          {/* Prompts textarea */}
          <div className="control-group">
            <label htmlFor="batch-prompts">
              Prompts ({promptCount} queued) — one per line
            </label>
            <textarea
              id="batch-prompts"
              value={promptsText}
              onChange={(e) => setPromptsText(e.target.value)}
              rows={10}
              placeholder={"A cyberpunk cityscape at night\nA peaceful mountain lake at dawn\nAn astronaut floating above Earth\nA cozy bookshop interior"}
            />
          </div>

          <div className="control-row">
            <div className="control-group">
              <label htmlFor="b-arch">Architecture</label>
              <select id="b-arch" value={shared.arch} onChange={(e) => set("arch", e.target.value)}>
                <option value="sdxl">SDXL</option>
                <option value="sd15">SD 1.5</option>
              </select>
            </div>
            <div className="control-group grow">
              <label htmlFor="b-model">Model</label>
              <select id="b-model" value={shared.model_path} onChange={(e) => set("model_path", e.target.value)}>
                <option value={shared.model_path}>{shared.model_path}</option>
                {models.filter((m) => m.path !== shared.model_path).map((m) => (
                  <option key={m.path} value={m.path}>{m.name}{m.arch ? ` [${m.arch}]` : ""}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Custom HF model input for batch mode */}
          <div className="control-group">
            <label htmlFor="b-customrepo">Add HuggingFace model (repo ID)</label>
            <div className="custom-model-row">
              <input
                id="b-customrepo"
                type="text"
                value={customRepo}
                onChange={(e) => setCustomRepo(e.target.value)}
                placeholder="e.g. stabilityai/sdxl-turbo"
                onKeyDown={(e) => e.key === "Enter" && handleDownload()}
              />
              <button
                type="button"
                className="btn-secondary"
                onClick={handleDownload}
                disabled={dlStatus?.state === "loading"}
              >
                {dlStatus?.state === "loading" ? "…" : "⬇ Download"}
              </button>
              <button
                type="button"
                className="btn-icon"
                onClick={() => customRepo.trim() && set("model_path", customRepo.trim())}
                title="Use this repo ID directly (downloaded on first generation)"
              >
                ➤
              </button>
            </div>
            {dlStatus && (
              <div className={`dl-status dl-${dlStatus.state}`}>{dlStatus.msg}</div>
            )}
          </div>

          <div className="control-row">
            <div className="control-group">
              <label>Steps: <strong>{shared.steps}</strong></label>
              <input type="range" min={1} max={50} value={shared.steps}
                onChange={(e) => set("steps", parseInt(e.target.value))} />
            </div>
            <div className="control-group">
              <label>CFG: <strong>{shared.guidance_scale}</strong></label>
              <input type="range" min={0} max={20} step={0.5} value={shared.guidance_scale}
                onChange={(e) => set("guidance_scale", parseFloat(e.target.value))} />
            </div>
          </div>

          <div className="control-row">
            <div className="control-group">
              <label htmlFor="b-width">Width</label>
              <input id="b-width" type="number" step={8} min={64} max={2048} value={shared.width}
                onChange={(e) => set("width", parseInt(e.target.value) || 1024)} />
            </div>
            <div className="control-group">
              <label htmlFor="b-height">Height</label>
              <input id="b-height" type="number" step={8} min={64} max={2048} value={shared.height}
                onChange={(e) => set("height", parseInt(e.target.value) || 1024)} />
            </div>
            <div className="control-group">
              <label htmlFor="b-seed">Base Seed</label>
              <input id="b-seed" type="number" value={shared.base_seed}
                onChange={(e) => set("base_seed", parseInt(e.target.value) || 0)} />
            </div>
            <div className="control-group">
              <label htmlFor="b-sched">Scheduler</label>
              <select id="b-sched" value={shared.scheduler} onChange={(e) => set("scheduler", e.target.value)}>
                <option value="default">Default</option>
                <option value="ddim">DDIM</option>
                <option value="euler">Euler</option>
                <option value="dpmpp_2m">DPM++ 2M</option>
              </select>
            </div>
          </div>

          <div className="control-group">
            <label htmlFor="b-neg">Negative Prompt</label>
            <textarea id="b-neg" value={shared.negative_prompt} onChange={(e) => set("negative_prompt", e.target.value)} rows={2} />
          </div>

          <div className="actions">
            <button className="btn-primary" onClick={handleBatch} disabled={busy && !jobId}>
              {busy ? "Working…" : `📦 Run Batch (${promptCount})`}
            </button>
            {error && <span className="inline-error">{error}</span>}
          </div>
        </div>

        <div className="panel-right">
          <h3>Batch Progress</h3>
          {jobId ? (
            <JobProgress jobId={jobId} variant="batch" onDone={() => setBusy(false)} />
          ) : (
            <div className="placeholder">
              <p>Batch progress and results will appear here.</p>
              <p className="hint">Up to 4 prompts run in parallel (one per GPU), each on a full model.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}