import React, { useState } from "react";
import { ASPECT_RATIOS, QUICK_PRESETS, DEFAULT_NEGATIVE, defaultModel } from "../presets.js";
import { downloadModel } from "../api.js";

/**
 * Shared generation-parameter controls.
 * Props:
 *   params, setParams — controlled state (the generate form config)
 *   models, loras — option lists from the API
 *   onModelDownloaded — optional callback when a new model is downloaded
 *                      (e.g. to refresh the models list in the parent)
 */
export default function Controls({ params, setParams, models = [], loras = [], onModelDownloaded }) {
  const set = (key, value) => setParams((p) => ({ ...p, [key]: value }));

  // Custom HF repo input + download state.
  const [customRepo, setCustomRepo] = useState("");
  const [dlStatus, setDlStatus] = useState(null); // {state:"idle|loading|ok|error", msg}

  const applyPreset = (preset) => {
    setParams((p) => ({
      ...p,
      ...preset.config,
      model_path: defaultModel(preset.config.arch),
      negative_prompt: preset.config.guidance_scale === 0 ? "" : DEFAULT_NEGATIVE,
    }));
  };

  const applyAspect = (ratio) => {
    const dim = ASPECT_RATIOS[ratio];
    if (dim) {
      set("width", dim.w);
      set("height", dim.h);
    }
  };

  const randomizeSeed = () => set("seed", Math.floor(Math.random() * 2 ** 31));

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
      // Auto-select the newly downloaded model + architecture.
      set("model_path", r.path);
      if (r.arch) set("arch", r.arch);
      if (onModelDownloaded) onModelDownloaded();
    } catch (e) {
      setDlStatus({ state: "error", msg: e.message });
    }
  };

  return (
    <div className="controls">
      {/* Quick presets */}
      <div className="control-group">
        <label>Quick Presets</label>
        <div className="preset-chips">
          {QUICK_PRESETS.map((p) => (
            <button
              key={p.id}
              className="chip"
              type="button"
              onClick={() => applyPreset(p)}
              title={`${p.config.steps} steps · CFG ${p.config.guidance_scale}`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Prompt */}
      <div className="control-group">
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          value={params.prompt}
          onChange={(e) => set("prompt", e.target.value)}
          placeholder="A serene Japanese garden, koi pond, cherry blossoms, golden hour, 8k..."
          rows={4}
        />
      </div>

      {/* Negative prompt */}
      <div className="control-group">
        <label htmlFor="negative">Negative Prompt</label>
        <textarea
          id="negative"
          value={params.negative_prompt}
          onChange={(e) => set("negative_prompt", e.target.value)}
          rows={2}
        />
      </div>

      {/* Architecture + model */}
      <div className="control-row">
        <div className="control-group">
          <label htmlFor="arch">Architecture</label>
          <select id="arch" value={params.arch} onChange={(e) => set("arch", e.target.value)}>
            <option value="sdxl">SDXL</option>
            <option value="sd15">SD 1.5</option>
          </select>
        </div>
        <div className="control-group grow">
          <label htmlFor="model">Model</label>
          <select
            id="model"
            value={params.model_path}
            onChange={(e) => set("model_path", e.target.value)}
          >
            {/* Current value is always present (even if it's a custom repo ID). */}
            <option value={params.model_path}>{params.model_path}</option>
            {models
              .filter((m) => m.path !== params.model_path)
              .map((m) => (
                <option key={m.path} value={m.path}>
                  {m.name}{m.arch ? ` [${m.arch}]` : ""}
                </option>
              ))}
          </select>
        </div>
      </div>

      {/* Custom HF model: type a repo ID to download or use directly. */}
      <div className="control-group">
        <label htmlFor="customrepo">Add HuggingFace model (repo ID)</label>
        <div className="custom-model-row">
          <input
            id="customrepo"
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
            title="Download the repo into ./models (stored as FP16; upcast to FP32 at load)"
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
        <div className="hint">
          Tip: a repo ID (e.g. <code>stabilityai/sdxl-turbo</code>) can be passed
          directly as the model — it is downloaded on first use. Models are stored
          on disk as FP16 and upcast to FP32 in memory at load time.
        </div>
      </div>

      {/* Dimensions */}
      <div className="control-row">
        <div className="control-group">
          <label>Aspect Ratio</label>
          <select onChange={(e) => applyAspect(e.target.value)} defaultValue="">
            <option value="" disabled>
              Choose…
            </option>
            {Object.keys(ASPECT_RATIOS).map((r) => (
              <option key={r} value={r}>
                {r} ({ASPECT_RATIOS[r].w}×{ASPECT_RATIOS[r].h})
              </option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <label htmlFor="width">Width</label>
          <input
            id="width"
            type="number"
            step={8}
            min={64}
            max={2048}
            value={params.width}
            onChange={(e) => set("width", parseInt(e.target.value) || 1024)}
          />
        </div>
        <div className="control-group">
          <label htmlFor="height">Height</label>
          <input
            id="height"
            type="number"
            step={8}
            min={64}
            max={2048}
            value={params.height}
            onChange={(e) => set("height", parseInt(e.target.value) || 1024)}
          />
        </div>
      </div>

      {/* Sliders */}
      <div className="control-row">
        <Slider
          label="Steps"
          value={params.steps}
          min={1}
          max={50}
          step={1}
          onChange={(v) => set("steps", v)}
        />
        <Slider
          label="CFG (Guidance)"
          value={params.guidance_scale}
          min={0}
          max={20}
          step={0.5}
          onChange={(v) => set("guidance_scale", v)}
        />
      </div>

      {/* Scheduler + seed */}
      <div className="control-row">
        <div className="control-group">
          <label htmlFor="scheduler">Scheduler</label>
          <select
            id="scheduler"
            value={params.scheduler}
            onChange={(e) => set("scheduler", e.target.value)}
          >
            <option value="default">Default</option>
            <option value="ddim">DDIM</option>
            <option value="euler">Euler</option>
            <option value="dpmpp_2m">DPM++ 2M</option>
          </select>
        </div>
        <div className="control-group">
          <label htmlFor="seed">Seed</label>
          <div className="seed-row">
            <input
              id="seed"
              type="number"
              value={params.seed}
              onChange={(e) => set("seed", parseInt(e.target.value) || -1)}
            />
            <button type="button" className="btn-icon" onClick={randomizeSeed} title="Random seed">
              🎲
            </button>
          </div>
        </div>
      </div>

      {/* LoRA */}
      <div className="control-row">
        <div className="control-group grow">
          <label htmlFor="lora">LoRA (optional)</label>
          <select
            id="lora"
            value={params.lora_path || ""}
            onChange={(e) => set("lora_path", e.target.value || null)}
          >
            <option value="">None</option>
            {loras.map((l) => (
              <option key={l.path} value={l.path}>
                {l.name}
              </option>
            ))}
          </select>
        </div>
        <Slider
          label="LoRA Scale"
          value={params.lora_scale}
          min={0}
          max={2}
          step={0.1}
          onChange={(v) => set("lora_scale", v)}
          disabled={!params.lora_path}
        />
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, disabled }) {
  return (
    <div className={`control-group ${disabled ? "disabled" : ""}`}>
      <label>
        {label}: <strong>{value}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}