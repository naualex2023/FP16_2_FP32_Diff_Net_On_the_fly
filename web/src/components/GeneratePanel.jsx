import React, { useState } from "react";
import Controls from "./Controls.jsx";
import JobProgress from "./JobProgress.jsx";
import { generate } from "../api.js";
import { DEFAULT_NEGATIVE, defaultModel } from "../presets.js";

/**
 * Single-image generation on one GPU pair.
 * Props:
 *   models, loras — option lists
 *   initialParams — optional preset config (from Gallery "reuse")
 *   onModelDownloaded — refresh callback after a new model is downloaded
 */
export default function GeneratePanel({ models, loras, initialParams, onModelDownloaded }) {
  const [params, setParams] = useState({
    prompt: initialParams?.prompt ?? "",
    negative_prompt: initialParams?.negative_prompt ?? DEFAULT_NEGATIVE,
    model_path: initialParams?.model_path ?? defaultModel("sdxl"),
    arch: initialParams?.arch ?? "sdxl",
    steps: initialParams?.steps ?? 25,
    width: initialParams?.width ?? 1024,
    height: initialParams?.height ?? 1024,
    seed: initialParams?.seed ?? -1,
    guidance_scale: initialParams?.guidance_scale ?? 7.5,
    scheduler: initialParams?.scheduler ?? "default",
    lora_path: initialParams?.lora_path ?? null,
    lora_scale: initialParams?.lora_scale ?? 1.0,
    gpu_pair: initialParams?.gpu_pair ?? "0+1",
    use_fp32: true,
  });

  const [jobId, setJobId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!params.prompt.trim()) {
      setError("Prompt is required");
      return;
    }
    setError(null);
    setBusy(true);
    setJobId(null);
    try {
      const { job_id } = await generate(params);
      setJobId(job_id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  return (
    <div className="panel generate-panel">
      <div className="panel-layout">
        <div className="panel-left">
          <h2>Generate (1 image · 2 GPUs)</h2>
          <Controls params={params} setParams={setParams} models={models} loras={loras} onModelDownloaded={onModelDownloaded} />

          {/* GPU pair selector */}
          <div className="control-row">
            <div className="control-group">
              <label htmlFor="gpu_pair">GPU Pair</label>
              <select
                id="gpu_pair"
                value={params.gpu_pair}
                onChange={(e) => setParams((p) => ({ ...p, gpu_pair: e.target.value }))}
              >
                <option value="0+1">Pair A — GPU 0 + 1</option>
                <option value="2+3">Pair B — GPU 2 + 3</option>
              </select>
            </div>
          </div>

          <div className="actions">
            <button
              className="btn-primary"
              onClick={handleGenerate}
              disabled={busy && !jobId}
            >
              {busy ? "Working…" : "🎨 Generate"}
            </button>
            {error && <span className="inline-error">{error}</span>}
          </div>
        </div>

        <div className="panel-right">
          <h3>Output</h3>
          {jobId ? (
            <JobProgress
              jobId={jobId}
              variant="single"
              onDone={() => setBusy(false)}
            />
          ) : (
            <div className="placeholder">
              <p>Your generated image will appear here.</p>
              <p className="hint">Tip: use the same GPU pair for back-to-back jobs — the model stays cached.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}