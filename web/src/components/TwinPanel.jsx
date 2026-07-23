import React, { useState } from "react";
import Controls from "./Controls.jsx";
import JobProgress from "./JobProgress.jsx";
import { twin } from "../api.js";
import { DEFAULT_NEGATIVE, defaultModel } from "../presets.js";

/**
 * Twin mode: 2 images of the SAME prompt on 4 GPUs simultaneously.
 * Pair A (GPU 0+1) generates with seed_a, Pair B (GPU 2+3) with seed_b.
 */
export default function TwinPanel({ models, loras, onModelDownloaded }) {
  const [params, setParams] = useState({
    prompt: "",
    negative_prompt: DEFAULT_NEGATIVE,
    model_path: defaultModel("sdxl"),
    arch: "sdxl",
    steps: 25,
    width: 1024,
    height: 1024,
    guidance_scale: 7.5,
    scheduler: "default",
    lora_path: null,
    lora_scale: 1.0,
    use_fp32: true,
  });

  const [seedA, setSeedA] = useState(-1);
  const [seedB, setSeedB] = useState(-1);
  const [jobId, setJobId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const handleTwin = async () => {
    if (!params.prompt.trim()) {
      setError("Prompt is required");
      return;
    }
    setError(null);
    setBusy(true);
    setJobId(null);
    try {
      const { job_id } = await twin({ ...params, seed_a: seedA, seed_b: seedB });
      setJobId(job_id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  const randomizeBoth = () => {
    setSeedA(Math.floor(Math.random() * 2 ** 31));
    setSeedB(Math.floor(Math.random() * 2 ** 31));
  };

  return (
    <div className="panel twin-panel">
      <div className="panel-layout">
        <div className="panel-left">
          <h2>Twin Mode (2 images · 4 GPUs)</h2>
          <p className="panel-desc">
            Generate <strong>two</strong> images of the same prompt simultaneously — Pair A on GPU
            0+1, Pair B on GPU 2+3. Different seeds give you two variations in the time of one.
          </p>

          <Controls params={params} setParams={setParams} models={models} loras={loras} onModelDownloaded={onModelDownloaded} />

          {/* Twin-specific: two seeds */}
          <div className="control-row twin-seeds">
            <div className="control-group">
              <label htmlFor="seed_a">Seed A (GPU 0+1)</label>
              <div className="seed-row">
                <input
                  id="seed_a"
                  type="number"
                  value={seedA}
                  onChange={(e) => setSeedA(parseInt(e.target.value) || -1)}
                />
              </div>
            </div>
            <div className="control-group">
              <label htmlFor="seed_b">Seed B (GPU 2+3)</label>
              <div className="seed-row">
                <input
                  id="seed_b"
                  type="number"
                  value={seedB}
                  onChange={(e) => setSeedB(parseInt(e.target.value) || -1)}
                />
              </div>
            </div>
            <div className="control-group">
              <button type="button" className="btn-icon" onClick={randomizeBoth} title="Randomize both">
                🎲 Both
              </button>
            </div>
          </div>

          <div className="control-group">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={seedA === seedB && seedA >= 0}
                onChange={(e) => {
                  if (e.target.checked) {
                    const s = Math.floor(Math.random() * 2 ** 31);
                    setSeedA(s);
                    setSeedB(s);
                  }
                }}
              />
              Use same seed for both (identical images)
            </label>
          </div>

          <div className="actions">
            <button className="btn-primary" onClick={handleTwin} disabled={busy && !jobId}>
              {busy ? "Working…" : "⚡ Generate Twin (4 GPU)"}
            </button>
            {error && <span className="inline-error">{error}</span>}
          </div>
        </div>

        <div className="panel-right">
          <h3>Output</h3>
          {jobId ? (
            <JobProgress jobId={jobId} variant="twin" onDone={() => setBusy(false)} />
          ) : (
            <div className="placeholder">
              <p>Two variations of your prompt will appear side by side.</p>
              <p className="hint">Uses all 4 GPUs at once for maximum throughput.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}