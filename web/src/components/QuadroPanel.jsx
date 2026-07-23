import React, { useState } from "react";
import Controls from "./Controls.jsx";
import JobProgress from "./JobProgress.jsx";
import { quadro } from "../api.js";
import { DEFAULT_NEGATIVE, defaultModel } from "../presets.js";

/**
 * Quadro mode: 4 images of the SAME prompt on 4 GPUs simultaneously.
 *
 * Unlike Twin, each GPU runs ONE complete (un-split) model in FP32 — no
 * pipeline-parallel UNet split.  Best for models that fit on a single GPU
 * (SD 1.5 FP32, SDXL FP16, SDXL-Turbo, ...).  GPU 0/1/2/3 each use
 * seed_a/seed_b/seed_c/seed_d respectively.
 */
export default function QuadroPanel({ models, loras, onModelDownloaded }) {
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
  const [seedC, setSeedC] = useState(-1);
  const [seedD, setSeedD] = useState(-1);
  const [jobId, setJobId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const handleQuadro = async () => {
    if (!params.prompt.trim()) {
      setError("Prompt is required");
      return;
    }
    setError(null);
    setBusy(true);
    setJobId(null);
    try {
      const { job_id } = await quadro({
        ...params,
        seed_a: seedA,
        seed_b: seedB,
        seed_c: seedC,
        seed_d: seedD,
      });
      setJobId(job_id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  const randomizeAll = () => {
    const r = () => Math.floor(Math.random() * 2 ** 31);
    setSeedA(r());
    setSeedB(r());
    setSeedC(r());
    setSeedD(r());
  };

  const sameSeed = seedA === seedB && seedB === seedC && seedC === seedD && seedA >= 0;

  return (
    <div className="panel quadro-panel">
      <div className="panel-layout">
        <div className="panel-left">
          <h2>Quadro Mode (4 images · 4 GPUs · no split)</h2>
          <p className="panel-desc">
            Generate <strong>four</strong> images of the same prompt simultaneously — one complete
            (un-split) FP32 model per GPU (0, 1, 2, 3). No UNet split is needed, so this is faster
            and simpler for models that fit on a single GPU. Different seeds give you four variations
            in the time of one.
          </p>

          <Controls params={params} setParams={setParams} models={models} loras={loras} onModelDownloaded={onModelDownloaded} />

          {/* Quadro-specific: four seeds */}
          <div className="control-row quadro-seeds">
            <div className="control-group">
              <label htmlFor="seed_a">Seed A (GPU 0)</label>
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
              <label htmlFor="seed_b">Seed B (GPU 1)</label>
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
              <label htmlFor="seed_c">Seed C (GPU 2)</label>
              <div className="seed-row">
                <input
                  id="seed_c"
                  type="number"
                  value={seedC}
                  onChange={(e) => setSeedC(parseInt(e.target.value) || -1)}
                />
              </div>
            </div>
            <div className="control-group">
              <label htmlFor="seed_d">Seed D (GPU 3)</label>
              <div className="seed-row">
                <input
                  id="seed_d"
                  type="number"
                  value={seedD}
                  onChange={(e) => setSeedD(parseInt(e.target.value) || -1)}
                />
              </div>
            </div>
            <div className="control-group">
              <button type="button" className="btn-icon" onClick={randomizeAll} title="Randomize all">
                🎲 All
              </button>
            </div>
          </div>

          <div className="control-group">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={sameSeed}
                onChange={(e) => {
                  if (e.target.checked) {
                    const s = Math.floor(Math.random() * 2 ** 31);
                    setSeedA(s);
                    setSeedB(s);
                    setSeedC(s);
                    setSeedD(s);
                  }
                }}
              />
              Use same seed for all (identical images)
            </label>
          </div>

          <div className="actions">
            <button className="btn-primary" onClick={handleQuadro} disabled={busy && !jobId}>
              {busy ? "Working…" : "⚡ Generate Quadro (4 GPU, no split)"}
            </button>
            {error && <span className="inline-error">{error}</span>}
          </div>
        </div>

        <div className="panel-right">
          <h3>Output</h3>
          {jobId ? (
            <JobProgress jobId={jobId} variant="quadro" onDone={() => setBusy(false)} />
          ) : (
            <div className="placeholder">
              <p>Four variations of your prompt will appear in a 2×2 grid.</p>
              <p className="hint">Each GPU loads its own full model — no pipeline split.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}