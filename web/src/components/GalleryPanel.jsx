import React, { useEffect, useState, useCallback } from "react";
import { getHistory, deleteHistory } from "../api.js";
import { DEFAULT_NEGATIVE } from "../presets.js";

/**
 * Gallery & history. Shows all generated images with metadata.
 * "Reuse params" loads config back into the active generate panel.
 */
export default function GalleryPanel({ onReuseParams }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const { history: h } = await getHistory();
      setHistory(h);
    } catch (e) {
      console.error("Failed to load history:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDelete = async (jobId) => {
    if (!confirm(`Delete image(s) for job ${jobId}?`)) return;
    await deleteHistory(jobId);
    refresh();
  };

  const handleReuse = (entry) => {
    onReuseParams({
      prompt: entry.prompt || "",
      negative_prompt: entry.negative_prompt || DEFAULT_NEGATIVE,
      model_path: entry.model_path || "./models/sdxl-base-fp16",
      arch: entry.arch || "sdxl",
      steps: entry.steps || 25,
      width: entry.width || 1024,
      height: entry.height || 1024,
      seed: entry.seed ?? -1,
      guidance_scale: entry.guidance_scale ?? 7.5,
      scheduler: entry.scheduler || "default",
      lora_path: entry.lora_path || null,
      lora_scale: entry.lora_scale ?? 1.0,
    });
  };

  return (
    <div className="panel gallery-panel">
      <div className="gallery-header">
        <h2>Gallery</h2>
        <button className="btn-secondary" onClick={refresh}>
          ↻ Refresh
        </button>
      </div>

      {loading ? (
        <p>Loading…</p>
      ) : history.length === 0 ? (
        <div className="placeholder">
          <p>No images yet. Generate some in the Generate, Twin, or Batch tabs!</p>
        </div>
      ) : (
        <div className="gallery-grid">
          {history.map((entry, i) => (
            <div key={i} className="gallery-card">
              {entry.image_url && (
                <img
                  src={entry.image_url}
                  alt={entry.prompt}
                  className="gallery-thumb"
                  onClick={() => setSelected(entry)}
                  loading="lazy"
                />
              )}
              <div className="gallery-meta">
                <div className="gallery-prompt" title={entry.prompt}>
                  {entry.prompt?.slice(0, 80)}
                </div>
                <div className="gallery-tags">
                  <span className="tag">{entry.arch}</span>
                  <span className="tag">{entry.width}×{entry.height}</span>
                  <span className="tag">{entry.steps} steps</span>
                  <span className="tag">CFG {entry.guidance_scale}</span>
                  <span className="tag">seed {entry.seed}</span>
                  {entry.gpu_pair && <span className="tag">GPU {entry.gpu_pair}</span>}
                </div>
                <div className="gallery-actions">
                  <button className="btn-small" onClick={() => handleReuse(entry)}>
                    ↻ Reuse
                  </button>
                  <a className="btn-small" href={entry.image_url} download>
                    ⬇ Download
                  </a>
                  <button className="btn-small danger" onClick={() => handleDelete(entry.job_id)}>
                    🗑 Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox modal */}
      {selected && (
        <div className="lightbox" onClick={() => setSelected(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img src={selected.image_url} alt={selected.prompt} />
            <div className="lightbox-meta">
              <p><strong>Prompt:</strong> {selected.prompt}</p>
              {selected.negative_prompt && <p><strong>Negative:</strong> {selected.negative_prompt}</p>}
              <p>
                {selected.arch} · {selected.width}×{selected.height} · {selected.steps} steps ·
                CFG {selected.guidance_scale} · seed {selected.seed} · {selected.scheduler}
              </p>
            </div>
            <button className="lightbox-close" onClick={() => setSelected(null)}>✕</button>
          </div>
        </div>
      )}
    </div>
  );
}