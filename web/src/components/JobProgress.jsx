import React, { useEffect, useState } from "react";
import { subscribeJob } from "../api.js";

/**
 * Live job progress display. Subscribes to SSE for the given jobId.
 * Props:
 *   jobId — the job to track
 *   onDone(job) — called when status === "done"
 *   variant — "single" | "twin" | "quadro" | "batch"
 */
export default function JobProgress({ jobId, onDone, variant = "single" }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    const cleanup = subscribeJob(
      jobId,
      (event) => {
        // Merge the event patch into accumulated job state.
        setJob((prev) => ({ ...(prev || {}), ...event }));
        if (event.status === "done" && onDone) onDone({ ...(job || {}), ...event });
        if (event.status === "failed") setError(event.error || "Generation failed");
      },
      setError
    );
    return cleanup;
  }, [jobId]);

  if (!jobId) return null;
  if (!job) return <div className="job-card pending">Queued…</div>;
  if (error) {
    return (
      <div className="job-card failed">
        <div className="job-header">
          <span className="status-badge failed">✗ Failed</span>
        </div>
        <pre className="error-text">{error}</pre>
      </div>
    );
  }

  const pct = job.progress || 0;
  const statusLabel = {
    queued: "Queued",
    running: "Running",
    done: "✓ Done",
    failed: "✗ Failed",
  }[job.status] || job.status;

  return (
    <div className={`job-card ${job.status}`}>
      <div className="job-header">
        <span className={`status-badge ${job.status}`}>{statusLabel}</span>
        {variant === "single" && <span className="job-prompt">{job.prompt?.slice(0, 60)}</span>}
        {variant === "twin" && <span className="job-prompt">Twin: {job.prompt?.slice(0, 50)}</span>}
        {variant === "quadro" && <span className="job-prompt">Quadro: {job.prompt?.slice(0, 46)}</span>}
        {variant === "batch" && (
          <span className="job-prompt">
            Batch: {job.completed || 0} / {job.total || 0}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-info">
        <span>{pct}%</span>
        {job.stage && <span className="stage">{job.stage}</span>}
        {job.step != null && job.total_steps && (
          <span>
            step {job.step}/{job.total_steps}
          </span>
        )}
      </div>

      {/* Twin sub-progress */}
      {variant === "twin" && job.sub_jobs && (
        <div className="twin-progress">
          <div className="sub-bar">
            <span>Pair A (GPU 0+1)</span>
            <div className="mini-bar">
              <div
                className="mini-fill"
                style={{ width: `${job.sub_jobs.a?.progress || 0}%` }}
              />
            </div>
            <span>{job.sub_jobs.a?.progress || 0}%</span>
          </div>
          <div className="sub-bar">
            <span>Pair B (GPU 2+3)</span>
            <div className="mini-bar">
              <div
                className="mini-fill"
                style={{ width: `${job.sub_jobs.b?.progress || 0}%` }}
              />
            </div>
            <span>{job.sub_jobs.b?.progress || 0}%</span>
          </div>
        </div>
      )}

      {/* Quadro sub-progress: one bar per GPU */}
      {variant === "quadro" && job.sub_jobs && (
        <div className="quadro-progress">
          {["a", "b", "c", "d"].map((label, i) => (
            <div className="sub-bar" key={label}>
              <span>GPU {i}</span>
              <div className="mini-bar">
                <div
                  className="mini-fill"
                  style={{ width: `${job.sub_jobs[label]?.progress || 0}%` }}
                />
              </div>
              <span>{job.sub_jobs[label]?.progress || 0}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Results */}
      {job.status === "done" && (
        <div className="job-results">
          {variant === "single" && job.image_url && (
            <img src={job.image_url} alt="result" className="result-img" />
          )}
          {variant === "twin" && (
            <div className="twin-results">
              {job.image_url_a && <img src={job.image_url_a} alt="result A" className="result-img half" />}
              {job.image_url_b && <img src={job.image_url_b} alt="result B" className="result-img half" />}
            </div>
          )}
          {variant === "quadro" && (
            <div className="quadro-results">
              {job.image_url_a && <img src={job.image_url_a} alt="result A" className="result-img quad" />}
              {job.image_url_b && <img src={job.image_url_b} alt="result B" className="result-img quad" />}
              {job.image_url_c && <img src={job.image_url_c} alt="result C" className="result-img quad" />}
              {job.image_url_d && <img src={job.image_url_d} alt="result D" className="result-img quad" />}
            </div>
          )}
          {variant === "batch" && job.image_urls && (
            <div className="batch-results">
              {job.image_urls.map((url, i) => (
                <img key={i} src={url} alt={`result ${i}`} className="result-thumb" />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}