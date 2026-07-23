import React, { useEffect, useState, useCallback } from "react";
import { getGpus, getCacheStats, unloadAll } from "../api.js";

/**
 * Status dashboard: GPU info, live VRAM, pipeline cache state, manual controls.
 */
export default function StatusPanel() {
  const [gpus, setGpus] = useState([]);
  const [vram, setVram] = useState([]);
  const [cache, setCache] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [g, c] = await Promise.all([getGpus(), getCacheStats()]);
      setGpus(g.gpus || []);
      setVram(g.live_vram || []);
      setCache(c);
    } catch (e) {
      console.error("Status fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    // Poll every 5s for live VRAM.
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleUnload = async () => {
    if (!confirm("Unload ALL cached models from VRAM?")) return;
    const { unloaded } = await unloadAll();
    alert(`Unloaded ${unloaded} pipeline(s).`);
    refresh();
  };

  const vramMap = {};
  vram.forEach((v) => (vramMap[v.index] = v));

  return (
    <div className="panel status-panel">
      <div className="status-header">
        <h2>Status Dashboard</h2>
        <button className="btn-secondary" onClick={refresh}>↻ Refresh</button>
      </div>

      {loading ? (
        <p>Loading…</p>
      ) : (
        <>
          {/* GPUs */}
          <div className="status-section">
            <h3>GPUs</h3>
            {gpus.length === 0 ? (
              <p className="hint">No CUDA GPUs detected.</p>
            ) : (
              <div className="gpu-grid">
                {gpus.map((gpu) => {
                  const live = vramMap[gpu.index] || {};
                  const pct = live.total_vram_gb
                    ? Math.round((live.allocated_gb / live.total_vram_gb) * 100)
                    : 0;
                  return (
                    <div key={gpu.index} className="gpu-card">
                      <div className="gpu-name">
                        GPU {gpu.index}: {gpu.name}
                      </div>
                      <div className="gpu-vram-bar">
                        <div className="gpu-vram-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <div className="gpu-vram-text">
                        {live.allocated_gb?.toFixed(1) || 0} / {gpu.total_vram_gb} GB
                        <span className="gpu-pct">({pct}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Pipeline cache */}
          <div className="status-section">
            <div className="cache-header">
              <h3>Pipeline Cache</h3>
              <button className="btn-small danger" onClick={handleUnload}>
                Unload All
              </button>
            </div>
            {cache?.error ? (
              <p className="inline-error">{cache.error}</p>
            ) : (
              <div className="cache-info">
                <div className="cache-stats-row">
                  <span>
                    Keep-alive: <strong>{String(cache?.keep_alive)}</strong>
                  </span>
                  <span>
                    Idle timeout: <strong>{cache?.idle_timeout}s</strong>
                  </span>
                  <span>
                    Resident: <strong>{cache?.resident}</strong>
                  </span>
                </div>
                {cache?.entries?.length > 0 && (
                  <table className="cache-table">
                    <thead>
                      <tr>
                        <th>Key</th>
                        <th>Idle (s)</th>
                        <th>Build (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cache.entries.map((e, i) => (
                        <tr key={i}>
                          <td className="cache-key">{e.key}</td>
                          <td>{e.idle_seconds}</td>
                          <td>{e.build_seconds}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}