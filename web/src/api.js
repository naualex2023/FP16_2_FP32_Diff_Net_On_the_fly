// api.js — thin fetch wrapper for the FP32 Diffusion backend.

const BASE = ""; // Same origin (vite proxy in dev, FastAPI static in prod)

async function jsonOrThrow(resp) {
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json();
}

export async function getConfig() {
  return jsonOrThrow(await fetch(`${BASE}/api/config`));
}

export async function getModels() {
  return jsonOrThrow(await fetch(`${BASE}/api/models`));
}

export async function getLora() {
  return jsonOrThrow(await fetch(`${BASE}/api/lora`));
}

export async function getGpus() {
  return jsonOrThrow(await fetch(`${BASE}/api/gpus`));
}

export async function getCacheStats() {
  return jsonOrThrow(await fetch(`${BASE}/api/cache/stats`));
}

export async function unloadAll() {
  return jsonOrThrow(
    await fetch(`${BASE}/api/cache/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "unload_all" }),
    })
  );
}

export async function generate(payload) {
  return jsonOrThrow(
    await fetch(`${BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function twin(payload) {
  return jsonOrThrow(
    await fetch(`${BASE}/api/twin`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function batch(payload) {
  return jsonOrThrow(
    await fetch(`${BASE}/api/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function getHistory() {
  return jsonOrThrow(await fetch(`${BASE}/api/history`));
}

export async function deleteHistory(jobId) {
  return jsonOrThrow(
    await fetch(`${BASE}/api/history/${jobId}`, { method: "DELETE" })
  );
}

export async function getJobs() {
  return jsonOrThrow(await fetch(`${BASE}/api/jobs`));
}

/**
 * Subscribe to a job's progress via Server-Sent Events.
 * @param {string} jobId
 * @param {(event: object) => void} onUpdate
 * @param {(err: Error) => void} onError
 * @returns {() => void} cleanup function to close the stream
 */
export function subscribeJob(jobId, onUpdate, onError) {
  const ctrl = new AbortController();

  fetch(`${BASE}/api/jobs/${jobId}/events`, {
    signal: ctrl.signal,
    headers: { Accept: "text/event-stream" },
  })
    .then(async (resp) => {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          try {
            onUpdate(JSON.parse(line.slice(5).trim()));
          } catch (e) {
            // ignore parse errors on heartbeats
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError(err);
    });

  return () => ctrl.abort();
}