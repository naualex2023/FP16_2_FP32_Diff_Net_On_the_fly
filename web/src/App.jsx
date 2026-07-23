import React, { useEffect, useState } from "react";
import GeneratePanel from "./components/GeneratePanel.jsx";
import TwinPanel from "./components/TwinPanel.jsx";
import QuadroPanel from "./components/QuadroPanel.jsx";
import BatchPanel from "./components/BatchPanel.jsx";
import GalleryPanel from "./components/GalleryPanel.jsx";
import StatusPanel from "./components/StatusPanel.jsx";
import { getModels, getLora } from "./api.js";

const TABS = [
  { id: "generate", label: "🎨 Generate", title: "Single image (2 GPUs)" },
  { id: "twin", label: "⚡ Twin", title: "2 images same prompt (4 GPUs, split)" },
  { id: "quadro", label: "🔮 Quadro", title: "4 images same prompt (4 GPUs, no split)" },
  { id: "batch", label: "📦 Batch", title: "Multiple prompts (4 GPUs, no split)" },
  { id: "gallery", label: "🖼 Gallery", title: "History & gallery" },
  { id: "status", label: "📊 Status", title: "GPU & cache dashboard" },
];

export default function App() {
  const [tab, setTab] = useState("generate");
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);
  const [reuseParams, setReuseParams] = useState(null);

  const refreshModels = () => {
    getModels().then((r) => setModels(r.models)).catch(() => {});
  };

  useEffect(() => {
    refreshModels();
    getLora().then((r) => setLoras(r.loras)).catch(() => {});
  }, []);

  const handleReuse = (params) => {
    setReuseParams(params);
    setTab("generate");
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">🖥</span>
          <div>
            <h1>FP32 Diffusion Studio</h1>
            <span className="subtitle">Pipeline-parallel FP32 on 4× Tesla P40</span>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
              title={t.title}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {tab === "generate" && <GeneratePanel models={models} loras={loras} key={reuseParams ? Date.now() : "g"} initialParams={reuseParams} onModelDownloaded={refreshModels} />}
        {tab === "twin" && <TwinPanel models={models} loras={loras} onModelDownloaded={refreshModels} />}
        {tab === "quadro" && <QuadroPanel models={models} loras={loras} onModelDownloaded={refreshModels} />}
        {tab === "batch" && <BatchPanel models={models} loras={loras} onModelDownloaded={refreshModels} />}
        {tab === "gallery" && <GalleryPanel onReuseParams={handleReuse} />}
        {tab === "status" && <StatusPanel />}
      </main>

      <footer className="app-footer">
        <span>FP32 Diffusion Studio · 4× Tesla P40 · Pipeline Parallel</span>
      </footer>
    </div>
  );
}