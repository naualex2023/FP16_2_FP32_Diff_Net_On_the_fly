// presets.js — default config and quick presets for the generate panel.

export const ASPECT_RATIOS = {
  "1:1": { w: 1024, h: 1024 },
  "3:4": { w: 896, h: 1152 },
  "4:3": { w: 1152, h: 896 },
  "16:9": { w: 1344, h: 768 },
  "9:16": { w: 768, h: 1344 },
  "2:3": { w: 832, h: 1216 },
  "3:2": { w: 1216, h: 832 },
};

export const QUICK_PRESETS = [
  {
    id: "sdxl-quality",
    label: "SDXL Quality",
    config: { steps: 25, guidance_scale: 7.5, scheduler: "default", arch: "sdxl" },
  },
  {
    id: "sdxl-fast",
    label: "SDXL Fast",
    config: { steps: 15, guidance_scale: 7.0, scheduler: "dpmpp_2m", arch: "sdxl" },
  },
  {
    id: "turbo",
    label: "Turbo (4 steps)",
    config: { steps: 4, guidance_scale: 0.0, scheduler: "default", arch: "sdxl" },
  },
  {
    id: "sd15-quick",
    label: "SD 1.5 Quick",
    config: { steps: 15, guidance_scale: 7.5, scheduler: "ddim", arch: "sd15" },
  },
  {
    id: "creative",
    label: "Creative",
    config: { steps: 35, guidance_scale: 12.0, scheduler: "euler", arch: "sdxl" },
  },
  {
    id: "pixart",
    label: "PixArt-α (DiT)",
    config: { steps: 20, guidance_scale: 4.5, scheduler: "default", arch: "dit" },
  },
  {
    id: "gguf-sd3",
    label: "SD3.5 GGUF (Q4)",
    config: { steps: 20, guidance_scale: 4.5, scheduler: "default", arch: "gguf" },
  },
];

export const DEFAULT_NEGATIVE =
  "blurry, low quality, distorted, ugly, watermark, text, deformed, extra limbs, bad anatomy";

export function defaultModel(arch) {
  if (arch === "sd15") return "./models/sd15-fp16";
  if (arch === "dit") return "PixArt-alpha/PixArt-XL-2-1024-MS";
  if (arch === "gguf") return "./models/sd35-large-Q4_K_M.gguf";
  return "./models/sdxl-base-fp16";
}