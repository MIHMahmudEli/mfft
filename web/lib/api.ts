import type { ModelOption, PredictionResult } from "./types";

/**
 * Detection API backend configuration.
 * Defaults to the persistent Hugging Face Spaces production service.
 * Automatic fallback prevents broken routing if an old Render variable is cached.
 */
export function getApiBase(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl && !envUrl.includes("onrender.com")) {
    return envUrl.replace(/\/+$/, "");
  }
  return "https://mohsinelis-mfft-api.hf.space";
}

export async function fetchModels(): Promise<{
  default: string | null;
  models: ModelOption[];
} | null> {
  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/models`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function predictImage(
  file: File,
  modelId: string
): Promise<PredictionResult> {
  const apiBase = getApiBase();
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${apiBase}/predict?model=${encodeURIComponent(modelId)}`, {
    method: "POST",
    headers: { Authorization: "Bearer free" },
    body: formData,
  });

  if (!res.ok) {
    let detail = "Detection failed";
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}
