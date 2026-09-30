import type { ModelOption, PredictionResult } from "./types";

/**
 * All requests go through the Next.js rewrite (`/api/*` -> detection API),
 * so the browser never makes cross-origin calls. The backend target is
 * configured with NEXT_PUBLIC_API_URL (see next.config.js).
 */

export async function fetchModels(): Promise<{
  default: string | null;
  models: ModelOption[];
} | null> {
  try {
    const res = await fetch("/api/models");
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
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`/api/predict?model=${encodeURIComponent(modelId)}`, {
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
