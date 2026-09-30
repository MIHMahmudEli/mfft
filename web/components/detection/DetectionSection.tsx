"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Shield, Loader2, Sparkles } from "lucide-react";
import { fetchModels, predictImage } from "@/lib/api";
import { FALLBACK_MODELS } from "@/lib/content";
import type { ModelOption, PredictionResult } from "@/lib/types";
import ModelSelector from "./ModelSelector";
import UploadZone from "./UploadZone";
import ResultCard from "./ResultCard";

const STORAGE_KEY = "mfft_model";

export default function DetectionSection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<string>("Analyzing image...");
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelOption[]>(FALLBACK_MODELS);
  const [modelId, setModelId] = useState<string>("base");

  // Restore saved model choice, then refresh the live list from the API
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setModelId(saved);
    fetchModels().then((data) => {
      if (!data?.models?.length) return;
      setModels(data.models);
      const stillValid = data.models.some(
        (m) => m.id === (saved ?? "base") && m.loaded
      );
      if (!stillValid && data.default) setModelId(data.default);
    });
  }, []);

  // Helpful loading phases in case Render free tier is waking up
  useEffect(() => {
    let t1: NodeJS.Timeout;
    let t2: NodeJS.Timeout;
    if (loading) {
      setLoadingPhase("Analyzing image across radial Fourier bands...");
      t1 = setTimeout(() => {
        setLoadingPhase("Performing cross-attention frequency fusion...");
      }, 3000);
      t2 = setTimeout(() => {
        setLoadingPhase("Waking up cloud inference worker (Render free tier cold start)...");
      }, 8000);
    }
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [loading]);

  const selectModel = (id: string) => {
    setModelId(id);
    localStorage.setItem(STORAGE_KEY, id);
    setResult(null);
  };

  const handleFile = (f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleLoadSample = async (sampleUrl: string, filename: string) => {
    try {
      setError(null);
      setResult(null);
      const res = await fetch(sampleUrl);
      const blob = await res.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
      handleFile(sampleFile);
    } catch (err) {
      console.error("Failed to load sample:", err);
      setError("Failed to load sample image. Please upload a file manually.");
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file, modelId);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="detect" className="relative bg-zinc-50/50 py-20 transition-colors duration-300 dark:bg-black sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="text-center">
          <span className="section-eyebrow">
            <Sparkles className="h-3 w-3" />
            Live Forensic Detection
          </span>
          <h2 className="section-heading">Inspect Image Authenticity</h2>
          <p className="section-sub">
            Drag & drop an image to perform frequency decomposition and verify whether it was synthesized by generative AI.
          </p>
        </div>

        {/* Model Selection Tabs */}
        <div className="mt-10 flex justify-center">
          <ModelSelector
            models={models}
            modelId={modelId}
            onSelect={selectModel}
          />
        </div>

        {/* Workspace: Upload + Analysis on Left, Results on Right */}
        <div className="mt-10 grid grid-cols-1 gap-8 lg:grid-cols-2">
          <div className="flex flex-col justify-between space-y-4">
            <UploadZone
              preview={preview}
              file={file}
              isAnalyzing={loading}
              onFile={handleFile}
              onReset={reset}
              onLoadSample={handleLoadSample}
            />

            {preview && !result && (
              <button
                onClick={handleDetect}
                disabled={loading}
                className="btn-primary mt-2 flex h-13 w-full items-center justify-center gap-2 rounded-xl text-base shadow-lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    <span>Analyzing Multi-Frequency Bands...</span>
                  </>
                ) : (
                  <>
                    <Shield className="h-5 w-5" />
                    <span>Run Forensic AI Detection</span>
                  </>
                )}
              </button>
            )}

            {/* Cold Start / Status Notice */}
            {loading && (
              <div className="animate-fade-in flex items-center justify-center gap-2 rounded-xl border border-blue-500/20 bg-blue-500/5 px-4 py-2.5 text-xs text-blue-600 dark:border-cyan-500/20 dark:bg-cyan-950/30 dark:text-cyan-300">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>{loadingPhase}</span>
              </div>
            )}

            {error && (
              <div className="animate-fade-in flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-700 dark:border-rose-500/40 dark:bg-rose-950/30 dark:text-rose-300">
                <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-rose-500" />
                <div className="flex-1">
                  <span className="font-bold">Error:</span> {error}
                  {error.includes("504") || error.includes("Failed to fetch") ? (
                    <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
                      The Render backend free tier may be waking up from idle. Please wait ~15 seconds and retry.
                    </p>
                  ) : null}
                </div>
              </div>
            )}
          </div>

          <div>
            <ResultCard
              result={result}
              modelId={modelId}
              originalImage={preview}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
