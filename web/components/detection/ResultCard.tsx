"use client";

import { useState } from "react";
import {
  CheckCircle2,
  XCircle,
  Activity,
  Layers,
  Eye,
  Info,
  SlidersHorizontal,
} from "lucide-react";
import type { PredictionResult } from "@/lib/types";

type Props = {
  result: PredictionResult | null;
  modelId: string;
  originalImage: string | null;
};

const BAND_DESCRIPTIONS: Record<string, string> = {
  low_frequency: "Macro composition, global illumination & color distribution",
  mid_frequency: "Semantic structures, skin pores & fine texture continuity",
  high_frequency: "Edge phase continuity & spectral generator checkerboards",
  low: "Macro composition & color distribution",
  mid: "Semantic textures & surface continuity",
  high: "Edge spectral signatures & artifact residuals",
};

export default function ResultCard({ result, modelId, originalImage }: Props) {
  const [activeTab, setActiveTab] = useState<"verdict" | "heatmap">("verdict");

  if (!result) {
    return (
      <div className="flex h-full min-h-[380px] flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/50 p-8 text-center dark:border-zinc-800 dark:bg-zinc-950/40">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <Activity className="h-7 w-7 text-zinc-400 dark:text-zinc-500" />
        </div>
        <h3 className="mt-4 text-base font-bold text-zinc-800 dark:text-zinc-200">
          Forensic Analysis Workspace
        </h3>
        <p className="mt-1.5 max-w-xs text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
          Select or drop an image on the left to inspect multi-frequency artifacts, probabilities, and anomaly heatmaps.
        </p>
      </div>
    );
  }

  const isAI = result.prediction === "ai_generated";
  const confidencePct = (result.confidence * 100).toFixed(1);
  const realPct = (result.real_probability * 100).toFixed(1);
  const aiPct = (result.ai_probability * 100).toFixed(1);

  const bands = result.frequency_band_contributions || {};
  const bandEntries = Object.entries(bands);
  const totalBandPower = bandEntries.reduce((acc, [, val]) => acc + val, 0);

  return (
    <div className="glass-card animate-fade-in flex flex-col justify-between overflow-hidden p-6 shadow-xl dark:shadow-2xl">
      <div className="space-y-6">
        {/* Top Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500 dark:bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-900 dark:text-white">
              Forensic Verdict
            </h3>
          </div>
          <div className="flex items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-100/80 px-2.5 py-0.5 text-[11px] font-semibold text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
            <span>MFFT-{modelId.toUpperCase()}</span>
            <span className="text-zinc-400">·</span>
            <span>{result.processing_time_ms.toFixed(0)}ms</span>
          </div>
        </div>

        {/* Verdict Badge */}
        <div
          className={`relative overflow-hidden rounded-2xl border p-5 transition-all duration-300 ${
            isAI
              ? "border-rose-500/30 bg-rose-500/5 text-rose-950 dark:border-rose-500/40 dark:bg-rose-950/20 dark:text-rose-100 shadow-glow-rose"
              : "border-emerald-500/30 bg-emerald-500/5 text-emerald-950 dark:border-emerald-500/40 dark:bg-emerald-950/20 dark:text-emerald-100 shadow-glow-emerald"
          }`}
        >
          <div className="flex items-start gap-4">
            <div
              className={`flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl ${
                isAI
                  ? "bg-rose-500 text-white shadow-md shadow-rose-500/30"
                  : "bg-emerald-500 text-white shadow-md shadow-emerald-500/30"
              }`}
            >
              {isAI ? (
                <XCircle className="h-7 w-7" />
              ) : (
                <CheckCircle2 className="h-7 w-7" />
              )}
            </div>

            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h4 className="text-xl font-extrabold tracking-tight text-zinc-900 dark:text-white">
                  {isAI ? "Synthetic / AI-Generated" : "Authentic / Real Image"}
                </h4>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    isAI
                      ? "bg-rose-500/10 text-rose-600 dark:bg-rose-500/20 dark:text-rose-300"
                      : "bg-emerald-500/10 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-300"
                  }`}
                >
                  {confidencePct}% Confidence
                </span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-zinc-600 dark:text-zinc-300">
                {isAI
                  ? "Multi-frequency cross-attention flagged persistent phase and high-frequency spectral artifacts characteristic of deep generative models."
                  : "Radial Fourier profile and natural gradient decay correspond to authentic optical sensor capture with no generative artifacts detected."}
              </p>
            </div>
          </div>
        </div>

        {/* Tab Switcher for Deep Dive */}
        <div className="flex rounded-xl border border-zinc-200 bg-zinc-100/60 p-1 dark:border-zinc-800 dark:bg-zinc-900/60">
          <button
            type="button"
            onClick={() => setActiveTab("verdict")}
            className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-1.5 text-xs font-semibold transition-all ${
              activeTab === "verdict"
                ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-white"
                : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            Probabilities & Frequency Bands
          </button>
          {result.anomaly_heatmap && (
            <button
              type="button"
              onClick={() => setActiveTab("heatmap")}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-1.5 text-xs font-semibold transition-all ${
                activeTab === "heatmap"
                  ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-white"
                  : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              Anomaly Heatmap
            </button>
          )}
        </div>

        {/* Tab 1: Probabilities and Frequency Bands */}
        {activeTab === "verdict" && (
          <div className="space-y-5 animate-fade-in">
            {/* Probability Bars */}
            <div className="space-y-3 rounded-xl border border-zinc-200 bg-zinc-50/70 p-4 dark:border-zinc-800 dark:bg-zinc-900/40">
              <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                Probability Distribution
              </span>

              {/* Dual Bar */}
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-rose-600 dark:text-rose-400">AI-Generated</span>
                    <span className="text-zinc-900 dark:text-white">{aiPct}%</span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-rose-500 to-orange-400 transition-all duration-700"
                      style={{ width: `${aiPct}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-emerald-600 dark:text-emerald-400">Authentic Camera</span>
                    <span className="text-zinc-900 dark:text-white">{realPct}%</span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700"
                      style={{ width: `${realPct}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Frequency Bands Breakdown */}
            {bandEntries.length > 0 && (
              <div className="space-y-3 rounded-xl border border-zinc-200 bg-zinc-50/70 p-4 dark:border-zinc-800 dark:bg-zinc-900/40">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    Fourier Frequency Band Contributions
                  </span>
                  <SlidersHorizontal className="h-3.5 w-3.5 text-zinc-400" />
                </div>

                <div className="space-y-2.5">
                  {bandEntries.map(([band, val]) => {
                    const pct = totalBandPower > 0 ? (val / totalBandPower) * 100 : 0;
                    const desc = BAND_DESCRIPTIONS[band] || "Spectral decomposition band";
                    const cleanName = band
                      .replace(/_/g, " ")
                      .replace("frequency", "band")
                      .toUpperCase();

                    return (
                      <div key={band} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                            {cleanName}
                          </span>
                          <span className="font-mono text-xs font-bold text-zinc-900 dark:text-cyan-400">
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-700"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <p className="text-[10px] text-zinc-500 dark:text-zinc-400">{desc}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Anomaly Heatmap Deep Dive */}
        {activeTab === "heatmap" && result.anomaly_heatmap && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {originalImage && (
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    Input Image
                  </span>
                  <div className="aspect-square overflow-hidden rounded-xl border border-zinc-200 bg-zinc-950 dark:border-zinc-800">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={originalImage}
                      alt="Original input"
                      className="h-full w-full object-contain"
                    />
                  </div>
                </div>
              )}

              <div className="space-y-1.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Spatial Artifact Heatmap
                </span>
                <div className="aspect-square overflow-hidden rounded-xl border border-zinc-200 bg-zinc-950 dark:border-zinc-800">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${result.anomaly_heatmap}`}
                    alt="MFFT Anomaly Heatmap"
                    className="h-full w-full object-contain"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-start gap-2 rounded-xl border border-blue-500/20 bg-blue-500/5 p-3 text-xs leading-relaxed text-blue-900 dark:text-cyan-200">
              <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-500 dark:text-cyan-400" />
              <span>
                Bright luminous zones highlight high-frequency structural dissonance and phase inconsistency identified by the Frequency-Guided Cross-Attention layer.
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 border-t border-zinc-200 pt-4 text-[11px] text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
        Scientific research preview · Results are calibrated probabilities. For legal or forensic publication, pair with human verification.
      </div>
    </div>
  );
}
