"use client";

import { Check, Cpu, Zap, ShieldCheck } from "lucide-react";
import type { ModelOption } from "@/lib/types";

type Props = {
  models: ModelOption[];
  modelId: string;
  onSelect: (id: string) => void;
};

const MODEL_SPECS: Record<string, { badge: string; icon: any; accuracy: string }> = {
  tiny: { badge: "Edge / Fast", icon: Zap, accuracy: "98.59% Acc" },
  base: { badge: "Recommended", icon: ShieldCheck, accuracy: "98.58% Acc" },
  large: { badge: "Forensic Max", icon: Cpu, accuracy: "98.11% Acc" },
};

export default function ModelSelector({ models, modelId, onSelect }: Props) {
  return (
    <div className="w-full max-w-2xl">
      <div className="mb-3 flex items-center justify-between px-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          Select Detection Architecture
        </span>
        <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[11px] font-medium text-blue-600 dark:text-cyan-400">
          Verified 20-Epoch Checkpoints
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        {models.map((m) => {
          const active = m.id === modelId;
          const meta = MODEL_SPECS[m.id] || { badge: "Model", icon: ShieldCheck, accuracy: "98%+" };
          const Icon = meta.icon;

          return (
            <button
              key={m.id}
              type="button"
              disabled={!m.loaded}
              onClick={() => onSelect(m.id)}
              className={`group relative flex flex-col justify-between rounded-xl border p-3.5 text-left transition-all duration-200 ${
                active
                  ? "border-blue-600 bg-blue-50/70 shadow-sm ring-1 ring-blue-600 dark:border-cyan-500/80 dark:bg-zinc-900/90 dark:ring-cyan-500/50 dark:shadow-glow-sm"
                  : "border-zinc-200 bg-white/70 hover:border-zinc-300 hover:bg-zinc-50/80 dark:border-zinc-800/80 dark:bg-zinc-950/60 dark:hover:border-zinc-700 dark:hover:bg-zinc-900/40"
              } ${!m.loaded ? "cursor-not-allowed opacity-40" : ""}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div
                    className={`flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${
                      active
                        ? "bg-blue-600 text-white dark:bg-cyan-500 dark:text-black"
                        : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <span className="block text-sm font-bold capitalize text-zinc-900 dark:text-white">
                      MFFT-{m.id}
                    </span>
                    <span className="text-[11px] text-zinc-500 dark:text-zinc-400">
                      {m.params} params
                    </span>
                  </div>
                </div>
                {active && (
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-600 text-white dark:bg-cyan-500 dark:text-black">
                    <Check className="h-3 w-3 stroke-[3]" />
                  </span>
                )}
              </div>

              <div className="mt-3 flex items-center justify-between border-t border-zinc-100 pt-2 text-[11px] dark:border-zinc-800/60">
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                  {meta.accuracy}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    active
                      ? "bg-blue-200/60 text-blue-800 dark:bg-cyan-950/80 dark:text-cyan-300"
                      : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
                  }`}
                >
                  {meta.badge}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
