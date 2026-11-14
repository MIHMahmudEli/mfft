"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, Sparkles, Image as ImageIcon, Scan } from "lucide-react";

type Props = {
  preview: string | null;
  file: File | null;
  isAnalyzing: boolean;
  onFile: (file: File) => void;
  onReset: () => void;
  onLoadSample: (sampleUrl: string, filename: string) => void;
};

export default function UploadZone({
  preview,
  file,
  isAnalyzing,
  onFile,
  onReset,
  onLoadSample,
}: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    maxSize: 20 * 1024 * 1024,
    disabled: isAnalyzing,
  });

  if (preview) {
    return (
      <div className="relative overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-950 shadow-md dark:border-zinc-800">
        <div className="relative flex aspect-square max-h-[420px] w-full items-center justify-center overflow-hidden bg-zinc-900/50">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Selected image"
            className="h-full w-full object-contain"
          />

          {/* Futuristic laser scanner when analyzing */}
          {isAnalyzing && (
            <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden bg-blue-500/5">
              <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_3px_rgba(6,182,212,0.8)] animate-scan" />
              <div className="absolute inset-x-0 bottom-4 flex justify-center">
                <span className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-black/80 px-3.5 py-1 text-xs font-semibold text-cyan-400 backdrop-blur-md">
                  <Scan className="h-3.5 w-3.5 animate-spin" />
                  Extracting Radial Fourier Bands...
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Top bar with file info and remove button */}
        <div className="absolute left-3 right-3 top-3 z-30 flex items-center justify-between">
          <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/60 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-md">
            <ImageIcon className="h-3.5 w-3.5 text-cyan-400" />
            <span className="max-w-[180px] truncate">{file?.name || "Sample Image"}</span>
            {file && (
              <span className="text-zinc-400">
                · {(file.size / (1024 * 1024)).toFixed(2)} MB
              </span>
            )}
          </div>

          {!isAnalyzing && (
            <button
              onClick={onReset}
              aria-label="Remove image"
              className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-black/60 text-zinc-300 backdrop-blur-md transition-all duration-200 hover:bg-black/90 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-300 ${
          isDragActive
            ? "border-blue-500 bg-blue-500/10 shadow-glow dark:border-cyan-400 dark:bg-cyan-500/10"
            : "border-zinc-300 bg-zinc-50/50 hover:border-zinc-400 hover:bg-zinc-100/50 dark:border-zinc-800 dark:bg-zinc-950/60 dark:hover:border-zinc-700 dark:hover:bg-zinc-900/40"
        }`}
      >
        <input {...getInputProps()} />

        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-blue-600/10 to-cyan-500/10 border border-blue-500/20 text-blue-600 transition-transform duration-300 group-hover:scale-110 dark:border-cyan-500/30 dark:text-cyan-400 dark:shadow-glow-sm">
          <Upload className="h-7 w-7" />
        </div>

        <p className="mt-4 text-base font-bold text-zinc-900 dark:text-white">
          Drop your image here, or{" "}
          <span className="text-blue-600 underline decoration-blue-500/30 underline-offset-4 dark:text-cyan-400">
            browse
          </span>
        </p>
        <p className="mt-1.5 text-xs text-zinc-500 dark:text-zinc-400">
          Supports PNG, JPG, JPEG, or WebP up to 20 MB
        </p>

        <div className="mt-5 flex items-center gap-2 rounded-full border border-zinc-200 bg-white/60 px-3 py-1 text-[11px] font-medium text-zinc-600 backdrop-blur-sm dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-400">
          <Sparkles className="h-3 w-3 text-blue-500 dark:text-cyan-400" />
          <span>Zero compression · Client-side preview</span>
        </div>
      </div>

      {/* Quick Test Samples */}
      <div className="flex flex-col items-center gap-2 pt-1 text-center sm:flex-row sm:justify-center">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
          Try a demo sample:
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => onLoadSample("/samples/ai_sample.jpg", "ai_synthetic_pattern.jpg")}
            className="rounded-lg border border-rose-500/20 bg-rose-500/5 px-2.5 py-1 text-xs font-medium text-rose-600 transition-colors hover:border-rose-500/40 hover:bg-rose-500/10 dark:text-rose-400"
          >
            Synthetic Pattern (AI)
          </button>
          <button
            type="button"
            onClick={() => onLoadSample("/samples/real_sample.jpg", "authentic_natural_photo.jpg")}
            className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-2.5 py-1 text-xs font-medium text-emerald-600 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/10 dark:text-emerald-400"
          >
            Authentic Photo (Real)
          </button>
        </div>
      </div>
    </div>
  );
}
