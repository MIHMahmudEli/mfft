"use client";

import { Shield, Code2, Key, CreditCard, ArrowLeft } from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";

const endpoints = [
  {
    method: "GET",
    path: "/",
    desc: "Health check — verifies the API is running and reports loaded models",
    response: `{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0",
  "models_loaded": ["tiny", "base", "large"],
  "timestamp": "2026-09-30T10:00:00Z"
}`,
  },
  {
    method: "GET",
    path: "/models",
    desc: "List available MFFT models and their load statuses",
    response: `{
  "default": "base",
  "models": [
    { "id": "tiny", "loaded": true, "params": "372K", "description": "Fastest — edge & mobile profile" },
    { "id": "base", "loaded": true, "params": "1.62M", "description": "Balanced accuracy and speed (98.58% acc)" },
    { "id": "large", "loaded": true, "params": "6.30M", "description": "Highest capacity profile (98.11% acc)" }
  ]
}`,
  },
  {
    method: "POST",
    path: "/predict",
    desc: "Upload an image for AI detection",
    request: `curl -X POST https://api.imageverify.ai/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@image.jpg"`,
    response: `{
  "prediction": "ai_generated",
  "confidence": 0.972,
  "real_probability": 0.028,
  "ai_probability": 0.972,
  "processing_time_ms": 7.1,
  "anomaly_heatmap": "base64...",
  "tier": { "rpm": 100, "batch_size": 10, "report": true }
}`,
  },
  {
    method: "POST",
    path: "/predict/batch",
    desc: "Analyze multiple images in one request (Pro tier: max 10, Enterprise: max 100)",
    response: `{
  "results": [
    { "filename": "img1.jpg", "prediction": "ai_generated", ... },
    { "filename": "img2.jpg", "prediction": "real", ... }
  ],
  "summary": {
    "total": 2,
    "ai_generated": 1,
    "real": 1,
    "avg_real_probability": 0.486,
    "avg_ai_probability": 0.514
  }
}`,
  },
  {
    method: "GET",
    path: "/usage",
    desc: "Check your current rate limit usage",
    response: `{
  "tier": "pro",
  "requests_this_minute": 42,
  "rate_limit": 100
}`,
  },
];

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-white transition-colors duration-300 dark:bg-black">
      <header className="sticky top-0 z-50 border-b border-zinc-200/80 bg-white/75 backdrop-blur-xl dark:border-zinc-800/80 dark:bg-black/75">
        <div className="mx-auto flex h-16 max-w-5xl items-center gap-3 px-4">
          <a
            href="/"
            className="flex items-center gap-1.5 text-sm text-zinc-500 transition-colors hover:text-blue-600 dark:text-zinc-400 dark:hover:text-cyan-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Home
          </a>
          <span className="h-5 w-px bg-zinc-200 dark:bg-zinc-800" />
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-tr from-blue-600 to-cyan-500">
            <Shield className="h-4 w-4 text-white" />
          </div>
          <span className="text-lg font-bold text-zinc-900 dark:text-white">
            Image<span className="text-blue-600 dark:text-cyan-400">Verify</span>{" "}
            AI
          </span>
          <span className="ml-auto text-sm text-zinc-500 dark:text-zinc-400">
            API Documentation
          </span>
          <ThemeToggle />
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 py-12">
        <h1 className="mb-4 text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-white">
          API Reference
        </h1>
        <p className="mb-8 max-w-2xl text-zinc-600 dark:text-zinc-400">
          Our REST API lets you integrate AI image detection directly into your platforms, workflows, and automated moderation pipelines.
        </p>

        <div className="mb-12 grid gap-6 md:grid-cols-3">
          {[
            {
              icon: Key,
              title: "Authentication",
              desc: "Pass your API key in the Authorization header: Bearer YOUR_KEY",
            },
            {
              icon: CreditCard,
              title: "Free tier included",
              desc: "30 requests/min free with full explainability heatmaps.",
            },
            {
              icon: Code2,
              title: "SDKs & Python Client",
              desc: "Simple standard HTTP multipart request interface.",
            },
          ].map((item) => (
            <div
              key={item.title}
              className="glass-card p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-blue-500/40 dark:hover:border-cyan-500/30 dark:hover:shadow-glow-sm"
            >
              <item.icon className="mb-3 h-6 w-6 text-blue-600 dark:text-cyan-400" />
              <h3 className="mb-1 font-bold text-zinc-900 dark:text-white">
                {item.title}
              </h3>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        <div className="space-y-8">
          {endpoints.map((ep) => (
            <div
              key={ep.path}
              className="glass-card overflow-hidden shadow-sm"
            >
              <div className="flex items-center gap-3 border-b border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
                <span
                  className={`rounded px-2 py-0.5 text-xs font-bold ${
                    ep.method === "GET"
                      ? "bg-emerald-500/10 text-emerald-600 dark:bg-emerald-400/10 dark:text-emerald-400"
                      : "bg-blue-500/10 text-blue-600 dark:bg-cyan-400/10 dark:text-cyan-400"
                  }`}
                >
                  {ep.method}
                </span>
                <code className="font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  {ep.path}
                </code>
              </div>
              <div className="space-y-4 p-5">
                <p className="text-sm text-zinc-600 dark:text-zinc-300">
                  {ep.desc}
                </p>
                <div>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    Example Request
                  </p>
                  <pre className="overflow-x-auto rounded-xl bg-zinc-950 p-4 font-mono text-xs text-zinc-200 border border-zinc-800">
                    <code>{ep.request || "N/A"}</code>
                  </pre>
                </div>
                <div>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    Example Response
                  </p>
                  <pre className="overflow-x-auto rounded-xl bg-zinc-950 p-4 font-mono text-xs text-zinc-200 border border-zinc-800">
                    <code>{ep.response}</code>
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 rounded-xl border border-blue-100 bg-blue-50 p-6 dark:border-blue-900/50 dark:bg-blue-950/40">
          <h2 className="mb-2 text-lg font-semibold text-stone-900 dark:text-white">
            Get Your API Key
          </h2>
          <p className="mb-4 text-sm text-stone-600 dark:text-slate-300">
            Sign up for a free account to receive your API key instantly.
          </p>
          <button className="btn-primary px-5 py-2 text-sm">
            Get Free API Key
          </button>
        </div>
      </div>
    </div>
  );
}
