import { ChevronRight, Sparkles } from "lucide-react";
import { HERO_STATS } from "@/lib/content";

export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-white text-zinc-900 transition-colors duration-300 dark:bg-black dark:text-white">
      {/* Light mode ambient glows */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 dark:hidden"
        style={{
          background:
            "radial-gradient(900px 450px at 50% 0%, rgba(59, 130, 246, 0.08), transparent 70%), radial-gradient(800px 400px at 80% 80%, rgba(99, 102, 241, 0.06), transparent 70%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.03] dark:hidden"
        style={{
          backgroundImage:
            "linear-gradient(to right, #000 1px, transparent 1px), linear-gradient(to bottom, #000 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Classic Deep Black Ambient Glows */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 hidden dark:block"
        style={{
          background:
            "radial-gradient(800px 400px at 50% -10%, rgba(59, 130, 246, 0.25), transparent 75%), radial-gradient(600px 300px at 20% 60%, rgba(99, 102, 241, 0.15), transparent 70%), radial-gradient(600px 300px at 80% 50%, rgba(6, 182, 212, 0.12), transparent 70%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 hidden opacity-[0.07] dark:block"
        style={{
          backgroundImage:
            "linear-gradient(to right, #27272a 1px, transparent 1px), linear-gradient(to bottom, #27272a 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      <div className="relative mx-auto max-w-6xl px-4 py-20 text-center sm:px-6 md:py-28">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50/80 px-4 py-1.5 text-xs font-medium text-zinc-700 shadow-sm backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-300 dark:shadow-glow-sm">
          <Sparkles className="h-3.5 w-3.5 text-blue-600 dark:text-cyan-400" />
          <span>Multi-Frequency Fusion Transformer · 98.6% Accuracy</span>
        </div>

        <h1 className="mx-auto max-w-4xl text-4xl font-extrabold leading-[1.1] tracking-tight sm:text-6xl md:text-7xl">
          Forensic AI Image Detection.
          <span
            className="block bg-gradient-to-r from-blue-600 via-indigo-500 to-cyan-500 bg-clip-text pb-2 pt-1 text-transparent dark:from-blue-400 dark:via-indigo-300 dark:to-cyan-400"
          >
            Unmask Generative Artifacts.
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-zinc-600 dark:text-zinc-400 sm:text-xl">
          MFFT decomposes images across radial Fourier frequency bands where generative models leave subtle fingerprints, delivering explainable verdicts, per-band evidence, and localized anomaly heatmaps.
        </p>

        <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <a
            href="#detect"
            className="inline-flex h-12 items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-600 px-8 text-sm font-semibold text-white shadow-md transition-all duration-200 hover:-translate-y-0.5 hover:shadow-glow hover:brightness-110 active:translate-y-0"
          >
            Analyze an Image <ChevronRight className="h-4 w-4" />
          </a>
          <a
            href="#how"
            className="inline-flex h-12 items-center gap-2 rounded-xl border border-zinc-200 bg-white/60 px-8 text-sm font-semibold text-zinc-800 backdrop-blur-md transition-all duration-200 hover:border-zinc-300 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900/50 dark:text-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800/80"
          >
            How It Works
          </a>
        </div>

        {/* Stats Grid */}
        <dl className="mx-auto mt-16 grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3">
          {HERO_STATS.map((s) => (
            <div
              key={s.label}
              className="glass-card group p-5 transition-all duration-300 hover:-translate-y-1 hover:border-blue-500/40 dark:hover:border-cyan-500/30 dark:hover:shadow-glow-sm"
            >
              <dd className="text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white">
                {s.value}
              </dd>
              <dt className="mt-1 text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                {s.label}
              </dt>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
