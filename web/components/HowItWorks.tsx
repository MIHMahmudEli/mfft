import { HOW_IT_WORKS_STEPS } from "@/lib/content";
import { Sparkles } from "lucide-react";

export default function HowItWorks() {
  return (
    <section id="how" className="relative border-y border-zinc-200/80 bg-zinc-50/70 py-24 transition-colors duration-300 dark:border-zinc-800/80 dark:bg-zinc-950/60 sm:py-32">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="text-center">
          <span className="section-eyebrow">
            <Sparkles className="h-3 w-3" />
            The Forensic Method
          </span>
          <h2 className="section-heading">How MFFT Analyzes Images</h2>
          <p className="section-sub">
            A frequency-native deep architecture built specifically to detect generative model fingerprints where human vision fails.
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
          {HOW_IT_WORKS_STEPS.map((item, i) => (
            <div
              key={item.step}
              className="glass-card group relative p-8 transition-all duration-300 hover:-translate-y-1 hover:border-blue-500/40 dark:hover:border-cyan-500/40 dark:hover:shadow-glow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-3xl font-black tracking-tight text-blue-600 dark:text-cyan-400">
                  {item.step}
                </span>
                <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                  Stage {i + 1}
                </span>
              </div>

              <h3 className="mt-5 text-lg font-bold tracking-tight text-zinc-900 dark:text-white">
                {item.title}
              </h3>
              <p className="mt-2.5 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
                {item.desc}
              </p>

              {/* Subtle top accent bar */}
              <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-blue-500/40 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100 dark:via-cyan-400/40" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
