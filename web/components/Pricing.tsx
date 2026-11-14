import { CheckCircle2, Sparkles, Zap } from "lucide-react";
import { PRICING_PLANS } from "@/lib/content";

export default function Pricing() {
  return (
    <section id="pricing" className="relative bg-white py-24 transition-colors duration-300 dark:bg-black sm:py-32">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="text-center">
          <span className="section-eyebrow">
            <Sparkles className="h-3 w-3" />
            Transparent Tiers
          </span>
          <h2 className="section-heading">Flexible Plans for Every Scale</h2>
          <p className="section-sub">
            Free tier includes full explainability heatmaps and frequency band analytics.
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
          {PRICING_PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`glass-card relative flex flex-col justify-between p-8 transition-all duration-300 hover:-translate-y-1 ${
                plan.featured
                  ? "border-blue-600 bg-blue-50/20 ring-1 ring-blue-600 dark:border-cyan-500/80 dark:bg-zinc-900/80 dark:ring-cyan-500/40 dark:shadow-glow md:scale-105"
                  : "hover:border-zinc-300 dark:hover:border-zinc-700"
              }`}
            >
              {plan.featured && (
                <div className="absolute -top-3.5 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 px-3.5 py-1 text-xs font-bold text-white shadow-glow-sm">
                  <Zap className="h-3 w-3 fill-current" />
                  Most Popular
                </div>
              )}

              <div>
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    {plan.name}
                  </h3>
                </div>

                <div className="mt-4 flex items-baseline gap-1">
                  <span className="text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-white">
                    {plan.price}
                  </span>
                  <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    {plan.period}
                  </span>
                </div>

                <ul className="mt-8 space-y-3.5 border-t border-zinc-100 pt-6 dark:border-zinc-800">
                  {plan.features.map((f) => (
                    <li
                      key={f}
                      className="flex items-start gap-3 text-sm text-zinc-600 dark:text-zinc-300"
                    >
                      <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-500 dark:text-emerald-400" />
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="mt-8 pt-4">
                <a
                  href="#detect"
                  className={`flex w-full items-center justify-center rounded-xl py-3 text-sm font-bold transition-all duration-200 ${
                    plan.featured
                      ? "btn-primary w-full shadow-md"
                      : "border border-zinc-300 bg-white text-zinc-800 hover:border-zinc-400 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800"
                  }`}
                >
                  {plan.cta}
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
