import { Shield, ExternalLink, Github } from "lucide-react";
import { LINKS } from "@/lib/content";

export default function Footer() {
  return (
    <footer className="border-t border-zinc-200/80 bg-zinc-50 py-16 text-zinc-600 transition-colors duration-300 dark:border-zinc-800/80 dark:bg-black dark:text-zinc-400">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="grid gap-12 md:grid-cols-4">
          <div className="md:col-span-2">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-400 p-0.5">
                <div className="flex h-full w-full items-center justify-center rounded-[10px] bg-white dark:bg-black">
                  <Shield className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                </div>
              </div>
              <span className="text-base font-bold text-zinc-900 dark:text-white">
                ImageVerify AI
              </span>
            </div>
            <p className="mt-3.5 max-w-sm text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
              Explainable AI-generated image forensics, powered by the Multi-Frequency Fusion Transformer (MFFT). Delivering radial Fourier decomposition, cross-attention fusion, and localized anomaly heatmaps.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-900 dark:text-white">
              Navigation
            </h4>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <a href="#detect" className="transition-colors hover:text-blue-600 dark:hover:text-cyan-400">
                  Detect Images
                </a>
              </li>
              <li>
                <a href="#how" className="transition-colors hover:text-blue-600 dark:hover:text-cyan-400">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#pricing" className="transition-colors hover:text-blue-600 dark:hover:text-cyan-400">
                  Pricing Plans
                </a>
              </li>
              <li>
                <a href="/docs" className="transition-colors hover:text-blue-600 dark:hover:text-cyan-400">
                  API Documentation
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-900 dark:text-white">
              Deployment & Science
            </h4>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <a
                  href="https://mfft-detection-api.onrender.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-blue-600 dark:hover:text-cyan-400"
                >
                  <span>Render API Endpoint</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href={LINKS.space}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-blue-600 dark:hover:text-cyan-400"
                >
                  <span>Hugging Face Space</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href={LINKS.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 transition-colors hover:text-blue-600 dark:hover:text-cyan-400"
                >
                  <Github className="h-3.5 w-3.5" />
                  <span>GitHub Repository</span>
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-14 flex flex-col items-center justify-between gap-4 border-t border-zinc-200/80 pt-6 text-xs text-zinc-500 dark:border-zinc-800/80 dark:text-zinc-500 md:flex-row">
          <span>
            &copy; {new Date().getFullYear()} ImageVerify AI. Multi-Frequency Fusion Transformer (MFFT).
          </span>
          <span className="text-zinc-500 dark:text-zinc-400">
            Calibrated forensic evidence · Not legal proof.
          </span>
        </div>
      </div>
    </footer>
  );
}
