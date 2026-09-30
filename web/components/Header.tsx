import { Shield } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const NAV = [
  { href: "#detect", label: "Detect" },
  { href: "#how", label: "How It Works" },
  { href: "#pricing", label: "Pricing" },
  { href: "/docs", label: "API Docs" },
];

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-200/80 bg-white/75 backdrop-blur-xl dark:border-zinc-800/80 dark:bg-black/75">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <a href="#" className="group flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 p-0.5 shadow-sm transition-all duration-300 group-hover:shadow-glow-sm">
            <div className="flex h-full w-full items-center justify-center rounded-[10px] bg-white dark:bg-black">
              <Shield className="h-4.5 w-4.5 text-blue-600 transition-transform duration-300 group-hover:scale-110 dark:text-cyan-400" />
            </div>
          </div>
          <span className="text-lg font-bold tracking-tight text-zinc-900 dark:text-white">
            ImageVerify{" "}
            <span className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent dark:from-blue-400 dark:to-cyan-400">
              AI
            </span>
          </span>
        </a>

        <nav className="hidden items-center gap-8 text-sm font-medium text-zinc-600 dark:text-zinc-300 md:flex">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="transition-colors hover:text-blue-600 dark:hover:text-cyan-400"
            >
              {item.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-600 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-400 sm:flex">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500 dark:bg-emerald-400" />
            <span>MFFT v2.1 Online</span>
          </div>
          <ThemeToggle />
          <a
            href="#detect"
            className="btn-primary hidden px-4 py-2 text-xs md:inline-flex"
          >
            Start Detection
          </a>
        </div>
      </div>
    </header>
  );
}
