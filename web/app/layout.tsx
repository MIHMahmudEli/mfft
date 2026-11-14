import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "ImageVerify AI — Detect AI-Generated Images",
  description:
    "Explainable AI-generated image detection powered by the Multi-Frequency Fusion Transformer (MFFT). Frequency-band analysis, anomaly heatmaps, three model sizes.",
  keywords: [
    "AI detection",
    "deepfake detection",
    "image verification",
    "AI-generated images",
    "image forensics",
    "MFFT",
  ],
};

// runs before paint so the saved/system theme applies without a flash
const themeInit = `(function(){try{var t=localStorage.getItem("theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme: dark)").matches))document.documentElement.classList.add("dark")}catch(e){}})()`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body
        className={`${inter.className} min-h-screen bg-cream-50 text-stone-900 antialiased dark:bg-slate-950 dark:text-slate-100`}
      >
        {children}
      </body>
    </html>
  );
}
