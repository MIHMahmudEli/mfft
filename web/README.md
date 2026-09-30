# ImageVerify AI

**Explainable AI-generated image detection, in the browser.**

ImageVerify AI is the production web front-end for **MFFT** (Multi-Frequency
Fusion Transformer), a research model that detects AI-generated images by
analyzing them across multiple frequency bands — the domain where generative
models leave their most persistent fingerprints. Upload an image and get an
explainable verdict: real-vs-AI probability, per-frequency-band evidence, and
an anomaly heatmap highlighting the suspicious regions.

**Live detection API:** [`mfft-api` on Hugging Face Spaces](https://huggingface.co/spaces/MohsinEli/mfft-api)

Built with Next.js 14 (App Router), TypeScript, and Tailwind CSS.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Integration](#api-integration)
- [Available Models](#available-models)
- [Theming](#theming)
- [Scripts](#scripts)
- [Deployment](#deployment)
- [Browser Support](#browser-support)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Features

- **Drag-and-drop analysis** — drop or browse a PNG/JPG/WebP image (up to
  20 MB) and get a verdict in seconds, with client-side preview.
- **Explainable results**, not just a score:
  - Real vs AI probability bars with model confidence
  - **Frequency-band contribution chart** — which bands (low/mid/high)
    drove the decision
  - **Anomaly heatmap** overlay pinpointing the regions the model flagged
- **Model selector** — switch between three MFFT sizes (Tiny / Base / Large)
  at runtime; the choice persists in `localStorage`.
- **Same-origin API proxy** — the browser only ever calls `/api/*`; Next.js
  rewrites forward requests to the detection backend, so no CORS setup is
  needed anywhere.
- **Vintage-classic theme** — warm cream "paper" light mode and espresso
  dark mode with a flash-free theme toggle (saved preference, falls back to
  the system setting).
- **Fully static-prerendered marketing pages** — hero, how-it-works,
  pricing, and API docs ship as static HTML for instant loads.
- **Responsive & accessible** — mobile-first layout, semantic markup,
  keyboard-friendly controls.

## Architecture

```
┌────────────┐     /api/models        ┌─────────────────────┐
│  Browser    │ ──────────────────►   │  Next.js (this app) │
│  (React UI) │     /api/predict      │  rewrites /api/* ──►│──┐
└────────────┘                        └─────────────────────┘  │
                                                               ▼
                                              ┌───────────────────────────┐
                                              │  MFFT Detection API        │
                                              │  (FastAPI on HF Spaces or  │
                                              │   any self-hosted server)  │
                                              └───────────────────────────┘
```

1. The UI requests the available model registry from `/api/models`.
2. An uploaded image is POSTed as multipart form data to
   `/api/predict?model=<id>`.
3. Next.js rewrites both to `NEXT_PUBLIC_API_URL` server-side — the backend
   URL is a single environment variable and the browser never makes a
   cross-origin call.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Next.js 14](https://nextjs.org/) (App Router, static prerendering) |
| Language | TypeScript 5 |
| Styling | Tailwind CSS 3 with a custom vintage palette |
| Charts | [Recharts](https://recharts.org/) (frequency-band visualization) |
| Upload | [react-dropzone](https://react-dropzone.js.org/) |
| Icons | [lucide-react](https://lucide.dev/) |
| Font | Inter via `next/font` (self-hosted, zero layout shift) |

## Project Structure

```
.
├── app/
│   ├── layout.tsx          # Root layout, metadata, flash-free theme init
│   ├── page.tsx            # Landing page (hero → detect → how → pricing)
│   ├── globals.css         # Tailwind layers + design tokens (light/dark)
│   ├── icon.svg            # Favicon (auto-served by the App Router)
│   └── docs/page.tsx       # API documentation page
├── components/
│   ├── Header.tsx          # Sticky nav with theme toggle
│   ├── Hero.tsx            # Headline, stats, backdrop
│   ├── HowItWorks.tsx      # MFFT pipeline explainer
│   ├── Pricing.tsx         # Plan tiers
│   ├── Footer.tsx
│   ├── ThemeToggle.tsx     # Dark/light switch (localStorage + system)
│   └── detection/
│       ├── DetectionSection.tsx   # Orchestrates the analysis flow
│       ├── UploadZone.tsx         # Drag-and-drop + preview
│       ├── ModelSelector.tsx      # Runtime model switching
│       └── ResultCard.tsx         # Verdict, bars, band chart, heatmap
├── lib/
│   ├── api.ts              # Typed API client (fetchModels, predictImage)
│   ├── types.ts            # PredictionResult, ModelOption
│   └── content.ts          # Copy, stats, nav, pricing data
├── next.config.js          # /api/* → backend rewrite
├── tailwind.config.js      # Vintage cream/espresso palette, animations
└── tsconfig.json
```

## Getting Started

### Prerequisites

- Node.js 18.17+ (or 20+)
- npm 9+

### Install & Run

```bash
git clone https://github.com/MIHMahmudEli/AI-ImageVerify.git
cd AI-ImageVerify
npm install
cp .env.example .env.local   # points at the live API by default
npm run dev                  # → http://localhost:3000
```

The default `.env.example` targets the hosted detection API, so the app is
fully functional out of the box — no backend setup required.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Base URL of the MFFT detection API | `http://localhost:8000` |

Examples:

```bash
# Hosted demo backend (Hugging Face Space)
NEXT_PUBLIC_API_URL=https://mohsineli-mfft-api.hf.space

# Local FastAPI dev server
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The value is read at **build/start time** by `next.config.js` to configure
the `/api/*` rewrite — change it and restart the dev server (or rebuild).

## API Integration

The typed client lives in `lib/api.ts` and talks to two endpoints:

### `GET /api/models`

Returns the model registry:

```json
{
  "default": "base",
  "models": [
    { "id": "tiny",  "loaded": true, "params": "372K",  "description": "Fastest" },
    { "id": "base",  "loaded": true, "params": "1.62M", "description": "Balanced" },
    { "id": "large", "loaded": true, "params": "6.30M", "description": "Most accurate" }
  ]
}
```

### `POST /api/predict?model=<id>`

Multipart form upload (`file` field). Returns:

```json
{
  "prediction": "ai_generated",
  "confidence": 0.97,
  "real_probability": 0.03,
  "ai_probability": 0.97,
  "processing_time_ms": 412,
  "anomaly_heatmap": "<base64 PNG>",
  "frequency_band_contributions": { "low": 0.21, "mid": 0.33, "high": 0.46 }
}
```

Error responses surface their `detail` message directly in the UI.

## Available Models

| Model | Parameters | Profile |
|---|---|---|
| MFFT-Tiny | 372K | Fastest inference, edge-friendly |
| MFFT-Base | 1.62M | Balanced speed/accuracy (default) |
| MFFT-Large | 6.30M | Highest accuracy |

> **Note:** the public API currently serves *pilot* demo checkpoints
> (1 training epoch) while the full-scale training campaign completes.
> Verdict quality will improve when the final weights ship; the interface
> and response contract stay the same.

## Theming

The UI implements a "vintage classic" dual theme:

- **Light** — warm cream paper tones (`cream-50…300`) with stone text
- **Dark** — espresso/aged-leather neutrals (a warm remap of the Tailwind
  `slate` scale in `tailwind.config.js`) with amber-tinted glows

Theme selection is applied by an inline script in `app/layout.tsx` **before
first paint** (no flash of the wrong theme), persisted to `localStorage`,
and defaults to the OS preference via `prefers-color-scheme`.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start the dev server at `http://localhost:3000` |
| `npm run build` | Production build (static prerender + optimization) |
| `npm run start` | Serve the production build |
| `npm run lint` | Run Next.js ESLint checks |

## Deployment

### Vercel (recommended)

1. Import this repository in Vercel.
2. Add the environment variable
   `NEXT_PUBLIC_API_URL=https://mohsineli-mfft-api.hf.space`.
3. Deploy — no other configuration needed.

### Any Node host

```bash
npm ci
NEXT_PUBLIC_API_URL=https://your-api.example.com npm run build
npm run start   # listens on :3000
```

Works behind any reverse proxy (nginx, Caddy, etc.); all API traffic is
already same-origin.

## Browser Support

Modern evergreen browsers (Chrome, Edge, Firefox, Safari). The upload flow
uses standard `FormData`/`fetch`; no polyfills required.

## Roadmap

- [ ] Full-scale MFFT checkpoints on the public API
- [ ] Batch analysis (multiple images per request)
- [ ] Shareable, permalinked analysis reports
- [ ] API keys and rate-limit tiers to match the pricing page

## Related Work

This front-end is part of a broader research project: *MFFT — Multi-Frequency
Fusion Transformer for AI-Generated Image Detection* (paper in preparation).
The detection API is a FastAPI service deployed to Hugging Face Spaces.

## Author

**MIH Mahmud Eli** — [github.com/MIHMahmudEli](https://github.com/MIHMahmudEli)

Issues and pull requests are welcome.
