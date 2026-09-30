import type { ModelOption } from "./types";

export const FALLBACK_MODELS: ModelOption[] = [
  { id: "tiny", loaded: true, params: "372K", description: "Fastest — edge & mobile profile" },
  { id: "base", loaded: true, params: "1.62M", description: "Balanced accuracy and speed" },
  { id: "large", loaded: true, params: "6.30M", description: "Highest accuracy profile" },
];

export const HERO_STATS = [
  { value: "3", label: "Frequency Bands Analyzed" },
  { value: "372K–6.3M", label: "Model Sizes (Parameters)" },
  { value: "<100ms", label: "Inference per Image" },
];

export const HOW_IT_WORKS_STEPS = [
  {
    step: "01",
    title: "Frequency Decomposition",
    desc: "The image is split into low, mid, and high frequency bands via Fourier radial filtering — where AI generators leave their fingerprints.",
  },
  {
    step: "02",
    title: "Per-Band Feature Extraction",
    desc: "Each band is independently analyzed by a dedicated lightweight CNN, specializing in that band's artifact patterns.",
  },
  {
    step: "03",
    title: "Cross-Attention Fusion",
    desc: "Band features are fused via multi-head cross-attention with frequency-guided weighting to produce an explainable verdict.",
  },
];

export const PRICING_PLANS = [
  {
    name: "Free",
    price: "$0",
    period: "forever",
    features: [
      "10 detections / min",
      "Single image upload",
      "All three model sizes",
      "Community support",
    ],
    cta: "Get Started",
    featured: false,
  },
  {
    name: "Pro",
    price: "$9.99",
    period: "/month",
    features: [
      "100 detections / min",
      "Batch upload (up to 10)",
      "Anomaly heatmaps",
      "Detailed PDF reports",
      "API access",
      "Priority support",
    ],
    cta: "Start Free Trial",
    featured: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "",
    features: [
      "Unlimited detections",
      "On-premise deployment",
      "Custom model fine-tuning",
      "SLA guarantees",
      "Dedicated account manager",
      "White-label option",
    ],
    cta: "Contact Sales",
    featured: false,
  },
];

export const LINKS = {
  api: "https://mohsinelis-mfft-api.hf.space",
  space: "https://huggingface.co/spaces/MohsinElis/mfft-api",
  github: "https://github.com/MIHMahmudEli/AI-ImageVerify-",
};
