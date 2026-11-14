export type PredictionResult = {
  prediction: string;
  confidence: number;
  real_probability: number;
  ai_probability: number;
  processing_time_ms: number;
  anomaly_heatmap?: string;
  frequency_band_contributions?: Record<string, number>;
};

export type ModelOption = {
  id: string;
  loaded: boolean;
  params: string;
  description: string;
};
