export type RicDetection = {
  class: string;
  confidence: number;
  bbox: [number, number, number, number] | number[];
  model?: string;
};

/** Per-model result: detection object, or null when that model found nothing. */
export type RicDetectionsByModel = Record<string, RicDetection | null>;

export type RicPredictSuccess = {
  success: true;
  inference_speed_ms: number;
  suggestion_result: RicDetection | null;
  detections: RicDetectionsByModel;
};

export type RicPredictFailure = {
  success: false;
  error: string;
};

export type RicPredictResponse = RicPredictSuccess | RicPredictFailure;
