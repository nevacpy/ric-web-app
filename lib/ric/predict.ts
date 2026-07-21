import type { RicPredictResponse } from "./types";

/**
 * Upload an image to the Next.js `/api/predict` endpoint.
 * Works from the web app and from Expo (pass your Vercel origin as `apiBaseUrl`).
 */
export async function predictRic(
  file: Blob | File,
  options?: {
    apiBaseUrl?: string;
    filename?: string;
    signal?: AbortSignal;
  },
): Promise<RicPredictResponse> {
  const formData = new FormData();
  const filename = options?.filename ?? "bottle.jpg";
  formData.append("file", file, filename);

  const base = (options?.apiBaseUrl ?? "").replace(/\/$/, "");
  const url = `${base}/api/predict`;

  console.log('url', url);
  const response = await fetch(url, {
    method: "POST",
    body: formData,
    credentials: "omit",
    signal: options?.signal,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return {
      success: false,
      error: text || `HTTP ${response.status}`,
    };
  }

  return (await response.json()) as RicPredictResponse;
}
