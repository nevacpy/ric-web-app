"use client";

import { useState } from "react";
import { predictRic } from "@/lib/ric/predict";
import type { RicDetection, RicPredictResponse } from "@/lib/ric/types";

function formatDetection(det: RicDetection) {
  const pct = Math.round(det.confidence * 100);
  const model = det.model ? ` · ${det.model}` : "";
  return `${det.class} (${pct}%)${model}`;
}

export function RicUploadDetector() {
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<RicPredictResponse | null>(null);

  async function onFileChange(file: File | null) {
    if (!file) return;

    setBusy(true);
    setResult(null);

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));

    try {
      const data = await predictRic(file, { filename: file.name || "bottle.jpg" });
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex w-full max-w-xl flex-col gap-6">
      <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-emerald-700/30 bg-emerald-50/60 px-6 py-10 text-center transition hover:bg-emerald-50">
        <span className="text-lg font-semibold text-emerald-950">
          Upload or capture a bottle label
        </span>
        <span className="text-sm text-emerald-900/70">
          Sends the image to <code className="font-mono">/api/predict</code>
        </span>
        <input
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          disabled={busy}
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
      </label>

      {previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl}
          alt="Uploaded preview"
          className="max-h-72 w-full rounded-2xl object-contain bg-zinc-100"
        />
      ) : null}

      {busy ? (
        <p className="text-sm text-zinc-600">Running dual-model RIC detection…</p>
      ) : null}

      {result?.success === false ? (
        <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-800">
          {result.error}
        </p>
      ) : null}

      {result?.success === true ? (
        <div className="flex flex-col gap-3 rounded-2xl bg-white px-4 py-4 shadow-sm ring-1 ring-zinc-200">
          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-500">
              Suggestion
            </p>
            <p className="text-xl font-semibold text-emerald-900">
              {result.suggestion_result
                ? formatDetection(result.suggestion_result)
                : "No RIC symbol detected"}
            </p>
          </div>

          <div>
            <p className="mb-2 text-xs uppercase tracking-wide text-zinc-500">
              Per-model detections
            </p>
            <ul className="space-y-1 text-sm text-zinc-800">
              {Object.entries(result.detections).map(([model, det]) => (
                <li key={model}>
                  <span className="font-mono text-zinc-500">{model}:</span>{" "}
                  {det ? formatDetection(det) : "null"}
                </li>
              ))}
            </ul>
          </div>

          <p className="text-xs text-zinc-500">
            Inference: {result.inference_speed_ms.toFixed(1)} ms
          </p>
        </div>
      ) : null}
    </section>
  );
}
