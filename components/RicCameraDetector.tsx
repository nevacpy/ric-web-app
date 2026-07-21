"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { predictRic } from "@/lib/ric/predict";
import type { RicDetection } from "@/lib/ric/types";

function formatSuggestion(det: RicDetection | null) {
  if (!det) return "No RIC symbol detected";
  return `${det.class} (${Math.round(det.confidence * 100)}%)`;
}

export function RicCameraDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [facingMode, setFacingMode] = useState<"environment" | "user">(
    "environment",
  );
  const [restartKey, setRestartKey] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [permission, setPermission] = useState<
    "pending" | "granted" | "denied"
  >("pending");
  const [busy, setBusy] = useState(false);
  const [lastSuggestion, setLastSuggestion] = useState<RicDetection | null>(
    null,
  );

  // Acquire the camera stream. Attachment happens in a separate effect once
  // the <video> is mounted (it was previously missing while permission was
  // "pending", so the stream never got connected → black preview).
  useEffect(() => {
    let cancelled = false;
    let activeStream: MediaStream | null = null;

    async function startCamera() {
      setPermission("pending");
      setStream(null);

      try {
        const media = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });

        if (cancelled) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }

        activeStream = media;
        setStream(media);
        setPermission("granted");
      } catch {
        if (!cancelled) {
          setStream(null);
          setPermission("denied");
        }
      }
    }

    void startCamera();

    return () => {
      cancelled = true;
      activeStream?.getTracks().forEach((track) => track.stop());
      setStream(null);
    };
  }, [facingMode, restartKey]);

  // Attach stream after React has mounted the <video> element.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;
    void video.play().catch(() => {
      video.muted = true;
      void video.play();
    });

    return () => {
      video.srcObject = null;
    };
  }, [stream]);

  function requestPermission() {
    setRestartKey((key) => key + 1);
  }

  function flipCamera() {
    setFacingMode((current) =>
      current === "environment" ? "user" : "environment",
    );
  }

  async function takePicture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || busy) return;

    if (video.readyState < 2 || video.videoWidth === 0) {
      toast.error("Camera is not ready yet — wait for the preview.");
      return;
    }

    setBusy(true);
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (result) =>
            result ? resolve(result) : reject(new Error("Capture failed")),
          "image/jpeg",
          0.8,
        );
      });

      const data = await predictRic(blob, { filename: "bottle.jpg" });

      if (data.success) {
        setLastSuggestion(data.suggestion_result);
        toast.success(formatSuggestion(data.suggestion_result));
      } else {
        toast.error(`Detection failed: ${data.error}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error(`API error: ${message}`);
    } finally {
      setBusy(false);
    }
  }

  if (permission === "denied") {
    return (
      <div className="flex min-h-[28rem] flex-col items-center justify-center gap-4 rounded-2xl bg-zinc-950 px-6 text-center">
        <p className="max-w-sm text-sm text-zinc-300">
          Camera permission is required to detect RIC symbols.
        </p>
        <button
          type="button"
          onClick={requestPermission}
          className="rounded-xl bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-600"
        >
          Grant permission
        </button>
      </div>
    );
  }

  return (
    <section className="relative overflow-hidden rounded-2xl bg-black shadow-sm ring-1 ring-zinc-200">
      {permission === "pending" ? (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-zinc-950 text-sm text-zinc-300">
          Requesting camera access…
        </div>
      ) : null}

      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className="aspect-[3/4] w-full bg-black object-cover sm:aspect-video"
      />
      <canvas ref={canvasRef} className="hidden" />

      {lastSuggestion ? (
        <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded-xl bg-emerald-800/90 px-4 py-2 text-sm font-semibold text-white shadow">
          Suggested: {lastSuggestion.class} ·{" "}
          {Math.round(lastSuggestion.confidence * 100)}%
        </div>
      ) : null}

      <div className="absolute inset-x-0 bottom-0 z-10 flex gap-3 bg-gradient-to-t from-black/70 to-transparent p-4 pt-10">
        <button
          type="button"
          onClick={flipCamera}
          disabled={busy || permission !== "granted"}
          className="flex-1 rounded-xl bg-black/55 px-4 py-3 text-sm font-bold text-white disabled:opacity-50"
        >
          Flip
        </button>
        <button
          type="button"
          onClick={takePicture}
          disabled={busy || permission !== "granted"}
          className="flex-[1.4] rounded-xl bg-emerald-700/95 px-4 py-3 text-sm font-bold text-white disabled:opacity-50"
        >
          {busy ? "Detecting…" : "Detect RIC"}
        </button>
      </div>
    </section>
  );
}
