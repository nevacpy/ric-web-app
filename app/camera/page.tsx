import Link from "next/link";
import { RicCameraDetector } from "@/components/RicCameraDetector";

export default function CameraPage() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-[radial-gradient(circle_at_top_left,rgba(15,139,76,0.12),transparent_32%),linear-gradient(180deg,#f7fbf7_0%,#eef5ef_100%)]">
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 py-12">
        <header className="flex flex-col gap-3">
          <p className="text-sm font-medium uppercase tracking-[0.18em] text-emerald-800/70">
            HK Smart Plastic Sorter
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-emerald-950">
            Camera detection
          </h1>
          <p className="max-w-2xl text-base leading-7 text-emerald-950/70">
            Point the camera at a bottle label and tap Detect RIC. The frame is
            sent to{" "}
            <code className="rounded bg-white/70 px-1.5 py-0.5 font-mono text-sm">
              /api/predict
            </code>
            , and the suggestion is shown as a toast.
          </p>
          <Link
            href="/"
            className="w-fit text-sm font-medium text-emerald-800 underline-offset-4 hover:underline"
          >
            ← Back to upload
          </Link>
        </header>

        <RicCameraDetector />
      </main>
    </div>
  );
}
