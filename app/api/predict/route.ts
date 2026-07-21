import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
/** Allow enough time for dual-model YOLO inference behind the proxy. */
export const maxDuration = 60;

const CORS_HEADERS: HeadersInit = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function jsonWithCors(body: unknown, init?: { status?: number }) {
  return NextResponse.json(body, {
    status: init?.status ?? 200,
    headers: CORS_HEADERS,
  });
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS });
}

/**
 * Public predict endpoint for the web app + Expo client.
 *
 * Forwards multipart uploads to the Python FastAPI service (`ric-api/main.py`),
 * which runs the YOLO models. Set `RIC_API_URL` to that service's origin
 * (e.g. `http://127.0.0.1:8000` locally, or your hosted ML backend on Vercel).
 */
export async function POST(request: NextRequest) {
  const ricApiUrl = process.env.RIC_API_URL?.replace(/\/$/, "");

  if (!ricApiUrl) {
    return jsonWithCors(
      {
        success: false,
        error:
          "RIC_API_URL is not configured. Point it at the FastAPI ric-api service (e.g. http://127.0.0.1:8000).",
      },
      { status: 500 },
    );
  }

  try {
    const incoming = await request.formData();
    const file = incoming.get("file");
    console.log('file', file)

    if (!file || !(file instanceof Blob)) {
      return jsonWithCors(
        { success: false, error: "Missing image file. Send multipart field `file`." },
        { status: 400 },
      );
    }

    const outbound = new FormData();
    const filename =
      typeof File !== "undefined" && file instanceof File && file.name
        ? file.name
        : "bottle.jpg";
    outbound.append("file", file, filename);

    // const blobResponse = await fetch(photo.uri);
    // const blob = await blobResponse.blob();
    // formData.append("file", blob, "bottle.jpg");

    const upstream = await fetch(`${ricApiUrl}/predict`, {
      method: "POST",
      body: outbound,
      // Do not set Content-Type — fetch will set the multipart boundary.
    });

    const payload = await upstream.json().catch(async () => ({
      success: false,
      error: await upstream.text(),
    }));

    return jsonWithCors(payload, { status: upstream.ok ? 200 : upstream.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return jsonWithCors(
      {
        success: false,
        error: `Failed to reach RIC API at ${ricApiUrl}: ${message}`,
      },
      { status: 502 },
    );
  }
}
