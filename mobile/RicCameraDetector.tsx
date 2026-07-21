/**
 * Expo / React Native camera component for RIC symbol detection.
 *
 * This file is intentionally kept under `mobile/` and excluded from the Next.js
 * TypeScript project — it depends on `expo-camera` and `react-native`.
 *
 * Copy into your Expo app (or import from a shared workspace package) and set
 * `apiBaseUrl` to your Vercel deployment, e.g. `https://your-app.vercel.app`.
 *
 * Required Expo deps: expo-camera, react-native
 */

import { CameraView, useCameraPermissions } from "expo-camera";
import { useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Button,
  Platform,
  StyleSheet,
  Text,
  ToastAndroid,
  TouchableOpacity,
  View,
} from "react-native";

export type RicDetection = {
  class: string;
  confidence: number;
  bbox: number[];
  model?: string;
};

export type RicPredictResponse =
  | {
      success: true;
      inference_speed_ms: number;
      suggestion_result: RicDetection | null;
      detections: Record<string, RicDetection | null>;
    }
  | { success: false; error: string };

export type RicCameraDetectorProps = {
  /** Origin of the Next.js app, e.g. https://fyp-web-app.vercel.app */
  apiBaseUrl: string;
  onResult?: (result: RicPredictResponse) => void;
};

function showToast(message: string) {
  if (Platform.OS === "android") {
    ToastAndroid.show(message, ToastAndroid.SHORT);
  } else {
    Alert.alert("", message);
  }
}

export function RicCameraDetector({
  apiBaseUrl,
  onResult,
}: RicCameraDetectorProps) {
  const [facing, setFacing] = useState<"back" | "front">("back");
  const [busy, setBusy] = useState(false);
  const [lastSuggestion, setLastSuggestion] = useState<RicDetection | null>(
    null,
  );
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required to detect RIC symbols.
        </Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  async function takePicture() {
    if (!cameraRef.current || busy) return;

    setBusy(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        skipProcessing: false,
      });

      if (!photo?.uri) {
        throw new Error("No photo URI returned from camera");
      }

      const formData = new FormData();

      if (Platform.OS === "web") {
        const blobResponse = await fetch(photo.uri);
        const blob = await blobResponse.blob();
        formData.append("file", blob, "bottle.jpg");
      } else {
        // React Native multipart file descriptor
        formData.append("file", {
          uri: photo.uri,
          name: "bottle.jpg",
          type: "image/jpeg",
        } as unknown as Blob);
      }

      const base = apiBaseUrl.replace(/\/$/, "");
      const response = await fetch(`${base}/api/predict`, {
        method: "POST",
        body: formData,
        credentials: "omit",
      });

      const data = (await response.json()) as RicPredictResponse;
      onResult?.(data);

      if (data.success) {
        setLastSuggestion(data.suggestion_result);
        const label = data.suggestion_result
          ? `${data.suggestion_result.class} (${Math.round(
              data.suggestion_result.confidence * 100,
            )}%)`
          : "No RIC symbol detected";
        showToast(label);
      } else {
        showToast(`Detection failed: ${data.error}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const failure: RicPredictResponse = {
        success: false,
        error: message,
      };
      onResult?.(failure);
      showToast(`API error: ${message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
        <View style={styles.overlay}>
          {lastSuggestion ? (
            <View style={styles.banner}>
              <Text style={styles.bannerText}>
                Suggested: {lastSuggestion.class} ·{" "}
                {Math.round(lastSuggestion.confidence * 100)}%
              </Text>
            </View>
          ) : null}

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.button}
              onPress={toggleCameraFacing}
              disabled={busy}
            >
              <Text style={styles.buttonText}>Flip</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, styles.snapButton]}
              onPress={takePicture}
              disabled={busy}
            >
              {busy ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.buttonText}>Detect RIC</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    justifyContent: "space-between",
    padding: 24,
  },
  banner: {
    alignSelf: "center",
    marginTop: 48,
    backgroundColor: "rgba(10, 107, 59, 0.88)",
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
  },
  bannerText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  buttonRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 32,
  },
  button: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingVertical: 14,
    borderRadius: 12,
  },
  snapButton: {
    backgroundColor: "rgba(15, 139, 76, 0.9)",
  },
  buttonText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#fff",
  },
  permissionText: {
    textAlign: "center",
    color: "#fff",
    marginBottom: 16,
    paddingHorizontal: 24,
  },
});

export default RicCameraDetector;
