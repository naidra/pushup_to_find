import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Sparkles, Sun, Moon, ShieldCheck, Activity, RotateCcw, X } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";

type RepPhase = "calibrating" | "down" | "up";
type DecryptFromHex = (encoded: string, key: string) => string;
type EncryptDecryptModule = {
  cwrap?: (name: string, returnType: "string", argTypes: ["string", "string"]) => DecryptFromHex;
  locateFile?: (path: string) => string;
  onRuntimeInitialized?: () => void;
  onAbort?: (reason: unknown) => void;
};

declare global {
  interface Window {
    Module?: EncryptDecryptModule;
    __encryptDecryptRuntime?: Promise<string>;
    __encryptDecryptRuntimeKey?: string;
  }
}

type RepSignalTracker = {
  phase: RepPhase;
  min: number | null;
  max: number | null;
  smoothed: number | null;
  samples: { value: number; t: number }[];
  lastRisingAt: number;
};

type PushUpTracker = {
  count: number;
  phase: RepPhase;
  elbowExtend: RepSignalTracker;
  lastRepAt: number;
};

const MEDIAPIPE_ASSET_BASE = `${import.meta.env.BASE_URL}mediapipe`;
const WASM_BASE = `${MEDIAPIPE_ASSET_BASE}/wasm`;
const POSE_MODEL_PATH = `${MEDIAPIPE_ASSET_BASE}/models/pose_landmarker_full.task`;

const VISIBLE_LANDMARK = 0.45;
const REP_BODY_LANDMARK_MIN_VISIBILITY = 0.12;
const REP_ANGLE_RANGE_MIN = 0.12;
const REP_SAMPLE_WINDOW_MS = 4500;
const REP_SMOOTHING = 0.44;
const REP_OUTLIER_JUMP = 0.32;
const REP_DOWN_THRESHOLD = 0.32;
const REP_UP_THRESHOLD = 0.68;
const REP_RISING_LOOKBACK_MS = 140;
const REP_RISING_DELTA_MIN = 0.003;
const REP_RISING_GRACE_MS = 260;
const PUSHUP_RANGE_MIN = 0.1;
const PUSHUP_POSTURE_MAX_VERTICAL_SPREAD = 0.36;
const PUSHUP_POSTURE_MIN_HORIZONTAL_SPREAD = 0.18;
const PUSHUP_COOLDOWN_MS = 500;
const REVEAL_PUSHUP_COUNT = 100;
const REVEAL_ENCRYPTION_KEY = "flawless";
const ENCRYPTED_REVEAL_URL =
  "0E1815071F5F5C5C05030C0703101D174B0400150511005D0902131202011601480F0E1A";
const ENCRYPT_DECRYPT_ASSET_BASE = `${import.meta.env.BASE_URL}encrypt_decrypt`;
const ENCRYPT_DECRYPT_RUNTIME_KEY = `encrypt-decrypt-public-v2:${ENCRYPT_DECRYPT_ASSET_BASE}`;

const loadDecryptRuntime = () => {
  if (window.__encryptDecryptRuntimeKey !== ENCRYPT_DECRYPT_RUNTIME_KEY) {
    window.__encryptDecryptRuntime = undefined;
    window.__encryptDecryptRuntimeKey = ENCRYPT_DECRYPT_RUNTIME_KEY;
  }

  if (window.__encryptDecryptRuntime) return window.__encryptDecryptRuntime;

  window.__encryptDecryptRuntime = new Promise((resolve, reject) => {
    const previousModule = window.Module;
    let settled = false;
    const timeoutId = window.setTimeout(() => {
      fail(new Error("Decrypt runtime timed out."));
    }, 6000);
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      window.Module = previousModule;
      window.__encryptDecryptRuntime = undefined;
      reject(error);
    };
    const moduleConfig: EncryptDecryptModule = {
      locateFile: (path) =>
        path.endsWith(".wasm") ? `${ENCRYPT_DECRYPT_ASSET_BASE}/encrypt_decrypt.wasm` : path,
      onRuntimeInitialized: () => {
        if (settled) return;
        const decryptFromHex = moduleConfig.cwrap!("decrypt_from_hex", "string", [
          "string",
          "string",
        ]);
        const decodedUrl = decryptFromHex(ENCRYPTED_REVEAL_URL, REVEAL_ENCRYPTION_KEY);
        settled = true;
        window.clearTimeout(timeoutId);
        window.setTimeout(() => {
          window.Module = previousModule;
        }, 0);
        resolve(decodedUrl);
      },
      onAbort: (reason) => {
        fail(new Error(`Decrypt runtime aborted: ${String(reason)}`));
      },
    };

    window.Module = moduleConfig;

    const script = document.createElement("script");
    script.src = `${ENCRYPT_DECRYPT_ASSET_BASE}/encrypt_decrypt.js`;
    script.async = true;
    script.onerror = () => {
      fail(new Error("Unable to load decrypt runtime."));
    };
    document.body.appendChild(script);
  });

  return window.__encryptDecryptRuntime;
};

const getAverageWithMinVisibility = (
  landmarks: NormalizedLandmark[],
  indices: number[],
  minVisibility: number,
) => {
  const points = indices
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 0) >= minVisibility);

  if (!points.length) return null;

  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
};

const createRepSignalTracker = (): RepSignalTracker => ({
  phase: "calibrating",
  min: null,
  max: null,
  smoothed: null,
  samples: [],
  lastRisingAt: 0,
});

const percentile = (values: number[], ratio: number) => {
  if (!values.length) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * ratio)));
  return sorted[index];
};

const getAngle = (a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) => {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);

  if (!mag) return null;

  const cosine = Math.max(-1, Math.min(1, dot / mag));
  return (Math.acos(cosine) * 180) / Math.PI;
};

const getAverageElbowAngle = (landmarks: NormalizedLandmark[]) => {
  const sides = [
    [11, 13, 15],
    [12, 14, 16],
  ] as const;
  const angles = sides
    .map(([shoulderIndex, elbowIndex, wristIndex]) => {
      const shoulder = landmarks[shoulderIndex];
      const elbow = landmarks[elbowIndex];
      const wrist = landmarks[wristIndex];

      if (
        !shoulder ||
        !elbow ||
        !wrist ||
        (shoulder.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY ||
        (elbow.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY ||
        (wrist.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY
      ) {
        return null;
      }

      return getAngle(shoulder, elbow, wrist);
    })
    .filter((angle): angle is number => angle != null);

  if (!angles.length) return null;

  return angles.reduce((sum, angle) => sum + angle, 0) / angles.length;
};

const getPushUpPosture = (landmarks: NormalizedLandmark[]) => {
  const keyIndices = [11, 12, 23, 24, 25, 26, 27, 28];
  const points = keyIndices
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 0) >= REP_BODY_LANDMARK_MIN_VISIBILITY);
  const shoulders = getAverageWithMinVisibility(
    landmarks,
    [11, 12],
    REP_BODY_LANDMARK_MIN_VISIBILITY,
  );
  const hips = getAverageWithMinVisibility(landmarks, [23, 24], REP_BODY_LANDMARK_MIN_VISIBILITY);

  if (!shoulders || !hips || points.length < 4) return false;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const horizontalSpread = Math.max(...xs) - Math.min(...xs);
  const verticalSpread = Math.max(...ys) - Math.min(...ys);
  const torsoVerticalGap = Math.abs(shoulders.y - hips.y);

  return (
    horizontalSpread >= PUSHUP_POSTURE_MIN_HORIZONTAL_SPREAD &&
    verticalSpread <= PUSHUP_POSTURE_MAX_VERTICAL_SPREAD &&
    torsoVerticalGap <= PUSHUP_POSTURE_MAX_VERTICAL_SPREAD * 0.55
  );
};

const updateRepSignal = (
  signal: RepSignalTracker,
  value: number | null,
  now: number,
  minRange: number,
) => {
  if (value == null) {
    return { ready: false, progress: 0, reachedTop: false, rising: false };
  }

  const lastValue = signal.smoothed ?? value;
  const rawValue = Math.abs(value - lastValue) > REP_OUTLIER_JUMP ? lastValue : value;
  const smoothed =
    signal.smoothed == null
      ? rawValue
      : signal.smoothed + (rawValue - signal.smoothed) * REP_SMOOTHING;

  signal.smoothed = smoothed;
  signal.samples = [...signal.samples, { value: smoothed, t: now }].filter(
    (sample) => now - sample.t <= REP_SAMPLE_WINDOW_MS,
  );
  const lookbackSample = [...signal.samples]
    .reverse()
    .find((sample) => now - sample.t >= REP_RISING_LOOKBACK_MS);
  const rising = lookbackSample
    ? smoothed - lookbackSample.value > REP_RISING_DELTA_MIN
    : smoothed - lastValue > REP_RISING_DELTA_MIN;
  if (rising) {
    signal.lastRisingAt = now;
  }

  const values = signal.samples.map((sample) => sample.value);
  const sampleMin = percentile(values, 0.08);
  const sampleMax = percentile(values, 0.92);

  if (sampleMin != null && sampleMax != null) {
    signal.min = signal.min == null ? sampleMin : signal.min * 0.8 + sampleMin * 0.2;
    signal.max = signal.max == null ? sampleMax : signal.max * 0.8 + sampleMax * 0.2;
  }

  const range = Math.max(0, (signal.max ?? smoothed) - (signal.min ?? smoothed));
  const ready = range > minRange;
  const progress = ready
    ? Math.max(0, Math.min(1, (smoothed - (signal.min ?? smoothed)) / range))
    : 0;

  if (!ready) {
    return { ready, progress, reachedTop: false, rising };
  }

  if (progress <= REP_DOWN_THRESHOLD) {
    signal.phase = "down";
  } else if (signal.phase === "calibrating") {
    signal.phase = progress > 0.5 ? "up" : "down";
  }

  return {
    ready,
    progress,
    rising,
    reachedTop: progress >= REP_UP_THRESHOLD && signal.phase === "down",
  };
};

const estimatePushUp = (poseRes: PoseLandmarkerResult | null, tracker: PushUpTracker) => {
  const pose = poseRes?.landmarks[0];
  if (!pose) {
    return {
      pushUpCount: tracker.count,
      pushUpPhase: tracker.phase,
      poseDetected: false,
      pushUpReady: false,
      pushUpProgress: 0,
    };
  }

  const now = performance.now();
  const inPushUpPosture = getPushUpPosture(pose);
  const elbowAngle = getAverageElbowAngle(pose);
  const elbowExtend = updateRepSignal(
    tracker.elbowExtend,
    inPushUpPosture && elbowAngle != null ? elbowAngle / 180 : null,
    now,
    PUSHUP_RANGE_MIN,
  );
  const rising = now - tracker.elbowExtend.lastRisingAt <= REP_RISING_GRACE_MS;
  const reachedTop = elbowExtend.reachedTop && rising;

  if (reachedTop && now - tracker.lastRepAt > PUSHUP_COOLDOWN_MS) {
    tracker.count += 1;
    tracker.phase = "up";
    tracker.elbowExtend.phase = "up";
    tracker.lastRepAt = now;
  } else if (elbowExtend.phase === "down") {
    tracker.phase = "down";
  } else if (tracker.phase === "calibrating" && elbowExtend.ready) {
    tracker.phase = elbowExtend.progress > 0.5 ? "up" : "down";
  }

  return {
    pushUpCount: tracker.count,
    pushUpPhase: tracker.phase,
    poseDetected: true,
    pushUpReady: inPushUpPosture && elbowExtend.ready,
    pushUpProgress: inPushUpPosture ? elbowExtend.progress : 0,
  };
};

export function VisionApp() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const lastPushUpSoundCountRef = useRef(0);
  const revealStartedRef = useRef(false);
  const pushUpTrackerRef = useRef<PushUpTracker>({
    count: 0,
    phase: "calibrating",
    elbowExtend: createRepSignalTracker(),
    lastRepAt: 0,
  });

  const [loading, setLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState("Initializing WASM runtime…");
  const [revealedUrl, setRevealedUrl] = useState("");
  const [revealOpen, setRevealOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState({
    poseDetected: false,
    pushUpCount: 0,
    pushUpPhase: "calibrating" as RepPhase,
    pushUpReady: false,
    pushUpProgress: 0,
    fps: 0,
  });
  const fpsRef = useRef({ frames: 0, t0: performance.now() });

  // Load the pose model once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoadProgress("Loading WASM runtime…");
        const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
        if (cancelled) return;

        setLoadProgress("Loading pose landmarker…");
        const poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: POSE_MODEL_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.35,
          minPosePresenceConfidence: 0.35,
          minTrackingConfidence: 0.35,
        });

        if (cancelled) {
          poseLandmarker.close();
          return;
        }

        poseLandmarkerRef.current = poseLandmarker;
        setLoading(false);
      } catch (err) {
        console.error(err);
        setLoadProgress(`Failed to load models: ${(err as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      poseLandmarkerRef.current?.close();
      audioContextRef.current?.close();
    };
  }, []);

  const stopLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const stopCameraStream = useCallback(() => {
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
  }, []);

  const resetRepTracker = useCallback(() => {
    lastPushUpSoundCountRef.current = 0;
    pushUpTrackerRef.current = {
      count: 0,
      phase: "calibrating",
      elbowExtend: createRepSignalTracker(),
      lastRepAt: 0,
    };
    setStats((current) => ({
      ...current,
      poseDetected: false,
      pushUpCount: 0,
      pushUpPhase: "calibrating",
      pushUpReady: false,
      pushUpProgress: 0,
    }));
  }, []);

  useEffect(() => {
    if (stats.pushUpCount < REVEAL_PUSHUP_COUNT || revealStartedRef.current) return;

    revealStartedRef.current = true;
    void loadDecryptRuntime()
      .then((decodedUrl) => {
        setRevealedUrl(decodedUrl);
        setRevealOpen(true);
        setCopied(false);
      })
      .catch((error) => {
        console.error(error);
        revealStartedRef.current = false;
      });
  }, [stats.pushUpCount]);

  const copyRevealedUrl = useCallback(async () => {
    if (!revealedUrl) return;

    if (navigator.clipboard) {
      await navigator.clipboard.writeText(revealedUrl);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = revealedUrl;
      textarea.setAttribute("readonly", "");
      textarea.className = "fixed -left-[9999px] top-0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    setCopied(true);
  }, [revealedUrl]);

  const playRepSound = useCallback(() => {
    const AudioContextConstructor =
      window.AudioContext ??
      (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;

    if (!AudioContextConstructor) return;

    const audioContext = audioContextRef.current ?? new AudioContextConstructor();
    audioContextRef.current = audioContext;

    if (audioContext.state === "suspended") {
      void audioContext.resume();
    }

    const now = audioContext.currentTime;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(620, now);
    oscillator.frequency.exponentialRampToValueAtTime(440, now + 0.12);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.315, now + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);

    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.15);
  }, []);

  const draw = useCallback(
    (sw: number, sh: number, poseRes: PoseLandmarkerResult | null) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Match canvas to source intrinsic size for crisp drawing
      if (canvas.width !== sw || canvas.height !== sh) {
        canvas.width = sw;
        canvas.height = sh;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, sw, sh);

      if (poseRes && poseRes.landmarks.length > 0) {
        ctx.strokeStyle = "oklch(0.76 0.17 75)";
        ctx.lineWidth = Math.max(3, sw / 320);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        for (const pose of poseRes.landmarks) {
          for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
            const start = pose[connection.start];
            const end = pose[connection.end];
            if (
              !start ||
              !end ||
              start.visibility < VISIBLE_LANDMARK ||
              end.visibility < VISIBLE_LANDMARK
            ) {
              continue;
            }
            ctx.beginPath();
            ctx.moveTo(start.x * sw, start.y * sh);
            ctx.lineTo(end.x * sw, end.y * sh);
            ctx.stroke();
          }

          ctx.fillStyle = "oklch(0.82 0.18 75)";
          for (const point of pose) {
            if (point.visibility < VISIBLE_LANDMARK) continue;
            ctx.beginPath();
            ctx.arc(point.x * sw, point.y * sh, Math.max(3, sw / 240), 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      const pushUps = estimatePushUp(poseRes, pushUpTrackerRef.current);
      if (pushUps.pushUpCount > lastPushUpSoundCountRef.current) {
        lastPushUpSoundCountRef.current = pushUps.pushUpCount;
        playRepSound();
      }

      // FPS
      const f = fpsRef.current;
      f.frames++;
      const now = performance.now();
      let fps = stats.fps;
      if (now - f.t0 > 500) {
        fps = Math.round((f.frames * 1000) / (now - f.t0));
        f.frames = 0;
        f.t0 = now;
      }

      setStats({
        poseDetected: pushUps.poseDetected,
        pushUpCount: pushUps.pushUpCount,
        pushUpPhase: pushUps.pushUpPhase,
        pushUpReady: pushUps.pushUpReady,
        pushUpProgress: pushUps.pushUpProgress,
        fps,
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [playRepSound],
  );

  const runVideoLoop = useCallback(() => {
    const video = videoRef.current;
    const pl = poseLandmarkerRef.current;
    if (!video || !pl) return;

    const tick = () => {
      if (video.readyState >= 2 && !video.paused && !video.ended) {
        const ts = performance.now();
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const poseRes = pl.detectForVideo(video, ts);
          draw(video.videoWidth, video.videoHeight, poseRes);
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [draw]);

  // Start camera
  const startCamera = useCallback(async () => {
    stopLoop();
    stopCameraStream();
    resetRepTracker();
    try {
      await poseLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" });
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const v = videoRef.current!;
      v.srcObject = stream;
      v.muted = true;
      await v.play();
      runVideoLoop();
    } catch (err) {
      console.error("Camera error", err);
      alert("Could not access camera: " + (err as Error).message);
    }
  }, [resetRepTracker, runVideoLoop, stopCameraStream, stopLoop]);

  // Auto-start camera when models ready
  useEffect(() => {
    if (!loading && !videoRef.current?.srcObject) {
      startCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  const { theme, toggle } = useTheme();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border sticky top-0 z-10 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 min-w-0">
            <div className="h-8 w-8 rounded-md bg-primary/10 text-primary flex items-center justify-center shrink-0">
              <Sparkles className="h-4 w-4" />
            </div>
            <h1 className="text-sm font-semibold tracking-tight truncate">Push-up Counter</h1>
            <Badge
              variant="secondary"
              className="ml-1 hidden md:inline-flex text-[10px] font-normal"
            >
              <ShieldCheck className="h-3 w-3 mr-1" /> 100% on-device · WASM
            </Badge>
          </div>
          <div className="flex items-center gap-1.5">
            <Button
              size="icon"
              variant="ghost"
              onClick={toggle}
              className="h-8 w-8"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun className="h-3.5 w-3.5" />
              ) : (
                <Moon className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-4 grid lg:grid-cols-[1fr_300px] gap-4">
        <Card className="relative overflow-hidden bg-muted/30 border-border aspect-video flex items-center justify-center p-0">
          {loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-background/90 backdrop-blur">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="text-xs text-muted-foreground">{loadProgress}</p>
            </div>
          )}

          <video
            ref={videoRef}
            className="max-h-full max-w-full object-contain"
            playsInline
            muted
          />

          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none object-contain"
          />

          {!loading && (
            <div className="absolute top-2.5 left-2.5 flex gap-1.5 text-[10px]">
              <Badge
                variant="secondary"
                className="font-mono tabular-nums backdrop-blur bg-background/70"
              >
                {stats.fps} FPS
              </Badge>
              <Badge
                variant="secondary"
                className="font-mono tabular-nums backdrop-blur bg-background/70"
              >
                {stats.pushUpCount} push-ups
              </Badge>
            </div>
          )}
        </Card>

        <aside className="space-y-3">
          <Card className="p-3">
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
                <Activity className="h-3 w-3" /> PUSH-UPS
              </h2>
              <Button
                size="icon"
                variant="ghost"
                onClick={resetRepTracker}
                className="h-7 w-7 -mt-1 -mr-1"
                aria-label="Reset exercise counters"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
            <div className="flex items-end justify-between gap-3">
              <p className="text-4xl font-semibold tabular-nums leading-none">
                {stats.pushUpCount}
              </p>
              <Badge
                variant={stats.pushUpReady ? "default" : "secondary"}
                className="capitalize text-[10px] font-normal"
              >
                {stats.poseDetected ? stats.pushUpPhase : "No pose"}
              </Badge>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-150"
                style={{ width: `${Math.round(stats.pushUpProgress * 100)}%` }}
              />
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
              {stats.pushUpReady
                ? "Lower into the push-up, then press back up to count one rep."
                : "Get into a side-view push-up position so the tracker can learn your range."}
            </p>
          </Card>
        </aside>
      </main>

      {revealOpen && revealedUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 backdrop-blur-sm animate-in fade-in duration-300"
          role="dialog"
          aria-modal="true"
          aria-labelledby="reveal-title"
        >
          <Card className="relative w-full max-w-3xl border-primary/30 p-5 shadow-2xl animate-in zoom-in-95 slide-in-from-bottom-4 duration-300">
            <Button
              size="icon"
              variant="ghost"
              className="absolute right-3 top-3 h-8 w-8"
              onClick={() => setRevealOpen(false)}
              aria-label="Close reveal"
            >
              <X className="h-4 w-4" />
            </Button>

            <div className="pr-8">
              <p
                id="reveal-title"
                className="text-[15px] font-medium uppercase tracking-wider text-muted-foreground"
              >
                Unlocked
              </p>
              <p className="mt-3 break-all text-3xl font-normal leading-tight tracking-normal text-foreground">
                {revealedUrl}
              </p>
              <div className="mt-5">
                <Button onClick={copyRevealedUrl}>{copied ? "Copied" : "Copy"}</Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
