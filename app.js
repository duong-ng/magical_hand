// ============================================================
//  Magical Hands — MediaPipe Hand + Gesture Recognition
//  Performance-optimized for 120 FPS
// ============================================================

import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ---- Landmark names (21 points) ----
const LANDMARK_NAMES = [
  "Wrist",
  "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
  "Index MCP", "Index PIP", "Index DIP", "Index Tip",
  "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
  "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
  "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
];

// ---- Pre-computed tip indices (Set for O(1) lookup) ----
const TIP_INDICES = new Set([4, 8, 12, 16, 20]);

// ---- Hand colours ----
const HAND_COLORS = [
  { dot: "#a78bfa", line: "rgba(167,139,250,.55)", glow: "rgba(167,139,250,.35)" }, // violet
  { dot: "#22d3ee", line: "rgba(34,211,238,.55)", glow: "rgba(34,211,238,.35)" }, // cyan
  { dot: "#f472b6", line: "rgba(244,114,182,.55)", glow: "rgba(244,114,182,.35)" }, // pink
  { dot: "#facc15", line: "rgba(250,204,21,.55)", glow: "rgba(250,204,21,.35)" }, // amber
];

// ---- Finger connections for skeleton ----
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // index
  [0, 9], [9, 10], [10, 11], [11, 12],   // middle
  [0, 13], [13, 14], [14, 15], [15, 16], // ring
  [0, 17], [17, 18], [18, 19], [19, 20], // pinky
  [5, 9], [9, 13], [13, 17],             // palm cross
];

// ---- Constants ----
const TWO_PI = Math.PI * 2;
const DOM_THROTTLE_MS = 100; // Update DOM panels at 10 Hz, not every frame
const EMPTY_GESTURE_HTML = '<p class="panel__empty">Show your hand…</p>';
const EMPTY_LANDMARK_HTML = '<p class="panel__empty">No data yet</p>';
const NO_GESTURE_HTML = '<p class="panel__empty">No gesture detected</p>';

// ---- DOM refs ----
const $ = (s) => document.getElementById(s);
const startScreen = $("start-screen");
const cameraSection = $("camera-section");
const loadingOverlay = $("loading-overlay");
const loadingText = $("loading-text");
const btnStart = $("btn-start");
const btnStop = $("btn-stop");
const video = $("webcam");
const canvas = $("overlay");
const fpsBadge = $("fps-badge");
const handsBadge = $("hands-badge");
const gestureList = $("gesture-list");
const landmarkList = $("landmark-list");
const maxHandsSel = $("max-hands");
const confidenceRng = $("confidence");
const confidenceVal = $("confidence-val");
const toggleSkeleton = $("toggle-skeleton");
const toggleGlow = $("toggle-glow");
const toggleLabels = $("toggle-labels");

// ---- High-performance canvas context ----
const ctx = canvas.getContext("2d", {
  alpha: true,
  desynchronized: true,   // bypass compositor for lower latency
  willReadFrequently: false,
});

// ---- State ----
let gestureRecognizer = null;
let stream = null;
let animFrameId = null;
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;
let lastDomUpdate = 0;
let cachedW = 0;
let cachedH = 0;
let prevHandCount = -1;

// ---- Settings ----
const settings = {
  maxHands: 2,
  minConfidence: 0.5,
  showSkeleton: true,
  showGlow: true,
  showLabels: false,
};

// ---- Gesture emoji map (pre-built) ----
const GESTURE_EMOJI = {
  "Closed_Fist": "✊",
  "Open_Palm": "🖐️",
  "Pointing_Up": "☝️",
  "Thumb_Down": "👎",
  "Thumb_Up": "👍",
  "Victory": "✌️",
  "ILoveYou": "🤟",
};
function getGestureEmoji(name) {
  return GESTURE_EMOJI[name] || "🤚";
}

// ============================================================
//  Initialise MediaPipe
// ============================================================
async function initMediaPipe() {
  loadingOverlay.classList.remove("hidden");
  loadingText.textContent = "Downloading MediaPipe vision WASM…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  loadingText.textContent = "Creating Gesture Recognizer…";

  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: settings.maxHands,
    minHandDetectionConfidence: settings.minConfidence,
    minHandPresenceConfidence: settings.minConfidence,
    minTrackingConfidence: settings.minConfidence,
  });

  loadingOverlay.classList.add("hidden");
}

// ============================================================
//  Camera
// ============================================================
async function startCamera() {
  loadingOverlay.classList.remove("hidden");
  loadingText.textContent = "Accessing camera…";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 120 },  // Request high frame rate from camera
      },
      audio: false,
    });
  } catch (err) {
    loadingOverlay.classList.add("hidden");
    alert("Camera access denied. Please allow camera access and try again.");
    return;
  }

  video.srcObject = stream;
  await video.play();

  // Cache canvas dimensions
  cachedW = video.videoWidth;
  cachedH = video.videoHeight;
  canvas.width = cachedW;
  canvas.height = cachedH;

  // Switch views
  startScreen.classList.add("hidden");
  cameraSection.classList.remove("hidden");
  loadingOverlay.classList.add("hidden");

  // Reset counters
  lastTime = performance.now();
  frameCount = 0;
  fps = 0;
  lastDomUpdate = 0;
  prevHandCount = -1;

  // Start processing
  processFrame();
}

function stopCamera() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = null;

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;

  cameraSection.classList.add("hidden");
  startScreen.classList.remove("hidden");

  gestureList.innerHTML = EMPTY_GESTURE_HTML;
  landmarkList.innerHTML = EMPTY_LANDMARK_HTML;
}

// ============================================================
//  Frame processing loop (hot path — optimized)
// ============================================================
function processFrame() {
  if (!gestureRecognizer || !stream) return;

  const now = performance.now();

  // FPS counter (update badge only once per second)
  frameCount++;
  if (now - lastTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = now;
    fpsBadge.textContent = `${fps} FPS`;
  }

  // Run recognition
  const results = gestureRecognizer.recognizeForVideo(video, now);

  // Draw on canvas
  ctx.clearRect(0, 0, cachedW, cachedH);
  const landmarks = results.landmarks;
  const numHands = landmarks ? landmarks.length : 0;

  // Update hand count badge only on change
  if (numHands !== prevHandCount) {
    prevHandCount = numHands;
    handsBadge.textContent = `${numHands} hand${numHands !== 1 ? "s" : ""}`;
  }

  if (numHands > 0) {
    drawHands(results, numHands);

    // Throttle expensive DOM updates to ~10 Hz
    if (now - lastDomUpdate >= DOM_THROTTLE_MS) {
      lastDomUpdate = now;
      updateGesturePanel(results);
      updateLandmarkPanel(landmarks);
    }
  } else if (now - lastDomUpdate >= DOM_THROTTLE_MS) {
    lastDomUpdate = now;
    gestureList.innerHTML = EMPTY_GESTURE_HTML;
    landmarkList.innerHTML = EMPTY_LANDMARK_HTML;
  }

  animFrameId = requestAnimationFrame(processFrame);
}

// ============================================================
//  Drawing (hot path — heavily optimized)
// ============================================================
function drawHands(results, numHands) {
  const w = cachedW;
  const h = cachedH;
  const landmarks = results.landmarks;
  const showSkeleton = settings.showSkeleton;
  const showGlow = settings.showGlow;
  const showLabels = settings.showLabels;

  for (let i = 0; i < numHands; i++) {
    const hand = landmarks[i];
    const c = HAND_COLORS[i & 3]; // bitwise mod 4

    // --- Skeleton: batch all connections into one path ---
    if (showSkeleton) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = c.line;

      if (showGlow) {
        ctx.shadowColor = c.glow;
        ctx.shadowBlur = 16;
      }

      ctx.beginPath();
      for (let j = 0; j < HAND_CONNECTIONS.length; j++) {
        const conn = HAND_CONNECTIONS[j];
        const pA = hand[conn[0]];
        const pB = hand[conn[1]];
        ctx.moveTo(pA.x * w, pA.y * h);
        ctx.lineTo(pB.x * w, pB.y * h);
      }
      ctx.stroke();

      if (showGlow) ctx.shadowBlur = 0;
    }

    // --- Dots: group by type to minimize state changes ---
    // Pass 1: Glow rings on tips (only if glow enabled)
    if (showGlow) {
      ctx.fillStyle = c.glow;
      ctx.shadowColor = c.glow;
      ctx.shadowBlur = 22;
      ctx.beginPath();
      for (let idx = 0; idx < 21; idx++) {
        if (!TIP_INDICES.has(idx)) continue;
        const pt = hand[idx];
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.moveTo(x + 13, y);
        ctx.arc(x, y, 13, 0, TWO_PI);
      }
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Pass 2: All dots (batched by glow state)
    if (showGlow) {
      // Non-tip dots with smaller glow
      ctx.fillStyle = c.dot;
      ctx.shadowColor = c.glow;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      for (let idx = 0; idx < 21; idx++) {
        if (TIP_INDICES.has(idx)) continue;
        const pt = hand[idx];
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.moveTo(x + 4.5, y);
        ctx.arc(x, y, 4.5, 0, TWO_PI);
      }
      ctx.fill();

      // Tip dots with bigger glow
      ctx.shadowBlur = 22;
      ctx.beginPath();
      for (let idx = 0; idx < 21; idx++) {
        if (!TIP_INDICES.has(idx)) continue;
        const pt = hand[idx];
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.moveTo(x + 7, y);
        ctx.arc(x, y, 7, 0, TWO_PI);
      }
      ctx.fill();
      ctx.shadowBlur = 0;
    } else {
      // No glow — batch everything in one path per size
      ctx.fillStyle = c.dot;
      ctx.beginPath();
      for (let idx = 0; idx < 21; idx++) {
        if (TIP_INDICES.has(idx)) continue;
        const pt = hand[idx];
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.moveTo(x + 4.5, y);
        ctx.arc(x, y, 4.5, 0, TWO_PI);
      }
      ctx.fill();

      ctx.beginPath();
      for (let idx = 0; idx < 21; idx++) {
        if (!TIP_INDICES.has(idx)) continue;
        const pt = hand[idx];
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.moveTo(x + 7, y);
        ctx.arc(x, y, 7, 0, TWO_PI);
      }
      ctx.fill();
    }

    // Pass 3: White centres on tips (one batch)
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    for (let idx = 0; idx < 21; idx++) {
      if (!TIP_INDICES.has(idx)) continue;
      const pt = hand[idx];
      const x = pt.x * w;
      const y = pt.y * h;
      ctx.moveTo(x + 2.5, y);
      ctx.arc(x, y, 2.5, 0, TWO_PI);
    }
    ctx.fill();

    // --- Labels (only if enabled — expensive) ---
    if (showLabels) {
      ctx.font = "500 10px 'Inter', sans-serif";
      ctx.fillStyle = "rgba(255,255,255,.85)";
      for (let idx = 0; idx < 21; idx++) {
        const pt = hand[idx];
        ctx.fillText(LANDMARK_NAMES[idx], pt.x * w + 10, pt.y * h - 6);
      }
    }

    // --- Gesture label on wrist ---
    const gestures = results.gestures;
    if (gestures && gestures[i] && gestures[i].length > 0) {
      const gesture = gestures[i][0];
      if (gesture.categoryName && gesture.categoryName !== "None") {
        const wrist = hand[0];
        const gx = wrist.x * w;
        const gy = wrist.y * h + 30;
        const label = `${getGestureEmoji(gesture.categoryName)} ${gesture.categoryName}`;

        ctx.font = "700 16px 'Inter', sans-serif";
        const tw = ctx.measureText(label).width;
        const pad = 10;

        // Background pill
        ctx.fillStyle = "rgba(0,0,0,.6)";
        ctx.beginPath();
        ctx.roundRect(gx - tw / 2 - pad, gy - 14, tw + pad * 2, 26, 12);
        ctx.fill();

        // Border
        ctx.strokeStyle = c.line;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Text
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.fillText(label, gx, gy + 4);
        ctx.textAlign = "start";
      }
    }
  }
}

// ============================================================
//  Panels (throttled — called at ~10 Hz)
// ============================================================
function updateGesturePanel(results) {
  if (!results.gestures || results.gestures.length === 0) {
    gestureList.innerHTML = NO_GESTURE_HTML;
    return;
  }

  const parts = [];
  for (let i = 0; i < results.gestures.length; i++) {
    const gArr = results.gestures[i];
    if (!gArr || gArr.length === 0) continue;
    const g = gArr[0];
    if (g.categoryName === "None") continue;
    const handedness = results.handednesses?.[i]?.[0]?.categoryName || "?";
    parts.push(
      `<div class="gesture-item"><div><div class="gesture-item__name">${getGestureEmoji(g.categoryName)} ${g.categoryName.replace(/_/g, " ")}</div><div class="gesture-item__hand">${handedness} hand</div></div><span class="gesture-item__score">${(g.score * 100) | 0}%</span></div>`
    );
  }

  gestureList.innerHTML = parts.length > 0 ? parts.join("") : NO_GESTURE_HTML;
}

function updateLandmarkPanel(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    landmarkList.innerHTML = EMPTY_LANDMARK_HTML;
    return;
  }

  // Only show first hand
  const hand = landmarks[0];
  const parts = new Array(21);
  for (let idx = 0; idx < 21; idx++) {
    const pt = hand[idx];
    parts[idx] = `<div class="lm-row"><span class="lm-row__idx">${idx}</span><span class="lm-row__name">${LANDMARK_NAMES[idx]}</span><span class="lm-row__coords">${pt.x.toFixed(2)} ${pt.y.toFixed(2)} ${pt.z.toFixed(2)}</span></div>`;
  }
  landmarkList.innerHTML = parts.join("");
}

// ============================================================
//  Settings update
// ============================================================
function applySettings() {
  if (!gestureRecognizer) return;
  gestureRecognizer.setOptions({
    numHands: settings.maxHands,
    minHandDetectionConfidence: settings.minConfidence,
    minHandPresenceConfidence: settings.minConfidence,
    minTrackingConfidence: settings.minConfidence,
  });
}

// ============================================================
//  Event listeners
// ============================================================
btnStart.addEventListener("click", async () => {
  if (!gestureRecognizer) await initMediaPipe();
  startCamera();
});

btnStop.addEventListener("click", stopCamera);

maxHandsSel.addEventListener("change", () => {
  settings.maxHands = parseInt(maxHandsSel.value);
  applySettings();
});

confidenceRng.addEventListener("input", () => {
  settings.minConfidence = parseFloat(confidenceRng.value);
  confidenceVal.textContent = settings.minConfidence.toFixed(2);
  applySettings();
});

toggleSkeleton.addEventListener("change", () => {
  settings.showSkeleton = toggleSkeleton.checked;
});
toggleGlow.addEventListener("change", () => {
  settings.showGlow = toggleGlow.checked;
});
toggleLabels.addEventListener("change", () => {
  settings.showLabels = toggleLabels.checked;
});
