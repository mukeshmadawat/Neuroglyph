/**
 * script.js — NEUROGLYPH Neural Ink Interface
 * Handles navigation, canvas drawing, processing animation, CNN inference,
 * and state-based UI transitions.
 * ALL inference is delegated to server.py — no ML logic lives here.
 */

document.addEventListener("DOMContentLoaded", () => {

  // ============================================================
  // 0. Backend URL resolution
  // ============================================================
  const host = (window.location.hostname && window.location.hostname !== "")
    ? window.location.hostname : "localhost";
  const API = (window.location.protocol === "file:" || window.location.port !== "5000")
    ? `http://${host}:5000` : "";


  // ============================================================
  // 1. Section Navigation (smooth scroll + active indicator)
  // ============================================================
  const navLinks = document.querySelectorAll(".nav-link[data-section]");
  const sections = document.querySelectorAll(".section");

  function scrollToSection(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    closeMobileNav();
  }

  navLinks.forEach(link => {
    link.addEventListener("click", () => scrollToSection(link.dataset.section));
  });

  // Intersection observer for active nav state
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(l => {
          l.classList.toggle("active", l.dataset.section === entry.target.id);
        });
      }
    });
  }, { threshold: 0.3, rootMargin: "-52px 0px 0px 0px" });

  sections.forEach(sec => observer.observe(sec));


  // ============================================================
  // 2. Mobile Navigation
  // ============================================================
  const mobileBtn = document.getElementById("mobile-nav-btn");
  const mobileNav = document.getElementById("mobile-nav");

  function closeMobileNav() {
    mobileNav.classList.remove("open");
    if (mobileBtn) mobileBtn.setAttribute("aria-expanded", "false");
  }

  if (mobileBtn) {
    mobileBtn.addEventListener("click", () => {
      const isOpen = mobileNav.classList.toggle("open");
      mobileBtn.setAttribute("aria-expanded", isOpen);
    });
  }


  // ============================================================
  // 3. Drawing Canvas
  // ============================================================
  const canvas = document.getElementById("digit-canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const canvasFrame = document.getElementById("canvas-frame");
  const watermark = document.getElementById("canvas-watermark");
  const clearBtn = document.getElementById("clear-btn");
  const recognizeBtn = document.getElementById("recognize-btn");
  const recognizeBtnText = document.getElementById("recognize-btn-text");
  const liveToggle = document.getElementById("live-toggle");
  const liveToggleRow = document.getElementById("live-toggle-row");
  const liveToggleText = document.getElementById("live-toggle-text");
  const telemetryStatusPill = document.getElementById("telemetry-status-pill");
  const telemetryPanel = document.getElementById("telemetry-panel");
  const predDigitEl = document.getElementById("pred-digit-val");
  const predConfEl = document.getElementById("pred-conf-val");
  const flowResultDigit = document.getElementById("flow-result-digit");
  const clfGrid = document.getElementById("clf-grid");
  const clfDominantLabel = document.getElementById("clf-dominant-label");
  const procImgPreview = document.getElementById("proc-img-preview");
  const procImgPlaceholder = document.getElementById("proc-img-placeholder");
  const lensResult = document.getElementById("lens-result");

  // Pipeline Inspector Elements
  const pipelineToggleBtn = document.getElementById("pipeline-toggle-btn");
  const pipelinePanel = document.getElementById("pipeline-debug-panel");
  const pipelineHeaderStatus = document.getElementById("pipeline-header-status");

  const dbgImgRaw = document.getElementById("dbg-img-raw");
  const dbgRawPh = document.getElementById("dbg-raw-ph");
  const dbgRawDetail = document.getElementById("dbg-raw-detail");

  const dbgImgCrop = document.getElementById("dbg-img-crop");
  const dbgCropPh = document.getElementById("dbg-crop-ph");
  const dbgCropDetail = document.getElementById("dbg-crop-detail");

  const dbgImgCenter = document.getElementById("dbg-img-center");
  const dbgCenterPh = document.getElementById("dbg-center-ph");
  const dbgCenterDetail = document.getElementById("dbg-center-detail");

  const dbgImgFinal = document.getElementById("dbg-img-final");
  const dbgFinalPh = document.getElementById("dbg-final-ph");
  const dbgFinalDetail = document.getElementById("dbg-final-detail");

  const dbgShape = document.getElementById("dbg-shape");
  const dbgRange = document.getElementById("dbg-range");
  const dbgOrientation = document.getElementById("dbg-orientation");

  let isDrawing = false;
  let hasStrokes = false;
  let points = [];
  let liveDebounceTimer = null;
  let isPredicting = false;
  let pendingLivePredict = false;

  // ============================================================
  // Canvas Setup
  // ============================================================
  function initCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#FFFFFF";
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  }
  initCanvas();

  function getCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    let cx = e.clientX, cy = e.clientY;
    if (e.touches && e.touches.length > 0) {
      cx = e.touches[0].clientX;
      cy = e.touches[0].clientY;
    }
    return { x: (cx - rect.left) * sx, y: (cy - rect.top) * sy };
  }

  // ============================================================
  // Collapsible Pipeline Inspector
  // ============================================================
  if (pipelineToggleBtn && pipelinePanel) {
    pipelineToggleBtn.addEventListener("click", () => {
      const isCollapsed = pipelinePanel.classList.toggle("collapsed");
      pipelineToggleBtn.setAttribute("aria-expanded", (!isCollapsed).toString());
    });
  }

  // ============================================================
  // Live Analysis Toggle
  // ============================================================
  if (liveToggle && liveToggleRow) {
    liveToggle.addEventListener("change", () => {
      const isActive = liveToggle.checked;
      liveToggleRow.classList.toggle("active", isActive);
      if (isActive) {
        if (hasStrokes) {
          triggerLivePredict(true);
        } else if (telemetryStatusPill) {
          telemetryStatusPill.textContent = "LIVE ANALYSIS · STANDBY";
        }
      } else {
        clearTimeout(liveDebounceTimer);
        if (telemetryStatusPill) {
          telemetryStatusPill.textContent = hasStrokes ? "STROKE READY · CLICK ANALYZE" : "SYSTEM READY";
        }
      }
    });
  }

  // ============================================================
  // Class Distribution Grid (10 classes: 0-9)
  // ============================================================
  function initClassGrid() {
    if (!clfGrid) return;
    clfGrid.innerHTML = "";
    for (let d = 0; d < 10; d++) {
      const item = document.createElement("div");
      item.className = "clf-item";
      item.setAttribute("data-digit", d);
      item.innerHTML = `
        <span class="clf-digit">${d}</span>
        <div class="clf-bar-track">
          <div class="clf-bar-fill" style="width:0%"></div>
        </div>
        <span class="clf-pct">0.00%</span>
      `;
      clfGrid.appendChild(item);
    }
  }
  initClassGrid();

  function updateClassGrid(probs, predicted) {
    if (!clfGrid) return;
    const items = clfGrid.querySelectorAll(".clf-item");
    items.forEach((item, d) => {
      const p = probs && probs[d] !== undefined ? probs[d] : 0;
      const pctVal = p * 100;
      const isDominant = (d === predicted);
      item.classList.toggle("dominant", isDominant);
      
      const bar = item.querySelector(".clf-bar-fill");
      if (bar) bar.style.width = `${pctVal}%`;

      const pctEl = item.querySelector(".clf-pct");
      if (pctEl) {
        pctEl.classList.toggle("has-prob", p > 0.005 && !isDominant);
        pctEl.textContent = `${pctVal.toFixed(2)}%`;
      }
    });
  }

  // MNIST Preset Buttons
  const presetBtns = document.querySelectorAll(".preset-btn");
  function unmarkPresets() {
    presetBtns.forEach(b => b.classList.remove("active"));
  }

  // ============================================================
  // STATE 1 — IDLE
  // ============================================================
  function setIdleState() {
    hasStrokes = false;
    isDrawing = false;
    points = [];
    clearTimeout(liveDebounceTimer);
    unmarkPresets();

    // Primary button: disabled, "ANALYZE STROKE"
    recognizeBtn.disabled = true;
    recognizeBtn.classList.remove("ready", "analyzing");
    recognizeBtnText.textContent = "ANALYZE STROKE";

    // Visual frames & status
    canvasFrame.classList.remove("drawing", "analyzing", "classified");
    if (telemetryPanel) telemetryPanel.classList.remove("classified");

    const isLive = liveToggle && liveToggle.checked;
    if (telemetryStatusPill) {
      telemetryStatusPill.textContent = isLive ? "LIVE ANALYSIS · STANDBY" : "SYSTEM READY";
    }
    if (pipelineHeaderStatus) pipelineHeaderStatus.textContent = "4 STAGES · READY";

    // Inference Panel reset
    predDigitEl.textContent = "—";
    predDigitEl.classList.remove("digit-enter");
    predConfEl.textContent = "—";
    flowResultDigit.textContent = "—";
    if (lensResult) lensResult.textContent = "—";

    if (procImgPreview) procImgPreview.style.display = "none";
    if (procImgPlaceholder) procImgPlaceholder.style.display = "block";
    if (clfDominantLabel) clfDominantLabel.textContent = "AWAITING INFERENCE";

    // Reset Class Distribution
    updateClassGrid(null, -1);

    // Pipeline Inspector: Show idle placeholders, no fake data
    if (dbgImgRaw) { dbgImgRaw.style.display = "none"; dbgImgRaw.src = ""; }
    if (dbgRawPh) dbgRawPh.style.display = "block";
    if (dbgRawDetail) dbgRawDetail.textContent = "RAW CANVAS — —";

    if (dbgImgCrop) { dbgImgCrop.style.display = "none"; dbgImgCrop.src = ""; }
    if (dbgCropPh) dbgCropPh.style.display = "block";
    if (dbgCropDetail) dbgCropDetail.textContent = "BOUNDING BOX — —";

    if (dbgImgCenter) { dbgImgCenter.style.display = "none"; dbgImgCenter.src = ""; }
    if (dbgCenterPh) dbgCenterPh.style.display = "block";
    if (dbgCenterDetail) dbgCenterDetail.textContent = "ASPECT SCALED — —";

    if (dbgImgFinal) { dbgImgFinal.style.display = "none"; dbgImgFinal.src = ""; }
    if (dbgFinalPh) dbgFinalPh.style.display = "block";
    if (dbgFinalDetail) dbgFinalDetail.textContent = "CNN INPUT — —";

    if (dbgShape) dbgShape.textContent = "TENSOR: —";
    if (dbgRange) dbgRange.textContent = "RANGE: —";
    if (dbgOrientation) {
      dbgOrientation.textContent = "ORIENTATION: —";
      dbgOrientation.classList.remove("badge-verified");
    }
  }

  // ============================================================
  // STATE 2 — DRAWING READY
  // ============================================================
  function setDrawingReadyState() {
    hasStrokes = true;

    // Primary button becomes active: ANALYZE STROKE →
    recognizeBtn.disabled = false;
    recognizeBtn.classList.add("ready");
    recognizeBtn.classList.remove("analyzing");
    recognizeBtnText.textContent = "ANALYZE STROKE →";

    const isLive = liveToggle && liveToggle.checked;
    if (telemetryStatusPill) {
      telemetryStatusPill.textContent = isLive ? "LIVE INFERENCE ACTIVE" : "STROKE READY · CLICK ANALYZE";
    }

    // Pipeline Inspector updates Stage 01 with the actual canvas snapshot
    if (dbgImgRaw) {
      dbgImgRaw.src = canvas.toDataURL("image/png");
      dbgImgRaw.style.display = "block";
    }
    if (dbgRawPh) dbgRawPh.style.display = "none";
    if (dbgRawDetail) dbgRawDetail.textContent = "320 × 320 · LIVE STROKE";
  }

  // ============================================================
  // Live Analysis Trigger (debounced while drawing, instant on release)
  // ============================================================
  function triggerLivePredict(immediate = false) {
    if (!hasStrokes) return;
    clearTimeout(liveDebounceTimer);
    if (immediate) {
      executeInference(true);
    } else {
      liveDebounceTimer = setTimeout(() => executeInference(true), 120);
    }
  }

  // ============================================================
  // Drawing Handlers
  // ============================================================
  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    unmarkPresets();
    const c = getCoords(e);
    points = [c];
    ctx.beginPath();
    ctx.arc(c.x, c.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#FFFFFF";
    ctx.fill();

    if (watermark) watermark.style.opacity = "0";
    canvasFrame.classList.add("drawing");
    canvasFrame.classList.remove("classified");

    setDrawingReadyState();
    if (liveToggle && liveToggle.checked) {
      triggerLivePredict(false);
    }
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const c = getCoords(e);
    points.push(c);

    // Smooth Bezier interpolation
    if (points.length >= 3) {
      const p0 = points[points.length - 3];
      const p1 = points[points.length - 2];
      const p2 = points[points.length - 1];
      const mid1 = { x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2 };
      const mid2 = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };

      ctx.beginPath();
      ctx.moveTo(mid1.x, mid1.y);
      ctx.quadraticCurveTo(p1.x, p1.y, mid2.x, mid2.y);
      ctx.stroke();
    } else if (points.length === 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.stroke();
    }

    if (liveToggle && liveToggle.checked) {
      triggerLivePredict(false);
    }
  }

  function stopDraw() {
    if (!isDrawing) return;
    isDrawing = false;
    points = [];
    canvasFrame.classList.remove("drawing");

    // Update Stage 01 with completed stroke
    if (dbgImgRaw) {
      dbgImgRaw.src = canvas.toDataURL("image/png");
      dbgImgRaw.style.display = "block";
    }
    if (dbgRawPh) dbgRawPh.style.display = "none";

    if (liveToggle && liveToggle.checked) {
      triggerLivePredict(true);
    }
  }

  canvas.addEventListener("pointerdown", startDraw);
  canvas.addEventListener("pointermove", draw);
  window.addEventListener("pointerup", stopDraw);
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  window.addEventListener("touchend", stopDraw);

  // Clear Canvas
  function clearCanvas() {
    initCanvas();
    if (watermark) watermark.style.opacity = "1";
    setIdleState();
  }
  clearBtn.addEventListener("click", clearCanvas);

  // ============================================================
  // Unified Inference (Supports both manual click & live stream)
  // ============================================================
  async function executeInference(isLive = false) {
    if (!hasStrokes) return;

    if (isPredicting) {
      pendingLivePredict = true;
      return;
    }

    isPredicting = true;
    const dataUrl = canvas.toDataURL("image/png");

    // If manual analysis click, enter STATE 3 — ANALYZING
    if (!isLive) {
      recognizeBtn.disabled = true;
      recognizeBtn.classList.remove("ready");
      recognizeBtn.classList.add("analyzing");
      recognizeBtnText.textContent = "ANALYZING ···";

      predDigitEl.textContent = "ANALYZING";
      predDigitEl.classList.remove("digit-enter");
      predConfEl.textContent = "—";
      if (telemetryStatusPill) telemetryStatusPill.textContent = "ANALYZING STROKE ···";
    } else {
      if (telemetryStatusPill) telemetryStatusPill.textContent = "LIVE INFERENCE COMPUTING ···";
    }

    try {
      const response = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.status === "success") {
        displayResult(data);
      } else if (data.status === "empty") {
        if (!isLive) setIdleState();
      } else {
        console.error("Recognition error:", data.message);
        recognizeBtn.disabled = false;
        recognizeBtn.classList.add("ready");
        recognizeBtnText.textContent = "ANALYZE STROKE →";
      }
    } catch (err) {
      console.error("Prediction error:", err);
      recognizeBtn.disabled = false;
      recognizeBtn.classList.add("ready");
      recognizeBtnText.textContent = "ANALYZE STROKE →";
    } finally {
      isPredicting = false;
      recognizeBtn.classList.remove("analyzing");

      // Process queued live prediction if user drew while request was in-flight
      if (pendingLivePredict) {
        pendingLivePredict = false;
        if (liveToggle && liveToggle.checked && hasStrokes) {
          executeInference(true);
        }
      }
    }
  }

  // Manual button click initiates explicit prediction
  recognizeBtn.addEventListener("click", () => executeInference(false));

  // ============================================================
  // STATE 4 — RESULT
  // ============================================================
  function displayResult(data) {
    // Restore button to: ANALYZE AGAIN →
    recognizeBtn.disabled = false;
    recognizeBtn.classList.add("ready");
    recognizeBtn.classList.remove("analyzing");
    recognizeBtnText.textContent = "ANALYZE AGAIN →";

    // Visual frame classes
    canvasFrame.classList.add("classified");
    if (telemetryPanel) telemetryPanel.classList.add("classified");
    if (telemetryStatusPill) telemetryStatusPill.textContent = `INFERENCE LOCKED · ${data.confidence.toFixed(1)}%`;
    if (pipelineHeaderStatus) pipelineHeaderStatus.textContent = `4 STAGES · VERIFIED (${data.confidence.toFixed(1)}%)`;

    // Subtle 200–300ms opacity/scale transition for predicted digit
    predDigitEl.classList.add("digit-enter");
    predDigitEl.textContent = data.predicted_digit;
    requestAnimationFrame(() => {
      predDigitEl.classList.remove("digit-enter");
    });

    // Exact model confidence percentage
    predConfEl.textContent = `${data.confidence.toFixed(2)}%`;
    flowResultDigit.textContent = data.predicted_digit;
    if (lensResult) lensResult.textContent = data.predicted_digit;

    // Softmax class distribution bars (350ms transition, highlight predicted)
    updateClassGrid(data.probabilities, data.predicted_digit);
    if (clfDominantLabel) clfDominantLabel.textContent = `CLASS ${data.predicted_digit} DOMINANT`;

    // 28×28 CNN input strip preview
    if (data.processed_image && procImgPreview) {
      procImgPreview.src = data.processed_image;
      procImgPreview.style.display = "block";
      if (procImgPlaceholder) procImgPlaceholder.style.display = "none";
    }

    // Populate Pipeline Inspector with ACTUAL preprocessing data
    if (data.stages) {
      if (data.stages.raw && dbgImgRaw) {
        dbgImgRaw.src = data.stages.raw;
        dbgImgRaw.style.display = "block";
        if (dbgRawPh) dbgRawPh.style.display = "none";
        if (dbgRawDetail) dbgRawDetail.textContent = "320 × 320 · RAW CANVAS";
      }
      if (data.stages.cropped && dbgImgCrop) {
        dbgImgCrop.src = data.stages.cropped;
        dbgImgCrop.style.display = "block";
        if (dbgCropPh) dbgCropPh.style.display = "none";
      }
      if (data.stages.centered && dbgImgCenter) {
        dbgImgCenter.src = data.stages.centered;
        dbgImgCenter.style.display = "block";
        if (dbgCenterPh) dbgCenterPh.style.display = "none";
        if (dbgCenterDetail) dbgCenterDetail.textContent = "ASPECT SCALED 20×20";
      }
      if (data.stages.final && dbgImgFinal) {
        dbgImgFinal.src = data.stages.final;
        dbgImgFinal.style.display = "block";
        if (dbgFinalPh) dbgFinalPh.style.display = "none";
        if (dbgFinalDetail) dbgFinalDetail.textContent = "CNN INPUT 28×28";
      }
    }

    // Populate actual Technical Metadata (never placeholder telemetry)
    if (data.telemetry) {
      if (dbgShape && data.telemetry.tensor_shape) {
        dbgShape.textContent = `TENSOR: (${data.telemetry.tensor_shape.join(", ")})`;
      }
      if (dbgRange && data.telemetry.pixel_min !== undefined) {
        dbgRange.textContent = `RANGE: [${data.telemetry.pixel_min.toFixed(2)}, ${data.telemetry.pixel_max.toFixed(2)}]`;
      }
      if (dbgOrientation && data.telemetry.orientation) {
        dbgOrientation.textContent = data.telemetry.orientation;
        dbgOrientation.classList.add("badge-verified");
      }
      if (dbgCropDetail && data.telemetry.bbox) {
        const b = data.telemetry.bbox;
        dbgCropDetail.textContent = `BBOX: [${b.join(", ")}]`;
      }
    }
  }

  // ============================================================
  // MNIST Preset Samples (Loads exemplar, marks active, and displays verified telemetry)
  // ============================================================
  presetBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const digit = btn.dataset.digit;
      unmarkPresets();
      btn.classList.add("active");

      try {
        const res = await fetch(`${API}/api/sample/${digit}`);
        const sample = await res.json();
        if (sample.status === "success") {
          const img = new Image();
          img.onload = () => {
            initCanvas();
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 45, 45, 230, 230);
            if (watermark) watermark.style.opacity = "0";

            // Mark strokes available
            hasStrokes = true;

            // Immediately mark and display complete neural telemetry for this exemplar
            displayResult(sample);

            if (telemetryStatusPill) {
              telemetryStatusPill.textContent = `MNIST EXEMPLAR · DIGIT ${sample.digit} VERIFIED (${sample.confidence.toFixed(1)}%)`;
            }
          };
          img.src = sample.image;
        }
      } catch (err) {
        console.error("Error loading MNIST sample:", err);
      }
    });
  });

  // ============================================================
  // Keyboard Shortcuts
  // ============================================================
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") clearCanvas();
    if (e.key === "Enter" && !recognizeBtn.disabled) executeInference(false);
  });

  // ============================================================
  // Initialize to STATE 1 — IDLE on boot
  // ============================================================
  setIdleState();

  // Load model metrics
  (async function loadMetrics() {
    try {
      const res = await fetch(`${API}/api/metrics`);
      const data = await res.json();
      console.log("NEUROGLYPH · Model connected · Test Accuracy:", data.test_accuracy);
    } catch {
      // Silently fail
    }
  })();

});
