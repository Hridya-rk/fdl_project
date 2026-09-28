const API_BASE = ""; // same-origin: backend serves this frontend at /app

const GENRE_EMOJI = {
  blues: "🎷", classical: "🎻", country: "🤠", disco: "🕺", hiphop: "🎤",
  jazz: "🎺", metal: "🤘", pop: "🎧", reggae: "🌴", rock: "🎸",
};

let selectedFile = null;
let audioCtx = null;

// ---------- Status pill ----------
async function checkHealth() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_ready) {
      dot.classList.add("ok");
      text.textContent = "Model ready";
    } else {
      dot.classList.add("bad");
      text.textContent = "Model not trained yet";
    }
  } catch (e) {
    dot.classList.add("bad");
    text.textContent = "Backend unreachable";
  }
}

// ---------- Upload / dropzone ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

document.getElementById("clearBtn").addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.hidden = true;
  analyzeBtn.disabled = true;
  hideError();
});

function handleFile(file) {
  hideError();
  selectedFile = file;
  fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  fileInfo.hidden = false;
  analyzeBtn.disabled = false;
  drawWaveform(file);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}

// ---------- Waveform (client-side, via Web Audio API) ----------
async function drawWaveform(file) {
  const canvas = document.getElementById("waveformCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;

  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const raw = audioBuffer.getChannelData(0);
    const samples = 200;
    const blockSize = Math.floor(raw.length / samples);
    const peaks = [];
    for (let i = 0; i < samples; i++) {
      const block = raw.subarray(i * blockSize, (i + 1) * blockSize);
      let max = 0;
      for (let j = 0; j < block.length; j++) max = Math.max(max, Math.abs(block[j]));
      peaks.push(max);
    }
    renderPeaks(ctx, canvas, peaks);
  } catch (e) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function renderPeaks(ctx, canvas, peaks) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const barWidth = w / peaks.length;
  const gradient = ctx.createLinearGradient(0, 0, w, 0);
  gradient.addColorStop(0, "#ff5722");
  gradient.addColorStop(1, "#ffd600");
  ctx.fillStyle = gradient;
  peaks.forEach((p, i) => {
    const barHeight = Math.max(2, p * h * 0.9);
    ctx.fillRect(i * barWidth, (h - barHeight) / 2, Math.max(1, barWidth - 1), barHeight);
  });
}

// ---------- Analyze ----------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  setLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    renderResults(data);
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  document.getElementById("analyzeBtnText").textContent = loading ? "Analyzing…" : "Analyze audio";
  document.getElementById("analyzeSpinner").hidden = !loading;
}

function renderResults(data) {
  document.getElementById("resultsPlaceholder").hidden = true;
  const content = document.getElementById("resultsContent");
  content.hidden = false;

  const emoji = GENRE_EMOJI[data.top_genre] || "🎵";
  document.getElementById("topGenre").innerHTML =
    `${emoji} Predicted genre: <span style="text-transform:capitalize">${data.top_genre}</span>`;

  const bars = document.getElementById("bars");
  bars.innerHTML = "";
  data.predictions.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (idx === 0 ? " top" : "");
    row.innerHTML = `
      <span>${GENRE_EMOJI[p.genre] || ""} ${p.genre}</span>
      <div class="bar-track"><div class="bar-fill" data-width="${p.confidence}"></div></div>
      <span>${p.confidence.toFixed(1)}%</span>
    `;
    bars.appendChild(row);
  });
  // animate after insertion
  requestAnimationFrame(() => {
    bars.querySelectorAll(".bar-fill").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
  });

  document.getElementById("metaLine").textContent =
    `Analyzed ${data.segments_analyzed} segment(s) across ${data.duration}s of audio.`;

  const specCard = document.getElementById("spectrogramCard");
  specCard.hidden = false;
  document.getElementById("spectrogramImg").src = `data:image/png;base64,${data.spectrogram_image}`;
}

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------- Model info (architecture + training charts) ----------
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    const data = await res.json();

    const archList = document.getElementById("archList");
    archList.innerHTML = data.architecture.map(l => `
      <div class="arch-row">
        <span class="layer-name">${l.layer}</span>
        <span class="layer-detail">${l.detail}</span>
      </div>
    `).join("");

    if (data.history) {
      renderCharts(data.history);
    } else {
      document.getElementById("accuracyStats").innerHTML =
        `<p class="muted small">No training history found yet — run <code>src/train.py</code> to generate it.</p>`;
    }
  } catch (e) {
    console.error("Failed to load model info", e);
  }
}

function renderCharts(history) {
  const epochs = history.accuracy.map((_, i) => i + 1);

  new Chart(document.getElementById("accuracyChart"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "Training accuracy", data: history.accuracy, borderColor: "#ff5722", borderWidth: 3, tension: 0.35, pointRadius: 0 },
        { label: "Validation accuracy", data: history.val_accuracy, borderColor: "#111111", borderWidth: 3, tension: 0.35, pointRadius: 0 },
      ],
    },
    options: chartOptions("Accuracy"),
  });

  new Chart(document.getElementById("lossChart"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "Training loss", data: history.loss, borderColor: "#ff5722", borderWidth: 3, tension: 0.35, pointRadius: 0 },
        { label: "Validation loss", data: history.val_loss, borderColor: "#111111", borderWidth: 3, tension: 0.35, pointRadius: 0 },
      ],
    },
    options: chartOptions("Loss"),
  });

  document.getElementById("accuracyStats").innerHTML = `
    <div class="stat"><div class="value">${(history.test_accuracy * 100).toFixed(1)}%</div><div class="label">Test accuracy</div></div>
    <div class="stat"><div class="value">${history.test_loss.toFixed(3)}</div><div class="label">Test loss</div></div>
    <div class="stat"><div class="value">${history.accuracy.length}</div><div class="label">Epochs trained</div></div>
  `;
}

function chartOptions(label) {
  return {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#14110d", font: { weight: 700 } } },
      title: { display: false },
    },
    scales: {
      x: { ticks: { color: "#6b6259" }, grid: { color: "rgba(20,17,13,0.08)" }, title: { display: true, text: "Epoch", color: "#14110d", font: { weight: 700 } } },
      y: { ticks: { color: "#6b6259" }, grid: { color: "rgba(20,17,13,0.08)" }, title: { display: true, text: label, color: "#14110d", font: { weight: 700 } } },
    },
  };
}

checkHealth();
loadModelInfo();
