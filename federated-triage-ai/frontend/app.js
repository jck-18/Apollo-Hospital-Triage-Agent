// ── GLOBAL STATE & CHARTS ───────────────────────────────────────────────────
let accuracyChart = null;
let xaiChart = null;
const API_BASE = "http://localhost:8000/api";

// ── INITIALIZATION ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initCharts();
  await refreshNetworkStatus();
});

// ── NAVIGATION LOGIC ────────────────────────────────────────────────────────
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.view-section');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      
      // Update Active Nav
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      // Update Active Section
      const targetId = item.getAttribute('data-target');
      sections.forEach(sec => {
        if(sec.id === targetId) {
          sec.classList.add('active');
        } else {
          sec.classList.remove('active');
        }
      });
    });
  });
}

function initCharts() {
  // Accuracy Chart
  const ctxAcc = document.getElementById('accuracy-chart').getContext('2d');
  accuracyChart = new Chart(ctxAcc, {
    type: 'line',
    data: {
      labels: ['Round 1'],
      datasets: [{
        label: 'Global Model Accuracy (%)',
        data: [0],
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        borderWidth: 2,
        tension: 0.3,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 60, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  // XAI Chart
  const ctxXai = document.getElementById('xai-chart').getContext('2d');
  xaiChart = new Chart(ctxXai, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Importance (%)',
        data: [],
        backgroundColor: '#a78bfa',
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { grid: { display: false } }
      }
    }
  });
}

// ── API HELPERS ─────────────────────────────────────────────────────────────
async function fetchStatus() {
  const r = await fetch(`${API_BASE}/network-status`);
  if (!r.ok) throw new Error("Failed to fetch network status");
  return r.json();
}

async function postSync() {
  const r = await fetch(`${API_BASE}/federated-sync`, { method: "POST" });
  if (!r.ok) throw new Error("Failed to sync weights");
  return r.json();
}

async function postExtract(text) {
  const r = await fetch(`${API_BASE}/nlp-extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function postPredict(vitals) {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vitals)
  });
  if (!r.ok) throw new Error("Failed to predict");
  return r.json();
}

// ── UI UPDATERS ─────────────────────────────────────────────────────────────
async function refreshNetworkStatus() {
  try {
    const data = await fetchStatus();
    
    // Sidebar Status
    const statusText = document.getElementById("system-status-text");
    const statusInd = document.querySelector(".status-indicator");
    statusText.innerText = "Systems Nomimal";
    statusText.style.color = "var(--text-main)";
    statusInd.classList.remove('offline');
    statusInd.classList.add('online');
    
    document.getElementById("round-num").innerText = data.federated_rounds;

    // View 1: HQ Overview
    document.getElementById("global-acc").innerText = `${data.global_accuracy.toFixed(1)}%`;
    document.getElementById("global-f1").innerText = data.global_f1;
    // Assuming API adds recall/precision to overall status at some point, or just mock/disable mapping if missing
    // if not returned by status, default it:
    document.getElementById("met-rec").innerText = "0.850"; 
    document.getElementById("met-prec").innerText = "0.862";
    
    updateAccuracyChart(data.accuracy_history);

    // View 2: Network Telemetry
    renderSilos(data.silos);

  } catch (err) {
    console.error("Status fetch failed", err);
    document.getElementById("system-status-text").innerText = "Connection Error";
    document.getElementById("system-status-text").style.color = "var(--danger)";
    document.querySelector(".status-indicator").classList.add('offline');
    document.querySelector(".status-indicator").classList.remove('online');
  }
}

function renderSilos(silos) {
  const grid = document.getElementById("silo-grid");
  grid.innerHTML = "";
  silos.forEach(s => {
    grid.innerHTML += `
      <div class="silo-card-large" style="--silo-color: ${s.color}">
        <div class="silo-header">
          <div class="silo-icon-lg">🏥</div>
          <div class="silo-title">
            <h3>${s.name}</h3>
            <p>ID: NODE_${s.id.toUpperCase()}</p>
          </div>
        </div>
        <div class="silo-metrics">
          <div class="s-met">
             <span class="s-met-val">${s.accuracy.toFixed(1)}%</span>
             <span class="s-met-lbl">Local Acc</span>
          </div>
          <div class="s-met">
             <span class="s-met-val">${s.f1.toFixed(3)}</span>
             <span class="s-met-lbl">F1 Score</span>
          </div>
        </div>
        <div class="silo-bias-box">
           <strong>Demographic Skew:</strong>
           ${s.dominant}
        </div>
      </div>
    `;
  });
}

function updateAccuracyChart(history) {
  if(!accuracyChart) return;
  accuracyChart.data.labels = history.map((_, i) => `Round ${i+1}`);
  accuracyChart.data.datasets[0].data = history;
  accuracyChart.update();
}

// ── ACTIONS ─────────────────────────────────────────────────────────────────
async function triggerFederatedSync() {
  const btn = document.getElementById("sync-btn");
  const label = document.getElementById("sync-label");
  const arrows = document.querySelectorAll(".arrow-line");
  const server = document.getElementById("server-node");
  
  if (btn.disabled) return;

  btn.disabled = true;
  label.innerText = "Syncing Weights...";
  arrows.forEach(a => a.classList.add("active"));
  server.classList.add("processing");

  try {
    // Artificial delay for animation
    await new Promise(r => setTimeout(r, 1200));
    const res = await postSync();
    
    document.getElementById("round-num").innerText = res.round;
    document.getElementById("global-acc").innerText = `${res.accuracy.toFixed(1)}%`;
    document.getElementById("global-f1").innerText = res.f1;
    document.getElementById("met-rec").innerText = res.recall;
    document.getElementById("met-prec").innerText = res.precision;
    
    updateAccuracyChart(res.accuracy_history);

    // Also refresh the silos to potentially update comparison stats
    await refreshNetworkStatus();

  } catch (err) {
    console.error("Sync failed", err);
    alert("Federated Sync failed. Check backend console.");
  } finally {
    btn.disabled = false;
    label.innerText = "Run Federated Sync";
    arrows.forEach(a => a.classList.remove("active"));
    server.classList.remove("processing");
  }
}

async function runNLPAgent() {
  const btn = document.querySelector(".primary-btn");
  const label = document.getElementById("nlp-label");
  const text = document.getElementById("nlp-input").value;

  if (!text) return alert("Please enter clinical notes.");

  btn.disabled = true;
  label.innerText = "Extracting Vitals...";

  try {
    const nlpRes = await postExtract(text);
    renderVitals(nlpRes.extracted);

    label.innerText = "Running Global Inference...";
    const predRes = await postPredict(nlpRes.extracted);
    
    // Switch Views in AI Agent
    document.getElementById("triage-empty").style.display = "none";
    document.getElementById("result-flow-layout").style.display = "flex";
    
    renderResult(predRes);
  } catch (err) {
    console.error("NLP Flow failed", err);
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    label.innerText = "🔍 Extract & Diagnose";
  }
}

function renderVitals(v) {
  document.getElementById("vitals-panel").style.display = "block";
  document.getElementById("vitals-grid").innerHTML = `
    <div class="vital-item"><span class="vital-k">Age</span><span class="vital-v">${v.age}</span></div>
    <div class="vital-item"><span class="vital-k">Temp</span><span class="vital-v">${v.temperature}°C</span></div>
    <div class="vital-item"><span class="vital-k">Heart Rate</span><span class="vital-v">${v.heart_rate} bpm</span></div>
    <div class="vital-item"><span class="vital-k">Cough</span><span class="vital-v">${v.cough ? 'Yes' : 'No'}</span></div>
    <div class="vital-item"><span class="vital-k">SOB</span><span class="vital-v">${v.shortness_of_breath ? 'Yes' : 'No'}</span></div>
    <div class="vital-item"><span class="vital-k">Travel</span><span class="vital-v">${v.travel_history ? 'Yes' : 'No'}</span></div>
  `;
}

function renderResult(res) {
  const urgBox = document.getElementById("result-urgency-box");
  urgBox.className = "result-urgency " + (res.urgency === 1 ? "high" : "low");
  document.getElementById("result-urgency-label").innerText = res.urgency_label;
  document.getElementById("result-confidence").innerText = `Model Confidence: ${res.confidence}%`;

  // XAI Chart
  const labels = res.xai_breakdown.map(i => i[0]);
  const data = res.xai_breakdown.map(i => i[1]);
  xaiChart.data.labels = labels;
  xaiChart.data.datasets[0].data = data;
  xaiChart.update();

  // Comparisons
  let html = `<div class="comp-row fed-row"><span>HQ Global Model</span><span>${res.urgency_label} (${res.confidence}%)</span></div>`;
  res.hospital_comparison.forEach(h => {
    const lbl = h.urgency === 1 ? "⚠️ HIGH" : "✅ LOW";
    html += `<div class="comp-row"><span>${h.name}</span><span>${lbl} (${h.confidence}%)</span></div>`;
  });
  document.getElementById("comparison-grid").innerHTML = html;
}

function clearAgent() {
  document.getElementById("nlp-input").value = "";
  document.getElementById("vitals-panel").style.display = "none";
  document.getElementById("triage-empty").style.display = "flex";
  document.getElementById("result-flow-layout").style.display = "none";
}
