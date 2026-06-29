const $ = (sel) => document.querySelector(sel);

const themeToggle = $("#theme-toggle");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("sightline-theme", theme);
  if (!themeToggle) return;
  const isLight = theme === "light";
  themeToggle.textContent = isLight ? "☾" : "☀";
  themeToggle.setAttribute(
    "aria-label",
    isLight ? "Switch to dark mode" : "Switch to light mode",
  );
}

applyTheme(document.documentElement.dataset.theme || "dark");

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(next);
  });
}

const tabs = document.querySelectorAll(".tab");
const panes = document.querySelectorAll(".source-pane");
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const fileName = $("#file-name");
const youtubeInput = $("#youtube-url");
const convertBtn = $("#convert-btn");
const progressWrap = $("#progress-wrap");
const progressFill = $("#progress-fill");
const progressPct = $("#progress-pct");
const progressEta = $("#progress-eta");
const statusMsg = $("#status-msg");
const perfPanel = $("#perf-panel");
const historyBlock = $("#history-block");
const HISTORY_PANEL_KEY = "sightline-history-open";
const headerLive = $("#header-live");
const outputEmpty = $("#output-empty");
const docPreview = $("#doc-preview");
const docRaw = $("#doc-raw");
const btnPreview = $("#btn-preview");
const btnRaw = $("#btn-raw");
const chatPane = $("#chat-pane");
const chatEmpty = $("#chat-empty");
const chatMessages = $("#chat-messages");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatSend = $("#chat-send");
const chatContextWarn = $("#chat-context-warn");
const chatSuggestions = $("#chat-suggestions");
const btnDownloadMd = $("#btn-download-md");
const btnDownloadHtml = $("#btn-download-html");
const btnDownloadPdf = $("#btn-download-pdf");
const btnExportMetrics = $("#btn-export-metrics");
const btnExportMenu = $("#btn-export-menu");
const exportDropdown = $("#export-dropdown");
const exportMenu = $("#export-menu");
const historyList = $("#history-list");
const historyCount = $("#history-count");
const btnRefreshHistory = $("#btn-refresh-history");
const promptDefault = $("#prompt-default");
const promptCustom = $("#prompt-custom");

let selectedFile = null;
let videoDurationSec = null;
let currentJobId = null;
let pollTimer = null;
let pollGeneration = 0;
let plannedGanttTimer = null;
let plannedGanttStartedAt = 0;
let liveClockTimer = null;
let liveClockCreatedAt = null;
let lastRenderedMetricsKey = "";
let lastMarkdown = "";
let lastMetrics = null;
let chatSending = false;
let chatHasContext = true;
const enrichmentStore = new Map();
let enrichmentIdSeq = 0;

const HACKATHON_RPM = 100;
const ACTIVE_JOB_KEY = "sightline-active-job";
const POLL_INTERVAL_MS = 800;
const POLL_ERROR_RETRY_MS = 1500;
const POLL_MAX_ERRORS = 8;
const LIVE_API_CALLS_CAP = 40;
const RUNNING_STATUSES = new Set([
  "pending",
  "extracting",
  "transcribing",
  "analyzing",
  "composing",
]);
const HISTORY_OPEN_STATUSES = new Set(["completed", "failed", ...RUNNING_STATUSES]);

function activateTab(paneId) {
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.pane === paneId));
  panes.forEach((p) => p.classList.toggle("active", p.id === paneId));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.pane));
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function formatDuration(sec) {
  if (sec == null) return "—";
  const s = Math.round(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

let probeData = null;

function applyProbeData(data) {
  probeData = data;
  videoDurationSec = data.duration_sec;
  updateEstimate();
}

function formatPipelineMinutes(minutes) {
  if (minutes == null || Number.isNaN(minutes)) return null;
  const m = Number(minutes);
  if (m < 1) return "<1 min";
  if (m < 10) return `~${m.toFixed(1)} min`;
  return `~${Math.round(m)} min`;
}

function updateEstimate() {
  const box = $("#estimate-box");
  if (!probeData) {
    box.classList.remove("warn");
    box.textContent =
      "Full video processed · API calls paced at 100 req/min and 100K tok/min";
    return;
  }
  const scout = probeData.estimated_scout_calls;
  const analyze = probeData.estimated_analyze_calls;
  const total = probeData.estimated_total_api_calls;
  const durationNote = `Full video · ${formatDuration(probeData.duration_sec)}${
    probeData.max_frames != null ? ` · ${probeData.max_frames} key frames` : ""
  }`;
  const totalNote = formatPipelineMinutes(
    probeData.estimated_total_minutes ?? probeData.estimated_pipeline_minutes,
  );
  let callsNote = "";
  if (total != null && scout != null && analyze != null) {
    callsNote = ` · ~${total} Cerebras calls (scout ${scout} + analyze ${analyze} + compose 1)`;
  } else if (total != null) {
    callsNote = ` · ~${total} Cerebras calls`;
  }
  box.classList.remove("warn");
  box.textContent = `${durationNote}${callsNote}${totalNote ? ` · est. ${totalNote} total` : ""}`;
}

async function probeFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/probe", { method: "POST", body: form });
  if (!res.ok) return null;
  return res.json();
}

async function probeYoutubeUrl(url) {
  const form = new FormData();
  form.append("youtube_url", url);
  const res = await fetch("/api/probe", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function showVideoInfo(meta, sourceLabel) {
  const el = $("#video-info");
  if (!meta) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.textContent =
    `${sourceLabel} · ${formatDuration(meta.duration_sec)} · ${meta.width}×${meta.height}` +
    (meta.fps ? ` · ${meta.fps} fps` : "") +
    (meta.max_frames != null ? ` · full video · ${meta.max_frames} key frames` : " · full video") +
    (meta.estimated_total_api_calls != null
      ? ` · ~${meta.estimated_total_api_calls} Cerebras calls`
      : "") +
    (meta.estimated_total_minutes || meta.estimated_pipeline_minutes
      ? ` · est. ${formatPipelineMinutes(meta.estimated_total_minutes ?? meta.estimated_pipeline_minutes)} total`
      : "");
}

function setFile(file) {
  selectedFile = file;
  fileName.textContent = `${file.name} (${formatSize(file.size)})`;
  videoDurationSec = null;
  probeData = null;
  $("#video-info").hidden = true;
  updateEstimate();
}

updateEstimate();

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRelativeTime(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const diffSec = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

function historyLabel(job) {
  if (job.title) return job.title;
  if (job.source_name && job.source_type !== "youtube") return job.source_name;
  if (job.youtube_video_id) return `YouTube video ${job.youtube_video_id}`;
  return `Report ${job.job_id.slice(0, 8)}`;
}

function historyMediaHtml(job) {
  const thumb = job.thumbnail_url;
  const preview = job.preview_url;
  if (!thumb && !preview) {
    return '<div class="history-media history-media--empty" aria-hidden="true">◇</div>';
  }
  const thumbImg = thumb
    ? `<img class="history-img history-img--thumb" src="${escapeHtml(thumb)}" alt="" loading="lazy">`
    : "";
  const previewImg = preview
    ? `<img class="history-img history-img--preview" src="${escapeHtml(preview)}" alt="" loading="lazy">`
    : "";
  const cls = preview && thumb ? " history-media--dual" : preview ? " history-media--preview-only" : "";
  return `<div class="history-media${cls}">${thumbImg}${previewImg}</div>`;
}

function historyStatusClass(status) {
  if (status === "completed") return "history-status--completed";
  if (status === "failed") return "history-status--failed";
  return "history-status--running";
}

function canOpenHistoryJob(job) {
  return HISTORY_OPEN_STATUSES.has(job.status);
}

function formatJobError(error) {
  if (!error) return "Pipeline failed";
  const lines = String(error)
    .trim()
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (line.startsWith("Traceback") || line.startsWith('File "')) continue;
    return line.replace(/^[\w.]+Error:\s*/, "");
  }
  return lines[lines.length - 1] || "Pipeline failed";
}

function showFailedJobReport(job) {
  outputEmpty.style.display = "none";
  docPreview.innerHTML = "";
  docPreview.classList.remove("visible");
  docRaw.textContent = job.error || formatJobError(job.error);
  docRaw.classList.add("visible");
  btnPreview?.classList.remove("active");
  btnRaw?.classList.add("active");
  setExportButtonsEnabled(false);
  setChatEnabled(false);
}

function renderHistory(jobs) {
  if (!historyList) return;
  if (historyCount) {
    if (jobs.length) {
      historyCount.textContent = String(jobs.length);
      historyCount.hidden = false;
    } else {
      historyCount.hidden = true;
    }
  }
  if (!jobs.length) {
    historyList.innerHTML = '<li class="history-empty">No reports yet.</li>';
    return;
  }

  historyList.innerHTML = jobs
    .map((job) => {
      const meta = [
        formatRelativeTime(job.created_at),
        job.source_type || "source",
        job.elapsed_sec != null ? `${Number(job.elapsed_sec).toFixed(1)}s` : null,
        job.cerebras_calls != null ? `${job.cerebras_calls} calls` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      const active = job.job_id === currentJobId ? " active" : "";
      const canOpen = canOpenHistoryJob(job);
      const failed = job.status === "failed" ? " history-item--failed" : "";
      const openable = canOpen ? ' data-openable="true" role="button" tabindex="0"' : "";
      return `
        <li class="history-item${active}${failed}${canOpen ? " history-item--openable" : ""}" data-job-id="${job.job_id}"${openable}>
          ${historyMediaHtml(job)}
          <div class="history-item-main">
            <span class="history-item-title">${escapeHtml(historyLabel(job))}</span>
            <span class="history-item-meta">
              ${escapeHtml(meta)}
              <span class="history-status ${historyStatusClass(job.status)}">${job.status}</span>
            </span>
          </div>
          <div class="history-item-actions">
            <button type="button" class="history-btn history-btn--delete" data-action="delete" data-job-id="${job.job_id}" aria-label="Delete report">Delete</button>
          </div>
        </li>`;
    })
    .join("");
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const LATEX_SYMBOLS = {
  "\\rightarrow": "→",
  "\\leftarrow": "←",
  "\\Rightarrow": "⇒",
  "\\Leftrightarrow": "⇔",
  "\\leftrightarrow": "↔",
  "\\times": "×",
  "\\cdot": "·",
  "\\leq": "≤",
  "\\geq": "≥",
  "\\neq": "≠",
  "\\infty": "∞",
  "\\pm": "±",
  "\\approx": "≈",
};

function replaceLatexSymbols(body) {
  let out = body;
  for (const [cmd, ch] of Object.entries(LATEX_SYMBOLS)) {
    out = out.split(cmd).join(ch);
  }
  return out;
}

function normalizeInlineLatex(markdown) {
  return markdown
    .replace(/\$\$([\s\S]+?)\$\$/g, (match, body) => {
      const normalized = replaceLatexSymbols(body.trim());
      return /\\/.test(normalized) ? match : normalized;
    })
    .replace(/\$([^\$\n]+?)\$/g, (match, body) => {
      const normalized = replaceLatexSymbols(body.trim());
      return /\\/.test(normalized) ? match : normalized;
    });
}

function renderMarkdownPreview(markdown, jobId) {
  const html = marked.parse(rewriteAssetUrls(normalizeInlineLatex(markdown), jobId));
  docPreview.innerHTML = html;
}

function configureMarked() {
  if (typeof marked === "undefined" || marked.__sightlineFigures) return;
  marked.use({
    renderer: {
      image({ href, title, text }) {
        const alt = escapeHtml(text || "");
        const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
        const caption = text
          ? `<figcaption class="doc-figcaption">${alt}</figcaption>`
          : "";
        return `<figure class="doc-figure"><img src="${href}" alt="${alt}"${titleAttr} loading="lazy">${caption}</figure>`;
      },
    },
  });
  marked.__sightlineFigures = true;
}

configureMarked();

async function loadHistory() {
  if (!historyList) return;
  try {
    const res = await fetch("/api/jobs?limit=50");
    if (!res.ok) return;
    const data = await res.json();
    renderHistory(data.jobs || []);
  } catch {
    historyList.innerHTML = '<li class="history-empty">Could not load history.</li>';
  }
}

function probeFromJobMetrics(job) {
  const metrics = job.metrics || {};
  const steps = metrics.steps || [];
  const probeStep = steps.find(
    (s) => s.label && s.label.toLowerCase().includes("video probe"),
  );
  const detail = probeStep?.detail || {};
  const duration = detail.duration_sec;
  if (duration == null) return null;

  let width = 0;
  let height = 0;
  if (detail.resolution) {
    const [w, h] = detail.resolution.split("x").map(Number);
    width = w || 0;
    height = h || 0;
  }

  const scout = metrics.scout_calls;
  const analyze = metrics.analyze_calls ?? metrics.selected_frames;
  const total =
    scout != null && analyze != null ? scout + analyze + 1 : metrics.cerebras?.calls;

  return {
    duration_sec: duration,
    width,
    height,
    fps: detail.fps || 0,
    max_frames: metrics.selected_frames ?? metrics.frame_count,
    estimated_scout_calls: scout,
    estimated_analyze_calls: analyze,
    estimated_total_api_calls: total,
    estimated_total_minutes: metrics.estimated_total_minutes,
    estimated_pipeline_minutes: metrics.estimated_pipeline_minutes,
  };
}

function restoreJobInputs(job) {
  const sourceType = job.source_type === "file" ? "file" : "youtube";
  const languageSelect = $("#language");
  if (languageSelect) {
    languageSelect.value = job.language || "auto";
  }
  if (promptCustom) {
    promptCustom.value = job.custom_prompt || "";
  }

  if (sourceType === "youtube") {
    activateTab("pane-youtube");
    const url =
      job.source_name ||
      (job.youtube_video_id
        ? `https://www.youtube.com/watch?v=${job.youtube_video_id}`
        : "");
    youtubeInput.value = url;
    selectedFile = null;
    fileName.textContent = "";
    fileInput.value = "";
  } else {
    activateTab("pane-upload");
    selectedFile = null;
    fileInput.value = "";
    fileName.textContent = job.source_name || "";
    youtubeInput.value = "";
  }

  const probeLike = probeFromJobMetrics(job);
  if (probeLike) {
    applyProbeData(probeLike);
    showVideoInfo(probeLike, sourceType === "youtube" ? "YouTube" : "File");
  } else {
    probeData = null;
    videoDurationSec = null;
    $("#video-info").hidden = true;
    updateEstimate();
  }
}

function persistActiveJob(jobId) {
  if (jobId) {
    localStorage.setItem(ACTIVE_JOB_KEY, jobId);
    syncJobUrl(jobId);
  } else {
    localStorage.removeItem(ACTIVE_JOB_KEY);
    syncJobUrl(null);
  }
}

function syncJobUrl(jobId) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState({}, "", url);
}

async function attachToJob(job, { resumePoll = false } = {}) {
  currentJobId = job.job_id || currentJobId;
  persistActiveJob(currentJobId);
  restoreJobInputs(job);
  progressWrap.classList.add("visible");
  perfPanel.classList.add("active");

  if (job.metrics) renderPerformance(enrichLiveMetrics(job) || job.metrics);

  if (job.status === "completed") {
    await finalizeCompletedJob(job);
    return;
  }

  if (job.status === "failed") {
    finalizeFailedJob(job, { clearPersist: false });
    return;
  }

  if (RUNNING_STATUSES.has(job.status)) {
    convertBtn.disabled = true;
    setProgress(job.progress, job.message || job.status, false, job);
    startLiveClock(job.created_at, job.status);
    if (resumePoll) pollJob();
  }
}

async function openHistoryJob(jobId) {
  try {
    const jobRes = await fetch(`/api/jobs/${jobId}`);
    if (!jobRes.ok) throw new Error("Job not found");
    const job = await jobRes.json();
    await attachToJob(job, { resumePoll: RUNNING_STATUSES.has(job.status) });
    await loadHistory();
  } catch (e) {
    setProgress(0, e.message, true);
  }
}

async function resumeActiveJob() {
  const params = new URLSearchParams(window.location.search);
  const jobId = params.get("job") || localStorage.getItem(ACTIVE_JOB_KEY);
  if (!jobId) return;

  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) {
      persistActiveJob(null);
      return;
    }
    const job = await res.json();
    await attachToJob(job, { resumePoll: RUNNING_STATUSES.has(job.status) });
  } catch {
    persistActiveJob(null);
  }
}

async function deleteHistoryJob(jobId) {
  if (!confirm("Delete this report from history?")) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Could not delete report");
    if (currentJobId === jobId) {
      currentJobId = null;
      persistActiveJob(null);
      resetOutput();
      setProgress(0, "Report deleted.");
    }
    await loadHistory();
  } catch (e) {
    setProgress(0, e.message, true);
  }
}

function fmtSec(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${Number(v).toFixed(2)}s`;
}

function fmtMs(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${Number(v).toFixed(0)} ms`;
}

const GANTT_TICKS = 4;

function ganttAxisTicks(totalSec) {
  const ticks = [];
  for (let i = 0; i <= GANTT_TICKS; i++) {
    ticks.push({ sec: (totalSec * i) / GANTT_TICKS, pct: (i / GANTT_TICKS) * 100 });
  }
  return ticks;
}

function computeGanttLayout(steps, elapsedSec, estimatedTotalSec) {
  let offset = 0;
  const rows = steps.map((step) => {
    const start = offset;
    let dur = step.duration_sec;
    if (dur == null && step.status === "running") {
      dur = Math.max(0.05, (elapsedSec ?? offset) - start);
    }
    if (dur != null) offset += dur;
    return { step, start_sec: start, display_dur: dur };
  });
  const summed = rows.reduce((sum, row) => sum + (row.display_dur ?? 0), 0);
  const total = Math.max(
    elapsedSec ?? 0,
    summed,
    estimatedTotalSec ?? 0,
    0.001,
  );
  return { rows, total };
}

function buildPlannedPipelineSteps(probe, sourceType, phase = "convert") {
  const scoutDetail =
    probe?.estimated_scout_calls != null
      ? { planned_calls: probe.estimated_scout_calls }
      : {};
  const analyzeDetail =
    probe?.estimated_analyze_calls != null
      ? { planned_frames: probe.estimated_analyze_calls }
      : probe?.max_frames != null
        ? { planned_frames: probe.max_frames }
        : {};

  const pipelineTail = [
    {
      id: "extract_frames",
      label: "Frame extraction (1 fps)",
      kind: "local",
      status: "pending",
      duration_sec: null,
    },
    {
      id: "transcript",
      label: "Transcription",
      kind: "local",
      status: "pending",
      duration_sec: null,
    },
    {
      id: "scout",
      label: "Scout frames (Gemma 4 · Cerebras)",
      kind: "cerebras",
      status: "pending",
      duration_sec: null,
      detail: scoutDetail,
    },
    {
      id: "analyze",
      label: "Frame analysis (Gemma 4 · Cerebras)",
      kind: "cerebras",
      status: "pending",
      duration_sec: null,
      detail: analyzeDetail,
    },
    {
      id: "compose",
      label: "Document writing (Gemma 4 · Cerebras)",
      kind: "cerebras",
      status: "pending",
      duration_sec: null,
    },
  ];

  if (sourceType === "youtube") {
    return [
      {
        id: "youtube_metadata",
        label: "YouTube metadata",
        kind: "local",
        status: phase === "probe" ? "running" : "done",
        duration_sec: phase === "probe" ? null : 0.01,
      },
      {
        id: "thumbnail_download",
        label: "YouTube thumbnail",
        kind: "local",
        status: phase === "probe" ? "pending" : "done",
        duration_sec: phase === "probe" ? null : 0.01,
      },
      {
        id: "youtube_download",
        label: "YouTube video download",
        kind: "local",
        status: phase === "convert" ? "running" : "pending",
        duration_sec: null,
      },
      {
        id: "video_probe",
        label: "Video probe (ingest)",
        kind: "local",
        status: "pending",
        duration_sec: null,
      },
      ...pipelineTail,
    ];
  }

  return [
    {
      id: "file_upload",
      label: "File upload",
      kind: "local",
      status: phase === "convert" ? "running" : "pending",
      duration_sec: null,
    },
    {
      id: "video_probe",
      label: "Video probe (ingest)",
      kind: "local",
      status: "pending",
      duration_sec: null,
    },
    ...pipelineTail,
  ];
}

function buildPlannedMetrics(probe, sourceType, phase = "convert", elapsedSec = 0.05) {
  const estimatedTotalSec =
    probe?.estimated_total_minutes != null
      ? probe.estimated_total_minutes * 60
      : probe?.estimated_pipeline_minutes != null
        ? probe.estimated_pipeline_minutes * 60
        : null;

  return {
    elapsed_sec: elapsedSec,
    estimated_total_minutes: probe?.estimated_total_minutes,
    estimated_pipeline_minutes: probe?.estimated_pipeline_minutes,
    steps: buildPlannedPipelineSteps(probe, sourceType, phase),
    cerebras: { calls: 0, wall_sec: 0 },
    api_calls: [],
    planned: true,
    estimated_total_sec: estimatedTotalSec,
  };
}

function stopPlannedGanttTimer() {
  if (plannedGanttTimer) {
    clearInterval(plannedGanttTimer);
    plannedGanttTimer = null;
  }
}

function wallElapsedSec(createdAt) {
  if (!createdAt) return null;
  const started = new Date(createdAt).getTime();
  if (Number.isNaN(started)) return null;
  return Math.max(0, (Date.now() - started) / 1000);
}

function stopLiveClock() {
  if (liveClockTimer) {
    clearInterval(liveClockTimer);
    liveClockTimer = null;
  }
  liveClockCreatedAt = null;
}

function startLiveClock(createdAt, status) {
  if (!createdAt || !RUNNING_STATUSES.has(status)) {
    stopLiveClock();
    return;
  }
  liveClockCreatedAt = createdAt;
  if (liveClockTimer) return;

  const tick = () => {
    if (!liveClockCreatedAt) return;
    const elapsed = wallElapsedSec(liveClockCreatedAt);
    if (elapsed == null) return;

    const base = lastMetrics || {
      steps: [],
      cerebras: { calls: 0, wall_sec: 0 },
      api_calls: [],
    };
    const metrics = {
      ...base,
      elapsed_sec: Math.max(base.elapsed_sec ?? 0, elapsed),
    };

    const elapsedLabel = fmtSec(metrics.elapsed_sec);
    $("#m-elapsed").textContent = elapsedLabel;
    setStatValues(["ps-total", "ps-total-full"], elapsedLabel);
    renderGanttChart(metrics);
  };

  tick();
  liveClockTimer = setInterval(tick, 1000);
}

function openPerfPanel({ scroll = true } = {}) {
  if (!perfPanel) return;
  perfPanel.open = true;
  perfPanel.classList.add("active");
  if (scroll) perfPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showBootstrapGantt(message = "Preparing…") {
  perfPanel.classList.add("active");
  renderPerformance(
    {
      elapsed_sec: 0.05,
      steps: [
        {
          id: "bootstrap",
          label: message,
          kind: "local",
          status: "running",
          duration_sec: null,
        },
      ],
      cerebras: { calls: 0, wall_sec: 0 },
      api_calls: [],
      planned: true,
    },
    { live: true },
  );
  openPerfPanel();
}

function showPlannedGantt(probe, sourceType, phase = "convert") {
  plannedGanttStartedAt = Date.now();
  stopPlannedGanttTimer();
  perfPanel.classList.add("active");

  const render = () => {
    const elapsed = Math.max(0.05, (Date.now() - plannedGanttStartedAt) / 1000);
    const elapsedLabel = fmtSec(elapsed);
    $("#m-elapsed").textContent = elapsedLabel;
    setStatValues(["ps-total", "ps-total-full"], elapsedLabel);
    renderPerformance(buildPlannedMetrics(probe, sourceType, phase, elapsed), {
      live: true,
    });
  };

  render();
  plannedGanttTimer = setInterval(render, 500);
  openPerfPanel();
}

function renderGanttChart(metrics) {
  const steps = metrics.steps || [];
  if (!steps.length) {
    $("#gantt-chart").innerHTML = "";
    return;
  }

  const estimatedTotalSec =
    metrics.estimated_total_sec ??
    (metrics.estimated_total_minutes != null
      ? metrics.estimated_total_minutes * 60
      : null);
  const { rows, total } = computeGanttLayout(
    steps,
    metrics.elapsed_sec,
    estimatedTotalSec,
  );
  const ticks = ganttAxisTicks(total);
  const axisHtml = ticks
    .map(
      (tick) =>
        `<span class="gantt-axis-tick" style="left:${tick.pct}%">${fmtSec(tick.sec)}</span>`
    )
    .join("");

  const rowsHtml = rows
    .map(({ step, start_sec, display_dur }) => {
      const isCerebras = step.kind === "cerebras";
      const kindClass = isCerebras ? "cerebras" : "local";
      const kindLabel = isCerebras ? "Cerebras" : "local";
      const durLabel = display_dur != null ? fmtSec(display_dur) : "…";
      const startPct = (start_sec / total) * 100;
      const widthPct =
        display_dur != null ? Math.max((display_dur / total) * 100, 0.35) : 0;
      const showBarLabel = widthPct >= 7;
      const detail = step.detail
        ? Object.entries(step.detail)
            .map(([k, v]) => `${k}: ${v}`)
            .join(" · ")
        : "";
      const barHtml =
        display_dur != null
          ? `<div class="gantt-bar gantt-bar--${kindClass}${
              step.status === "running" ? " gantt-bar--running" : ""
            }" style="left:${startPct}%;width:${widthPct}%" title="${step.label}: ${durLabel}">${
              showBarLabel ? `<span class="gantt-bar-label">${durLabel}</span>` : ""
            }</div>`
          : "";
      return `
        <div class="gantt-row step-item ${step.status || ""}${
          step.status === "pending" ? " gantt-row--pending" : ""
        }">
          <div class="gantt-row-label">
            <span class="gantt-status-dot" aria-hidden="true"></span>
            <div class="gantt-row-text">
              <span class="gantt-row-title">${step.label}</span>
              <span class="gantt-row-meta">${kindLabel}${detail ? ` · ${detail}` : ""}</span>
            </div>
            <span class="gantt-row-duration step-duration--${kindClass}">${durLabel}</span>
          </div>
          <div class="gantt-row-chart">${barHtml}</div>
        </div>`;
    })
    .join("");

  $("#gantt-chart").innerHTML = `
    <div class="gantt-header">
      <div class="gantt-header-label">Step</div>
      <div class="gantt-axis">${axisHtml}</div>
    </div>
    <div class="gantt-body">${rowsHtml}</div>
    <div class="gantt-legend">
      <span class="gantt-legend-item">
        <span class="gantt-legend-swatch gantt-legend-swatch--local"></span> Local
      </span>
      <span class="gantt-legend-item">
        <span class="gantt-legend-swatch gantt-legend-swatch--cerebras"></span> Cerebras
      </span>
      <span class="gantt-legend-total">Total ${fmtSec(total)}</span>
    </div>`;
}

function formatProgressEta(job) {
  const totalMin =
    job?.estimated_total_minutes ??
    job?.metrics?.estimated_total_minutes ??
    probeData?.estimated_total_minutes ??
    probeData?.estimated_pipeline_minutes;
  if (totalMin == null || Number.isNaN(Number(totalMin))) return "";

  const totalLabel = formatPipelineMinutes(totalMin);
  if (!totalLabel) return "";

  if (job?.status === "completed") {
    const elapsed = job.metrics?.elapsed_sec;
    return elapsed != null ? `Done in ${formatDuration(elapsed)}` : `Est. ${totalLabel} total`;
  }
  if (job?.status === "failed") return "";

  const remainingSec = job?.estimated_remaining_sec;
  if (remainingSec != null && remainingSec > 0) {
    return `Est. ${totalLabel} total · ~${formatDuration(remainingSec)} left`;
  }
  return `Est. ${totalLabel} total`;
}

function setProgress(pct, message, isError = false, job = null) {
  progressWrap.classList.add("visible");
  progressFill.style.width = `${pct}%`;
  progressPct.textContent = `${Math.round(pct)}%`;
  statusMsg.textContent = message;
  statusMsg.classList.toggle("error", isError);
  if (progressEta) {
    progressEta.textContent = formatProgressEta(job);
  }
}

function setStatValues(ids, text) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }
}

function updateHeroStats(metrics) {
  if (!metrics) return;
  const cerebras = metrics.cerebras || {};
  const tps = cerebras.avg_output_tokens_per_sec;
  const llm = metrics.cerebras_llm_sec ?? cerebras.wall_sec;
  const calls = cerebras.calls;
  if (tps != null) {
    $("#hero-tps").textContent = tps;
    setStatValues(["m-tps", "m-tps-full"], String(tps));
  }
  if (llm != null) {
    const llmLabel = fmtSec(llm);
    $("#hero-llm").textContent = llmLabel;
    setStatValues(["ps-cerebras", "ps-cerebras-full"], llmLabel);
    $("#m-llm").textContent = llmLabel;
  }
  if (calls != null) $("#hero-calls").textContent = String(calls);
  if (headerLive) headerLive.hidden = false;
}

function metricsRenderKey(metrics) {
  const calls = metrics.api_calls?.length ?? 0;
  const steps = metrics.steps?.length ?? 0;
  const elapsed = metrics.elapsed_sec ?? 0;
  const lastStep = metrics.steps?.[steps - 1];
  const lastStatus = lastStep?.status ?? "";
  const lastDetail = lastStep?.detail
    ? JSON.stringify(lastStep.detail)
    : "";
  return `${calls}|${steps}|${elapsed}|${lastStatus}|${lastDetail}`;
}

const LIVE_STEP_BY_STATUS = {
  pending: { id: "queued", label: "Queued", kind: "local" },
  extracting: { id: "extract_frames", label: "Extracting frames", kind: "local" },
  transcribing: { id: "transcript", label: "Transcription", kind: "local" },
  analyzing: { id: "scout", label: "Scout & analyze", kind: "cerebras" },
  composing: { id: "compose", label: "Writing document", kind: "cerebras" },
};

function enrichLiveMetrics(job) {
  if (!job?.metrics) return null;
  const metrics = {
    ...job.metrics,
    steps: [...(job.metrics.steps || [])],
  };

  const hasRunning = metrics.steps.some((s) => s.status === "running");
  const liveStep = LIVE_STEP_BY_STATUS[job.status];
  if (!hasRunning && liveStep && !metrics.steps.some((s) => s.id === liveStep.id)) {
    metrics.steps.push({
      ...liveStep,
      status: "running",
      duration_sec: null,
      detail: {},
    });
  }

  const wall = wallElapsedSec(job.created_at);
  if (wall != null) {
    metrics.elapsed_sec = Math.max(metrics.elapsed_sec ?? 0, wall);
    metrics.wall_elapsed_sec = wall;
  }
  return metrics;
}

function renderApiCallsTable(calls, { live = false } = {}) {
  const tbody = $("#api-table-body");
  if (!calls.length) {
    tbody.innerHTML = "";
    return;
  }

  let display = calls;
  let note = "";
  if (live && calls.length > LIVE_API_CALLS_CAP) {
    display = calls.slice(-LIVE_API_CALLS_CAP);
    note = `<tr class="api-table-note"><td colspan="7">Showing last ${LIVE_API_CALLS_CAP} of ${calls.length} calls — full table when done</td></tr>`;
  }

  tbody.innerHTML =
    note +
    display
      .map(
        (c) => `
      <tr>
        <td>${c.label || c.stage}</td>
        <td class="api-time--cerebras">${fmtSec(c.wall_sec)}</td>
        <td>${fmtSec(c.queue_time_sec)}</td>
        <td>${fmtMs(c.ttft_ms)}</td>
        <td>${c.output_tokens_per_sec ?? "—"}</td>
        <td>${c.prompt_tokens ?? "—"}</td>
        <td>${c.completion_tokens ?? "—"}</td>
      </tr>`
      )
      .join("");
}

function renderPerformance(metrics, { live = false } = {}) {
  if (!metrics) return;
  lastMetrics = metrics;
  perfPanel?.classList.add("active");

  const cerebras = metrics.cerebras || {};
  const elapsedLabel = fmtSec(metrics.elapsed_sec);
  const llmLabel = fmtSec(metrics.cerebras_llm_sec ?? cerebras.wall_sec);
  const tpsLabel =
    cerebras.avg_output_tokens_per_sec != null
      ? `${cerebras.avg_output_tokens_per_sec}`
      : "—";

  $("#m-elapsed").textContent = elapsedLabel;
  $("#m-llm").textContent = llmLabel;
  setStatValues(["ps-total", "ps-total-full"], elapsedLabel);
  setStatValues(["ps-cerebras", "ps-cerebras-full"], llmLabel);
  setStatValues(["m-tps", "m-tps-full"], tpsLabel);
  updateHeroStats(metrics);

  $("#ps-ingest").textContent = fmtSec(metrics.ingest_sec);
  $("#ps-local").textContent = fmtSec(metrics.local_prep_sec);
  $("#ps-queue").textContent = fmtSec(cerebras.queue_sec);
  const pauseSec = metrics.rate_limit?.total_pause_sec;
  $("#ps-pause").textContent = pauseSec != null ? fmtSec(pauseSec) : "—";
  $("#ps-ttft").textContent = fmtMs(cerebras.avg_ttft_ms);
  $("#ps-tokens").textContent =
    cerebras.completion_tokens != null ? `${cerebras.completion_tokens}` : "—";
  $("#ps-calls").textContent = cerebras.calls != null ? `${cerebras.calls}` : "—";

  const key = metricsRenderKey(metrics);
  if (key !== lastRenderedMetricsKey || live) {
    lastRenderedMetricsKey = key;
    renderGanttChart(metrics);
  }

  renderApiCallsTable(metrics.api_calls || [], { live });
  refreshExportMenuState();
}

function setExportMenuOpen(open) {
  if (!exportDropdown || !btnExportMenu) return;
  exportDropdown.hidden = !open;
  btnExportMenu.setAttribute("aria-expanded", open ? "true" : "false");
}

function refreshExportMenuState() {
  const hasDoc = !!lastMarkdown;
  const hasMetrics = !!lastMetrics?.elapsed_sec;
  if (btnDownloadMd) btnDownloadMd.disabled = !hasDoc;
  if (btnDownloadHtml) btnDownloadHtml.disabled = !hasDoc;
  if (btnDownloadPdf) btnDownloadPdf.disabled = !hasDoc;
  if (btnExportMetrics) btnExportMetrics.disabled = !hasMetrics;
  if (btnExportMenu) btnExportMenu.disabled = !hasDoc && !hasMetrics;
  if (btnExportMenu?.disabled) setExportMenuOpen(false);
}

function setExportButtonsEnabled(enabled) {
  if (btnDownloadMd) btnDownloadMd.disabled = !enabled;
  if (btnDownloadHtml) btnDownloadHtml.disabled = !enabled;
  if (btnDownloadPdf) btnDownloadPdf.disabled = !enabled;
  refreshExportMenuState();
}

function downloadExport(format) {
  if (!currentJobId) return;
  const a = document.createElement("a");
  a.href = `/api/jobs/${currentJobId}/export/${format}`;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function setReportView(view) {
  const isPreview = view === "preview";
  const isRaw = view === "raw";

  outputEmpty.style.display = isPreview && !lastMarkdown ? "flex" : "none";
  docPreview.classList.toggle("visible", isPreview && !!lastMarkdown);
  docRaw.classList.toggle("visible", isRaw && !!lastMarkdown);

  btnPreview?.classList.toggle("active", isPreview);
  btnRaw?.classList.toggle("active", isRaw);
}

function setChatEnabled(enabled) {
  if (chatEmpty) chatEmpty.hidden = enabled;
  if (chatPane) chatPane.hidden = !enabled;
  if (chatInput) chatInput.disabled = !enabled || chatSending;
  if (chatSend) chatSend.disabled = !enabled || chatSending;
  chatSuggestions?.querySelectorAll(".chat-chip").forEach((chip) => {
    chip.disabled = !enabled || chatSending;
  });
}

function enrichmentCardHtml(enrichment, { withApply = false } = {}) {
  if (!enrichment) return "";
  const title = escapeHtml(enrichment.title || "New section");
  const preview = escapeHtml((enrichment.markdown || "").slice(0, 600));
  let applyBtn = "";
  if (withApply) {
    const id = `enr-${++enrichmentIdSeq}`;
    enrichmentStore.set(id, {
      title: enrichment.title || "New section",
      markdown: enrichment.markdown || "",
    });
    applyBtn = `<button type="button" class="btn btn-secondary btn-sm chat-apply-btn" data-apply-id="${id}">Add to report</button>`;
  }
  return `
    <div class="chat-enrichment">
      <p class="chat-enrichment-title">Suggested section: ${title}</p>
      <pre class="chat-enrichment-preview">${preview}</pre>
      ${applyBtn}
    </div>`;
}

function renderChatMessages(messages) {
  if (!chatMessages) return;
  if (!messages.length) {
    chatMessages.innerHTML = '<li class="chat-status">Ask a question or request a new section for the report.</li>';
    return;
  }
  chatMessages.innerHTML = messages
    .map((msg) => {
      const role = msg.role === "user" ? "user" : "assistant";
      const enrichment =
        msg.enrichment && role === "assistant"
          ? enrichmentCardHtml(msg.enrichment, { withApply: true })
          : "";
      return `<li class="chat-msg chat-msg--${role}">${escapeHtml(msg.content || "")}${enrichment}</li>`;
    })
    .join("");
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function loadChatHistory(jobId = currentJobId) {
  if (!jobId || !chatMessages) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}/chat`);
    if (!res.ok) return;
    const data = await res.json();
    chatHasContext = !!data.has_context;
    if (chatContextWarn) chatContextWarn.hidden = chatHasContext;
    renderChatMessages(data.messages || []);
    setChatEnabled(true);
  } catch {
    renderChatMessages([]);
    setChatEnabled(!!lastMarkdown);
  }
}

async function applyChatEnrichment(title, markdown) {
  if (!currentJobId || !title || !markdown) return;
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/chat/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, markdown }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    await loadDocument(currentJobId);
    setReportView("preview");
    setProgress(100, "Section added to report.");
  } catch (e) {
    setProgress(0, e.message, true);
  }
}

async function sendChatMessage(text) {
  const message = text.trim();
  if (!message || !currentJobId || chatSending) return;

  chatSending = true;
  setChatEnabled(true);

  const prior = chatMessages.querySelectorAll(".chat-msg, .chat-status");
  if (!prior.length) chatMessages.innerHTML = "";

  const userLi = document.createElement("li");
  userLi.className = "chat-msg chat-msg--user";
  userLi.textContent = message;
  chatMessages.appendChild(userLi);

  const assistantLi = document.createElement("li");
  assistantLi.className = "chat-msg chat-msg--assistant chat-msg--streaming";
  assistantLi.textContent = "";
  chatMessages.appendChild(assistantLi);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  if (chatInput) chatInput.value = "";

  try {
    const res = await fetch(`/api/jobs/${currentJobId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let reply = "";
    let enrichment = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const data = JSON.parse(line.slice(6));
        if (data.token) {
          reply += data.token;
          assistantLi.textContent = reply;
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        if (data.done) {
          enrichment = data.enrichment || null;
        }
      }
    }

    assistantLi.classList.remove("chat-msg--streaming");
    if (enrichment) {
      assistantLi.insertAdjacentHTML(
        "beforeend",
        enrichmentCardHtml(enrichment, { withApply: true }),
      );
    }
    await loadChatHistory(currentJobId);
  } catch (e) {
    assistantLi.textContent = e.message;
    assistantLi.classList.add("chat-msg--error");
  } finally {
    chatSending = false;
    setChatEnabled(!!lastMarkdown);
  }
}

function stopPolling() {
  pollGeneration += 1;
  stopPlannedGanttTimer();
  stopLiveClock();
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function finalizeCompletedJob(job) {
  stopPolling();
  progressWrap.classList.add("visible");
  progressWrap.classList.add("done");
  setProgress(100, job.message || "Done — loading report…", false, job);
  convertBtn.disabled = false;
  renderPerformance(job.metrics || {}, { live: false });

  try {
    await loadDocument(currentJobId, { deferChat: true });
    setProgress(100, "Document ready.", false, job);
    persistActiveJob(null);
    void loadHistory();
    void loadChatHistory(currentJobId);
  } catch (e) {
    setProgress(
      100,
      `Done — refresh or open from history (${e.message})`,
      false,
      job,
    );
  }
}

function finalizeFailedJob(job, { clearPersist = true } = {}) {
  stopPolling();
  progressWrap.classList.remove("done");
  showFailedJobReport(job);
  setProgress(job.progress ?? 0, formatJobError(job.error), true, job);
  convertBtn.disabled = false;
  if (clearPersist) persistActiveJob(null);
  focusReportPanel();
}

function resetOutput() {
  stopPolling();
  lastRenderedMetricsKey = "";
  progressWrap.classList.remove("done");
  outputEmpty.style.display = "flex";
  docPreview.classList.remove("visible");
  docRaw.classList.remove("visible");
  if (perfPanel) {
    perfPanel.classList.remove("active");
    perfPanel.open = false;
  }
  $("#gantt-chart").innerHTML = "";
  $("#api-table-body").innerHTML = "";
  setStatValues(["hero-tps", "hero-llm", "hero-calls"], "—");
  setStatValues(["ps-total", "ps-total-full", "ps-cerebras", "ps-cerebras-full", "m-tps", "m-tps-full"], "—");
  if (headerLive) headerLive.hidden = true;
  setExportMenuOpen(false);
  refreshExportMenuState();
  lastMarkdown = "";
  lastMetrics = null;
  if (chatMessages) chatMessages.innerHTML = "";
  if (chatContextWarn) chatContextWarn.hidden = true;
  setChatEnabled(false);
  setReportView("preview");
}

async function probeSource(activePane) {
  if (activePane === "pane-youtube") {
    const url = youtubeInput.value.trim();
    if (!url) {
      throw new Error("Enter a YouTube URL.");
    }
    return probeYoutubeUrl(url);
  }

  if (!selectedFile) {
    throw new Error("Choose a video file first.");
  }
  const probe = await probeFile(selectedFile);
  if (!probe) {
    throw new Error("Could not read video metadata.");
  }
  return probe;
}

async function startConvert() {
  const activePane = document.querySelector(".source-pane.active").id;
  const sourceType = activePane === "pane-youtube" ? "youtube" : "file";

  convertBtn.disabled = true;
  resetOutput();
  showBootstrapGantt("Analyzing video metadata…");
  setProgress(1, "Analyzing video metadata…");

  let probe;
  try {
    probe = await probeSource(activePane);
    applyProbeData(probe);
    showVideoInfo(probe, sourceType === "youtube" ? "YouTube" : "File");
    showPlannedGantt(probe, sourceType, "probe");
  } catch (e) {
    stopPlannedGanttTimer();
    setProgress(0, e.message, true);
    convertBtn.disabled = false;
    return;
  }

  const form = new FormData();
  form.append("language", $("#language").value);
  const customPrompt = promptCustom?.value.trim();
  if (customPrompt) {
    form.append("custom_prompt", customPrompt);
  }

  if (activePane === "pane-upload") {
    form.append("file", selectedFile);
  } else {
    form.append("youtube_url", youtubeInput.value.trim());
  }

  setProgress(2, "Starting conversion…");
  showPlannedGantt(probe, sourceType, "convert");

  try {
    const res = await fetch("/api/convert", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    currentJobId = data.job_id;
    persistActiveJob(currentJobId);
    stopPlannedGanttTimer();
    pollJob();
  } catch (e) {
    stopPlannedGanttTimer();
    setProgress(0, e.message, true);
    convertBtn.disabled = false;
  }
}

function pollJob() {
  stopPolling();
  const generation = pollGeneration;
  let consecutiveErrors = 0;

  const tick = async () => {
    if (generation !== pollGeneration || !currentJobId) return;

    try {
      const res = await fetch(`/api/jobs/${currentJobId}`);
      if (!res.ok) throw new Error("Job not found");
      const job = await res.json();
      consecutiveErrors = 0;

      setProgress(job.progress, job.message || job.status, false, job);
      stopPlannedGanttTimer();
      startLiveClock(job.created_at, job.status);
      if (job.metrics) {
        renderPerformance(enrichLiveMetrics(job) || job.metrics, { live: true });
      }

      if (job.status === "completed") {
        await finalizeCompletedJob(job);
        return;
      }
      if (job.status === "failed") {
        finalizeFailedJob(job);
        return;
      }

      pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
    } catch (e) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= POLL_MAX_ERRORS) {
        stopPolling();
        setProgress(
          0,
          `Connection lost — refresh to check status (${e.message})`,
          true,
        );
        convertBtn.disabled = false;
        return;
      }
      pollTimer = setTimeout(tick, POLL_ERROR_RETRY_MS);
    }
  };

  tick();
}

function rewriteAssetUrls(md, jobId) {
  return md.replace(
    /!\[([^\]]*)\]\(assets\/([^)]+)\)/g,
    `![$1](/api/jobs/${jobId}/assets/$2)`
  );
}

function focusReportPanel() {
  if (!window.matchMedia("(max-width: 900px)").matches) return;
  document.querySelector(".panel--report")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadDocument(jobId = currentJobId, { deferChat = false } = {}) {
  const res = await fetch(`/api/jobs/${jobId}/document`);
  if (!res.ok) throw new Error("Could not load document");
  const data = await res.json();
  currentJobId = jobId;
  lastMarkdown = data.markdown;
  renderMarkdownPreview(lastMarkdown, jobId);
  docRaw.textContent = lastMarkdown;
  setExportButtonsEnabled(true);
  setChatEnabled(true);
  setReportView("preview");
  focusReportPanel();
  outputEmpty.style.display = "none";
  if (!deferChat) {
    await loadChatHistory(jobId);
  }
}

btnPreview?.addEventListener("click", () => setReportView("preview"));
btnRaw?.addEventListener("click", () => setReportView("raw"));

chatForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  sendChatMessage(chatInput?.value || "");
});

chatSuggestions?.addEventListener("click", (e) => {
  const chip = e.target.closest(".chat-chip");
  if (!chip || chip.disabled) return;
  const prompt = chip.dataset.prompt;
  if (chatInput) chatInput.value = prompt;
  sendChatMessage(prompt);
});

chatMessages?.addEventListener("click", (e) => {
  const btn = e.target.closest(".chat-apply-btn");
  if (!btn) return;
  const enr = enrichmentStore.get(btn.dataset.applyId);
  if (enr) applyChatEnrichment(enr.title, enr.markdown);
});

btnDownloadMd?.addEventListener("click", () => {
  if (!lastMarkdown) return;
  const blob = new Blob([lastMarkdown], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `sightline-${currentJobId?.slice(0, 8) || "export"}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
});

btnDownloadHtml?.addEventListener("click", () => downloadExport("html"));
btnDownloadPdf?.addEventListener("click", () => downloadExport("pdf"));

btnExportMenu?.addEventListener("click", (e) => {
  e.stopPropagation();
  if (btnExportMenu.disabled) return;
  setExportMenuOpen(exportDropdown?.hidden !== false);
});

exportDropdown?.addEventListener("click", (e) => e.stopPropagation());

document.addEventListener("click", () => setExportMenuOpen(false));

btnExportMetrics?.addEventListener("click", () => {
  if (!lastMetrics) return;
  setExportMenuOpen(false);
  const payload = {
    job_id: currentJobId,
    exported_at: new Date().toISOString(),
    provider: "cerebras",
    model: "gemma-4-31b",
    ...lastMetrics,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `sightline-metrics-${currentJobId?.slice(0, 8) || "export"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

[btnDownloadMd, btnDownloadHtml, btnDownloadPdf].forEach((btn) => {
  btn?.addEventListener("click", () => setExportMenuOpen(false));
});

historyList?.addEventListener("click", (e) => {
  const deleteBtn = e.target.closest("[data-action='delete']");
  if (deleteBtn) {
    e.stopPropagation();
    deleteHistoryJob(deleteBtn.dataset.jobId);
    return;
  }
  const item = e.target.closest(".history-item[data-openable='true']");
  if (item) openHistoryJob(item.dataset.jobId);
});

historyList?.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  const item = e.target.closest(".history-item[data-openable='true']");
  if (!item) return;
  e.preventDefault();
  openHistoryJob(item.dataset.jobId);
});

btnRefreshHistory?.addEventListener("click", loadHistory);

async function loadDefaults() {
  if (!promptDefault) return;
  try {
    const res = await fetch("/api/defaults");
    if (!res.ok) return;
    const data = await res.json();
    promptDefault.value = data.compose_prompt || "";
  } catch {
    promptDefault.placeholder = "Could not load default prompt.";
  }
}

function initHistoryPanel() {
  if (!historyBlock) return;
  historyBlock.open = localStorage.getItem(HISTORY_PANEL_KEY) === "1";
  historyBlock.addEventListener("toggle", () => {
    localStorage.setItem(HISTORY_PANEL_KEY, historyBlock.open ? "1" : "0");
  });
}

convertBtn.addEventListener("click", startConvert);
initHistoryPanel();
loadDefaults();
loadHistory();
resumeActiveJob();