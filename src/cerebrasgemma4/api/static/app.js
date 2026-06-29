const $ = (sel) => document.querySelector(sel);

const themeToggle = $("#theme-toggle");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("vid2doc-theme", theme);
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
const statusMsg = $("#status-msg");
const metricsEl = $("#metrics");
const perfPanel = $("#perf-panel");
const outputEmpty = $("#output-empty");
const docPreview = $("#doc-preview");
const docRaw = $("#doc-raw");
const btnPreview = $("#btn-preview");
const btnRaw = $("#btn-raw");
const btnDownloadMd = $("#btn-download-md");
const btnDownloadHtml = $("#btn-download-html");
const btnDownloadPdf = $("#btn-download-pdf");
const btnExportMetrics = $("#btn-export-metrics");
const historyList = $("#history-list");
const btnRefreshHistory = $("#btn-refresh-history");

let selectedFile = null;
let videoDurationSec = null;
let currentJobId = null;
let pollTimer = null;
let lastMarkdown = "";
let lastMetrics = null;

const HACKATHON_RPM = 100;

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panes.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.pane).classList.add("active");
  });
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
  const breakdown = `scout ${scout} + analyze ${analyze} + compose 1`;
  const durationNote = `Full video · ${formatDuration(probeData.duration_sec)} · ${probeData.max_frames} key frames`;
  const paceNote = formatPipelineMinutes(probeData.estimated_pipeline_minutes);
  box.classList.remove("warn");
  box.textContent = `${durationNote} · ~${total} Cerebras calls (${breakdown})${
    paceNote ? ` · est. ${paceNote} with rate limiting` : ""
  }`;
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
    ` · full video · ${meta.max_frames} key frames` +
    ` · ~${meta.estimated_total_api_calls} Cerebras calls` +
    (meta.estimated_pipeline_minutes
      ? ` · est. ${formatPipelineMinutes(meta.estimated_pipeline_minutes)}`
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

function renderHistory(jobs) {
  if (!historyList) return;
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
      const canOpen = job.status === "completed";
      const openable = canOpen ? ' data-openable="true" role="button" tabindex="0"' : "";
      return `
        <li class="history-item${active}${canOpen ? " history-item--openable" : ""}" data-job-id="${job.job_id}"${openable}>
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

async function openHistoryJob(jobId) {
  currentJobId = jobId;
  try {
    const jobRes = await fetch(`/api/jobs/${jobId}`);
    if (!jobRes.ok) throw new Error("Job not found");
    const job = await jobRes.json();
    if (job.status !== "completed") {
      setProgress(job.progress, job.message || job.status);
      return;
    }
    if (job.metrics) renderPerformance(job.metrics);
    await loadDocument(jobId);
    await loadHistory();
  } catch (e) {
    setProgress(0, e.message, true);
  }
}

async function deleteHistoryJob(jobId) {
  if (!confirm("Delete this report from history?")) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Could not delete report");
    if (currentJobId === jobId) {
      currentJobId = null;
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

function setProgress(pct, message, isError = false) {
  progressWrap.classList.add("visible");
  progressFill.style.width = `${pct}%`;
  progressPct.textContent = `${Math.round(pct)}%`;
  statusMsg.textContent = message;
  statusMsg.classList.toggle("error", isError);
}

function updateHeroStats(metrics) {
  if (!metrics) return;
  const cerebras = metrics.cerebras || {};
  const tps = cerebras.avg_output_tokens_per_sec;
  const llm = metrics.cerebras_llm_sec ?? cerebras.wall_sec;
  const calls = cerebras.calls;
  if (tps != null) $("#hero-tps").textContent = tps;
  if (llm != null) $("#hero-llm").textContent = fmtSec(llm);
  if (calls != null) $("#hero-calls").textContent = String(calls);
}

function renderPerformance(metrics) {
  if (!metrics) return;
  lastMetrics = metrics;
  perfPanel.classList.add("active");
  metricsEl.classList.add("visible");

  const cerebras = metrics.cerebras || {};
  $("#m-elapsed").textContent = fmtSec(metrics.elapsed_sec);
  $("#m-llm").textContent = fmtSec(metrics.cerebras_llm_sec ?? cerebras.wall_sec);
  $("#m-tps").textContent =
    cerebras.avg_output_tokens_per_sec != null
      ? `${cerebras.avg_output_tokens_per_sec}`
      : "—";
  updateHeroStats(metrics);

  $("#ps-ingest").textContent = fmtSec(metrics.ingest_sec);
  $("#ps-local").textContent = fmtSec(metrics.local_prep_sec);
  $("#ps-cerebras").textContent = fmtSec(metrics.cerebras_llm_sec ?? cerebras.wall_sec);
  $("#ps-ttft").textContent = fmtMs(cerebras.avg_ttft_ms);
  $("#ps-tokens").textContent =
    cerebras.completion_tokens != null ? `${cerebras.completion_tokens}` : "—";
  $("#ps-calls").textContent = cerebras.calls != null ? `${cerebras.calls}` : "—";

  const steps = metrics.steps || [];
  const maxDur = Math.max(...steps.map((s) => s.duration_sec || 0), 0.001);
  const timeline = $("#step-timeline");
  timeline.innerHTML = steps
    .map((step) => {
      const dur = step.duration_sec;
      const durLabel = dur != null ? fmtSec(dur) : "…";
      const pct = dur != null ? Math.max(4, (dur / maxDur) * 100) : 0;
      const detail = step.detail
        ? Object.entries(step.detail)
            .map(([k, v]) => `${k}: ${v}`)
            .join(" · ")
        : "";
      const kindLabel = step.kind === "cerebras" ? "Cerebras" : "local";
      return `
        <li class="step-item ${step.status || ""}">
          <span class="step-dot" aria-hidden="true"></span>
          <div class="step-body">
            <span class="step-label">${step.label}</span>
            <span class="step-meta">${kindLabel}${detail ? ` · ${detail}` : ""}</span>
            <div class="step-bar-wrap"><div class="step-bar" style="width:${pct}%"></div></div>
          </div>
          <span class="step-duration">${durLabel}</span>
        </li>`;
    })
    .join("");

  const calls = metrics.api_calls || [];
  const tbody = $("#api-table-body");
  tbody.innerHTML = calls
    .map(
      (c) => `
      <tr>
        <td>${c.label || c.stage}</td>
        <td>${fmtSec(c.wall_sec)}</td>
        <td>${fmtMs(c.ttft_ms)}</td>
        <td>${c.output_tokens_per_sec ?? "—"}</td>
        <td>${c.prompt_tokens ?? "—"}</td>
        <td>${c.completion_tokens ?? "—"}</td>
      </tr>`
    )
    .join("");

  btnExportMetrics.disabled = !metrics.elapsed_sec;
}

function setExportButtonsEnabled(enabled) {
  if (btnDownloadMd) btnDownloadMd.disabled = !enabled;
  if (btnDownloadHtml) btnDownloadHtml.disabled = !enabled;
  if (btnDownloadPdf) btnDownloadPdf.disabled = !enabled;
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

function resetOutput() {
  outputEmpty.style.display = "block";
  docPreview.classList.remove("visible");
  docRaw.classList.remove("visible");
  metricsEl.classList.remove("visible");
  perfPanel.classList.remove("active");
  $("#step-timeline").innerHTML = "";
  $("#api-table-body").innerHTML = "";
  $("#hero-tps").textContent = "—";
  $("#hero-llm").textContent = "—";
  $("#hero-calls").textContent = "—";
  setExportButtonsEnabled(false);
  btnExportMetrics.disabled = true;
  lastMarkdown = "";
  lastMetrics = null;
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

  convertBtn.disabled = true;
  resetOutput();
  perfPanel.classList.add("active");
  setProgress(1, "Analyzing video metadata…");

  let probe;
  try {
    probe = await probeSource(activePane);
    applyProbeData(probe);
    showVideoInfo(probe, activePane === "pane-youtube" ? "YouTube" : "File");
  } catch (e) {
    setProgress(0, e.message, true);
    convertBtn.disabled = false;
    return;
  }

  const form = new FormData();
  form.append("language", $("#language").value);

  if (activePane === "pane-upload") {
    form.append("file", selectedFile);
  } else {
    form.append("youtube_url", youtubeInput.value.trim());
  }

  setProgress(2, "Starting conversion…");

  try {
    const res = await fetch("/api/convert", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    currentJobId = data.job_id;
    pollJob();
  } catch (e) {
    setProgress(0, e.message, true);
    convertBtn.disabled = false;
  }
}

function pollJob() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${currentJobId}`);
      if (!res.ok) throw new Error("Job not found");
      const job = await res.json();
      setProgress(job.progress, job.message || job.status);
      if (job.metrics) renderPerformance(job.metrics);

      if (job.status === "completed") {
        clearInterval(pollTimer);
        renderPerformance(job.metrics || {});
        await loadDocument();
        await loadHistory();
        convertBtn.disabled = false;
      } else if (job.status === "failed") {
        clearInterval(pollTimer);
        setProgress(job.progress, job.error || "Pipeline failed", true);
        convertBtn.disabled = false;
      }
    } catch (e) {
      clearInterval(pollTimer);
      setProgress(0, e.message, true);
      convertBtn.disabled = false;
    }
  }, 800);
}

function rewriteAssetUrls(md, jobId) {
  return md.replace(
    /!\[([^\]]*)\]\(assets\/([^)]+)\)/g,
    `![$1](/api/jobs/${jobId}/assets/$2)`
  );
}

async function loadDocument(jobId = currentJobId) {
  const res = await fetch(`/api/jobs/${jobId}/document`);
  if (!res.ok) throw new Error("Could not load document");
  const data = await res.json();
  currentJobId = jobId;
  lastMarkdown = data.markdown;
  const html = marked.parse(rewriteAssetUrls(lastMarkdown, jobId));
  docPreview.innerHTML = html;
  docRaw.textContent = lastMarkdown;
  outputEmpty.style.display = "none";
  docPreview.classList.add("visible");
  docRaw.classList.remove("visible");
  btnPreview.classList.add("active");
  btnRaw.classList.remove("active");
  setExportButtonsEnabled(true);
  setProgress(100, "Document ready.");
}

btnPreview.addEventListener("click", () => {
  docPreview.classList.add("visible");
  docRaw.classList.remove("visible");
  btnPreview.classList.add("active");
  btnRaw.classList.remove("active");
});

btnRaw.addEventListener("click", () => {
  docPreview.classList.remove("visible");
  docRaw.classList.add("visible");
  btnRaw.classList.add("active");
  btnPreview.classList.remove("active");
});

btnDownloadMd?.addEventListener("click", () => {
  if (!lastMarkdown) return;
  const blob = new Blob([lastMarkdown], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `vid2doc-${currentJobId?.slice(0, 8) || "export"}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
});

btnDownloadHtml?.addEventListener("click", () => downloadExport("html"));
btnDownloadPdf?.addEventListener("click", () => downloadExport("pdf"));

btnExportMetrics.addEventListener("click", () => {
  if (!lastMetrics) return;
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
  a.download = `vid2doc-metrics-${currentJobId?.slice(0, 8) || "export"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
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

convertBtn.addEventListener("click", startConvert);
loadHistory();