const $ = id => document.getElementById(id);
const CHECKS_KEY = "cn.taskChecks";
const state = {
  file: null, ws: null, streams: [], displayStream: null, audioContext: null, processor: null, sources: [], mute: null,
  timer: null, startedAt: 0, transcript: "", summary: "", blocks: [], sampleRate: 48000, pcmQueue: [], queuedSamples: 0,
  activeMode: null, stopping: false, busy: false,
  checks: loadChecks(),
};
const WS_BASE = (location.protocol === "https:" ? "wss://" : "ws://") + location.host;

// ---------------------------------------------------------------- server + small helpers

async function checkServer() {
  try {
    const r = await fetch("/health", { cache: "no-store" });
    if (!r.ok) throw Error();
    $("serverState").className = "server-indicator ok";
    $("serverState").innerHTML = "<i></i><span>Server connected</span>";
  } catch {
    $("serverState").className = "server-indicator bad";
    $("serverState").innerHTML = "<i></i><span>Server unavailable</span>";
  }
}
checkServer();
setInterval(checkServer, 30000);

document.querySelectorAll(".source-tab").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".source-tab").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".source-panel").forEach(x => x.classList.toggle("active", x.id === "panel-" + b.dataset.tab));
}));

function setState(message, tone) {
  if (!tone) tone = /issue|fail|unavailable|closed|error/i.test(message) ? "error"
    : /listening|receiving|connected/i.test(message) ? "live"
    : /updating|processing|preparing|transcrib/i.test(message) ? "busy" : "idle";
  $("sessionState").textContent = message;
  $("sessionState").dataset.tone = tone;
}

// Only one capture/upload at a time: starting a second one would overwrite the shared audio state.
function setBusy(busy) {
  state.busy = busy;
  $("micStart").disabled = busy;
  $("meetingStart").disabled = busy;
  $("uploadBtn").disabled = busy || !state.file;
}

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function hasArabic(s) { return /[؀-ۿݐ-ݿࢠ-ࣿ]/.test(s || ""); }
function dirAttr(s) { return hasArabic(s) ? ' dir="rtl" lang="ar"' : ' dir="ltr"'; }
function inlineMd(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}
function plainText(s) { return String(s).replace(/[*`_]/g, "").replace(/\s+/g, " ").trim(); }
function isActionHeading(s) { return /(next steps?|action items?|to[- ]?do|follow[- ]?ups?|tasks?|الخطوات القادمة|الخطوات التالية|الإجراءات|المهام|المطلوب|التوصيات العملية)/i.test(s || ""); }
function clockTime() { return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

function loadChecks() { try { return JSON.parse(localStorage.getItem(CHECKS_KEY)) || {}; } catch { return {}; } }
function saveChecks() { try { localStorage.setItem(CHECKS_KEY, JSON.stringify(state.checks)); } catch {} }

// ---------------------------------------------------------------- markdown

// Parse the summary Markdown into blocks. The same blocks render the page and the .md export,
// so the export reflects which tasks were ticked.
function parseMarkdown(raw) {
  let text = (raw || "").replace(/\r/g, "").trim();
  const fence = text.match(/^```[a-z]*\s*\n([\s\S]*?)\n?```$/i);
  if (fence) text = fence[1].trim();

  const blocks = [];
  let para = [], list = null, inActions = false;
  const flushPara = () => { if (para.length) { blocks.push({ type: "p", text: para.join(" ") }); para = []; } };
  const closeList = () => { list = null; };

  for (const original of text.split("\n")) {
    const line = original.trim();
    if (!line) { flushPara(); continue; }

    const h = line.match(/^(#{1,6})\s+(.+?)\s*:?\s*$/) || line.match(/^\*\*([^*]{2,100})\*\*\s*:?\s*$/);
    if (h) {
      flushPara(); closeList();
      const title = h.length === 3 ? h[2] : h[1];
      const level = h.length === 3 ? Math.min(4, h[1].length) : 2;
      inActions = isActionHeading(title);
      blocks.push({ type: "h", level, text: title.replace(/:$/, "").trim() });
      continue;
    }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { flushPara(); closeList(); blocks.push({ type: "hr" }); continue; }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) { flushPara(); closeList(); blocks.push({ type: "quote", text: quote[1] }); continue; }

    const task = line.match(/^(?:[-*+•]|\d+[.)])\s+\[( |x|X)\]\s+(.+)$/);
    const bullet = line.match(/^[-*+•]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (task || bullet || numbered) {
      flushPara();
      const isTask = !!task || inActions;
      const ordered = !isTask && !!numbered;
      const itemText = task ? task[2] : (bullet || numbered)[1];
      if (!list || list.ordered !== ordered || list.tasks !== isTask) {
        list = { type: "list", ordered, tasks: isTask, items: [] };
        blocks.push(list);
      }
      list.items.push({ text: itemText, task: isTask, done: !!task && task[1].toLowerCase() === "x" });
      continue;
    }

    closeList();
    para.push(line);
  }
  flushPara();
  return blocks;
}

function taskKey(text) { return plainText(text).toLowerCase(); }
function isDone(item) { const k = taskKey(item.text); return k in state.checks ? state.checks[k] : item.done; }

function renderBlocks(blocks) {
  const html = [];
  let section = null;
  const closeSection = () => {
    if (!section) return;
    const tasks = section.tasks;
    const progress = tasks.length ? `<span class="task-progress" data-progress>${tasks.filter(isDone).length}/${tasks.length}</span>` : "";
    html.push(`<section class="summary-section${section.action ? " action-section" : ""}">${section.heading.replace("{{progress}}", progress)}${section.body.join("")}</section>`);
    section = null;
  };

  for (const b of blocks) {
    if (b.type === "h" && b.level === 1) {
      closeSection();
      html.push(`<header class="summary-title"><h1${dirAttr(b.text)}>${inlineMd(b.text)}</h1></header>`);
      continue;
    }
    if (b.type === "h" && b.level === 2 || !section) {
      closeSection();
      section = { heading: "", body: [], tasks: [], action: false };
      if (b.type === "h") {
        section.action = isActionHeading(b.text);
        section.heading = `<h2${dirAttr(b.text)}><span>${inlineMd(b.text)}</span>{{progress}}</h2>`;
        continue;
      }
    }
    if (b.type === "h") section.body.push(`<h${b.level}${dirAttr(b.text)}>${inlineMd(b.text)}</h${b.level}>`);
    else if (b.type === "p") section.body.push(`<p${dirAttr(b.text)}>${inlineMd(b.text)}</p>`);
    else if (b.type === "quote") section.body.push(`<blockquote${dirAttr(b.text)}>${inlineMd(b.text)}</blockquote>`);
    else if (b.type === "hr") section.body.push("<hr>");
    else if (b.type === "list") {
      const tag = b.ordered ? "ol" : "ul";
      const dir = dirAttr(b.items.map(i => i.text).join(" "));
      const items = b.items.map(item => {
        if (!item.task) return `<li${dirAttr(item.text)}>${inlineMd(item.text)}</li>`;
        section.tasks.push(item);
        const done = isDone(item);
        return `<li class="task-item${done ? " done" : ""}"><label><input type="checkbox" class="task-check" data-key="${esc(taskKey(item.text))}"${done ? " checked" : ""}><span>${inlineMd(item.text)}</span></label></li>`;
      }).join("");
      section.body.push(`<${tag} class="summary-list${b.tasks ? " task-list" : ""}"${dir}>${items}</${tag}>`);
    }
  }
  closeSection();
  return html.join("");
}

function blocksToMarkdown(blocks) {
  return blocks.map(b => {
    if (b.type === "h") return "#".repeat(b.level) + " " + b.text;
    if (b.type === "p") return b.text;
    if (b.type === "quote") return "> " + b.text;
    if (b.type === "hr") return "---";
    return b.items.map((item, i) => item.task ? `- [${isDone(item) ? "x" : " "}] ${item.text}` : b.ordered ? `${i + 1}. ${item.text}` : `- ${item.text}`).join("\n");
  }).join("\n\n");
}

function renderSummary(raw) {
  const v = (raw || "").trim();
  if (!v) {
    state.blocks = [];
    $("summary").className = "summary-content empty-summary";
    $("summary").innerHTML = '<div class="empty-mark">✳</div><h3>Your notes will take shape here.</h3><p>Once audio is processed, the key points, decisions and next steps will appear in this space.</p>';
    $("summaryUpdated").textContent = "No summary yet";
    return;
  }
  state.blocks = parseMarkdown(v);
  $("summary").className = "summary-content";
  $("summary").innerHTML = renderBlocks(state.blocks);
  $("summaryUpdated").textContent = "Updated " + clockTime();
}

$("summary").addEventListener("change", e => {
  const box = e.target.closest(".task-check");
  if (!box) return;
  state.checks[box.dataset.key] = box.checked;
  saveChecks();
  box.closest(".task-item").classList.toggle("done", box.checked);
  const section = box.closest(".summary-section");
  const progress = section && section.querySelector("[data-progress]");
  if (progress) {
    const all = section.querySelectorAll(".task-check");
    progress.textContent = `${[...all].filter(x => x.checked).length}/${all.length}`;
  }
});

// ---------------------------------------------------------------- results + export

function refreshExportButtons() {
  const empty = !state.transcript.trim() && !state.summary.trim();
  $("downloadMd").disabled = empty;
  $("copyMd").disabled = empty;
}
function updateTranscript(t) {
  state.transcript = t || "";
  const box = $("transcript");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.textContent = state.transcript || "No speech recognized yet.";
  if (atBottom) box.scrollTop = box.scrollHeight;
  const count = state.transcript.trim() ? state.transcript.trim().split(/\s+/).length : 0;
  $("transcriptCount").textContent = `${count} words · Expand to review the recognized speech`;
  refreshExportButtons();
}
function updateSummary(t) { state.summary = t || ""; renderSummary(state.summary); refreshExportButtons(); }
function resetResults() { updateTranscript(""); updateSummary(""); $("summaryUpdated").textContent = "Waiting for audio"; setState("Preparing audio"); }

function notesMarkdown() {
  const hasTitle = state.blocks[0] && state.blocks[0].type === "h" && state.blocks[0].level === 1;
  const parts = [];
  if (!hasTitle) parts.push("# Conversation notes");
  parts.push(state.blocks.length ? blocksToMarkdown(state.blocks) : "_No summary available._");
  parts.push("---", "## Full transcript", state.transcript.trim() || "_No transcript available._");
  parts.push(`---\n_Exported ${new Date().toLocaleString()} · Review names, dates and numbers against the original audio._`);
  return parts.join("\n\n") + "\n";
}
$("downloadMd").onclick = () => {
  const url = URL.createObjectURL(new Blob([notesMarkdown()], { type: "text/markdown;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `conversation-notes-${new Date().toISOString().slice(0, 10)}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("copyMd").onclick = async () => {
  const text = notesMarkdown();
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
  }
  const label = $("copyMd").textContent;
  $("copyMd").textContent = "Copied ✓";
  setTimeout(() => { $("copyMd").textContent = label; }, 1500);
};

// ---------------------------------------------------------------- upload

const fileInput = $("audioFile");
fileInput.onchange = () => selectFile(fileInput.files[0]);
function selectFile(f) {
  if (!f) return;
  state.file = f;
  $("fileName").textContent = f.name;
  $("fileSize").textContent = (f.size / 1048576).toFixed(2) + " MB";
  $("fileRow").hidden = false;
  $("dropTitle").textContent = "Recording selected";
  $("uploadBtn").disabled = state.busy;
}
$("clearFile").onclick = () => {
  state.file = null; fileInput.value = "";
  $("fileRow").hidden = true; $("uploadBtn").disabled = true;
  $("dropTitle").textContent = "Drop a recording here, or browse";
};
const dz = $("dropzone");
dz.ondragover = e => { e.preventDefault(); dz.classList.add("dragging"); };
dz.ondragleave = () => dz.classList.remove("dragging");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("dragging"); selectFile(e.dataTransfer.files[0]); };

$("uploadBtn").onclick = async () => {
  if (!state.file || state.busy) return;
  setBusy(true);
  $("uploadProgress").hidden = false;
  $("uploadProgress").querySelector(".progress-track").hidden = false;
  $("uploadStatus").textContent = "Uploading and preparing your notes… long recordings can take several minutes.";
  resetResults();
  setState("Processing recording");
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("language", $("uploadLanguage").value);
  fd.append("include_summary", "true");
  try {
    const r = await fetch("/transcribe", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({ detail: `Server returned ${r.status}` }));
    if (!r.ok) throw Error(data.detail || JSON.stringify(data));
    updateTranscript(data.text || data.transcript || "");
    updateSummary(data.summary || "");
    if (data.summary_error) $("summary").insertAdjacentHTML("beforeend", `<p class="summary-warning"><strong>Summary could not be generated:</strong> ${esc(data.summary_error)}</p>`);
    setState("Notes ready", "idle");
    $("uploadStatus").textContent = "Finished. Review the notes below.";
  } catch (e) {
    setState("Processing failed");
    $("uploadStatus").textContent = e.message;
  } finally {
    $("uploadProgress").querySelector(".progress-track").hidden = true;
    setBusy(false);
  }
};

// ---------------------------------------------------------------- live audio

function toPCM16(f) {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) { const s = Math.max(-1, Math.min(1, f[i])); out[i] = s < 0 ? s * 32768 : s * 32767; }
  return out.buffer;
}
function fmt(s) { return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }
function startClock(id) {
  state.startedAt = Date.now();
  clearInterval(state.timer);
  state.timer = setInterval(() => { $(id).textContent = fmt(Math.floor((Date.now() - state.startedAt) / 1000)); }, 500);
}
function sendQueue() {
  if (!state.pcmQueue.length || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const joined = new Float32Array(state.queuedSamples);
  let p = 0;
  for (const a of state.pcmQueue) { joined.set(a, p); p += a.length; }
  state.pcmQueue = []; state.queuedSamples = 0;
  state.ws.send(toPCM16(joined));
}

async function startAudio(mode, language, streams, timerId, levelId, onUnexpectedClose) {
  resetResults();
  state.streams = [].concat(streams); state.activeMode = mode; state.stopping = false;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw Error("Web Audio is not supported in this browser.");
  state.audioContext = new AC();
  await state.audioContext.resume();
  state.sampleRate = state.audioContext.sampleRate;

  state.ws = new WebSocket(WS_BASE + "/ws/live");
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = () => {
    state.ws.send(JSON.stringify({ type: "start", sample_rate: state.sampleRate, language, source: mode }));
    setState("Connected · listening");
  };
  state.ws.onmessage = e => {
    let m;
    try { m = JSON.parse(e.data); } catch { return; }
    if (m.type === "transcript") { updateTranscript(state.transcript ? state.transcript + "\n" + m.text : m.text); setState("Listening · transcript updated", "live"); }
    else if (m.type === "summary") { updateSummary(m.summary); setState(state.stopping ? "Finishing notes" : "Listening · notes updated", state.stopping ? "busy" : "live"); }
    else if (m.type === "status") setState(m.message);
    else if (m.type === "error") setState("Issue: " + m.message);
    else if (m.type === "finished") {
      updateTranscript(m.transcript || state.transcript);
      setState("Session finished · notes ready", "idle");
      state.ws.close();
    }
  };
  state.ws.onerror = () => setState("Live connection failed. Check server and network.");
  state.ws.onclose = () => {
    if (state.activeMode && !state.stopping) { setState("Connection closed by the server"); onUnexpectedClose(); }
  };

  state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
  // ScriptProcessor sums every connected source into its mono input, which mixes tab + mic audio.
  state.sources = state.streams.map(s => state.audioContext.createMediaStreamSource(s));
  state.pcmQueue = []; state.queuedSamples = 0;
  const threshold = state.sampleRate * 5;
  const meter = $(levelId);
  state.processor.onaudioprocess = e => {
    const d = e.inputBuffer.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; }
    meter.style.width = Math.min(100, Math.round(Math.sqrt(peak) * 100)) + "%";
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN || state.stopping) return;
    const copy = new Float32Array(d.length);
    copy.set(d);
    state.pcmQueue.push(copy); state.queuedSamples += copy.length;
    if (state.queuedSamples >= threshold) sendQueue();
  };
  state.sources.forEach(s => s.connect(state.processor));
  state.mute = state.audioContext.createGain(); state.mute.gain.value = 0;
  state.processor.connect(state.mute); state.mute.connect(state.audioContext.destination);
  startClock(timerId);
}

async function stopAudio(levelId) {
  if (state.stopping) return;
  state.stopping = true;
  clearInterval(state.timer); state.timer = null;
  try {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      sendQueue();
      state.ws.send(JSON.stringify({ type: "stop" }));
      setState("Preparing final notes…", "busy");
    }
    if (state.processor) { state.processor.onaudioprocess = null; state.processor.disconnect(); }
    state.sources.forEach(s => s.disconnect());
    if (state.mute) state.mute.disconnect();
    state.streams.forEach(s => s.getTracks().forEach(t => t.stop()));
    if (state.audioContext) await state.audioContext.close();
  } catch (e) { console.warn(e); }
  if (levelId) $(levelId).style.width = "0%";
  state.streams = []; state.processor = null; state.sources = []; state.mute = null; state.audioContext = null; state.activeMode = null;
}

function secureContextHelp() { return "Open this site over HTTPS (or use localhost). Browsers block microphone and screen-capture APIs on ordinary HTTP LAN addresses."; }
function mediaApiError(kind) {
  if (!window.isSecureContext) return secureContextHelp();
  if (!navigator.mediaDevices) return "This browser has not exposed mediaDevices. Try the latest Chrome or Edge and allow site permissions.";
  if (kind === "mic" && !navigator.mediaDevices.getUserMedia) return "Microphone capture is unavailable in this browser. Try the latest Chrome or Edge.";
  if (kind === "meeting" && !navigator.mediaDevices.getDisplayMedia) return "Screen/tab sharing is unavailable in this browser. Try the latest Chrome or Edge.";
  return "";
}
const MIC_CONSTRAINTS = { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } };

// ---------------------------------------------------------------- microphone

async function stopMic() {
  $("micStop").disabled = true;
  await stopAudio("micLevel");
  setBusy(false);
  $("micTitle").textContent = "Recording stopped";
  $("micHint").textContent = "Your notes are shown below. You can record again.";
  $("panel-mic").classList.remove("active-recording");
}
$("micStart").onclick = async () => {
  try {
    const problem = mediaApiError("mic");
    if (problem) throw Error(problem);
    setBusy(true);
    const s = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
    $("micTitle").textContent = "Microphone is active";
    $("micHint").textContent = "Audio is being sent to your private server. The first text appears after about 15 seconds of speech.";
    $("panel-mic").classList.add("active-recording");
    $("micStop").disabled = false;
    await startAudio("microphone", $("micLanguage").value, s, "micTimer", "micLevel", stopMic);
  } catch (e) {
    setState("Microphone unavailable: " + e.message);
    state.streams.forEach(s => s.getTracks().forEach(t => t.stop()));
    $("panel-mic").classList.remove("active-recording");
    $("micStop").disabled = true;
    setBusy(false);
  }
};
$("micStop").onclick = stopMic;

// ---------------------------------------------------------------- meeting tab

async function stopMeeting() {
  if (!state.displayStream && !state.activeMode) return;
  $("meetingStop").disabled = true;
  if (state.displayStream) state.displayStream.getTracks().forEach(t => t.stop());
  state.displayStream = null;
  await stopAudio("meetingLevel");
  setBusy(false);
  $("panel-meeting").classList.remove("active-recording");
  $("meetingTitle").textContent = "Meeting capture stopped";
  $("meetingHint").textContent = "Join a meeting in its tab, then share that tab again to continue.";
}
$("meetingStart").onclick = async () => {
  try {
    const problem = mediaApiError("meeting");
    if (problem) throw Error(problem);
    setBusy(true);
    const display = await navigator.mediaDevices.getDisplayMedia({
      video: { displaySurface: "browser" }, audio: true, preferCurrentTab: false,
      selfBrowserSurface: "exclude", surfaceSwitching: "include", systemAudio: "exclude",
    });
    const tracks = display.getAudioTracks();
    if (!tracks.length) {
      display.getTracks().forEach(t => t.stop());
      throw Error("The browser shared the tab without audio. In the sharing chooser, select the Google Meet tab and enable 'Share tab audio'. If that option is missing, use an up-to-date Chrome or Edge browser.");
    }
    state.displayStream = display;
    const streams = [new MediaStream(tracks)];
    let micNote = "";
    if ($("meetingIncludeMic").checked) {
      try { streams.push(await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS)); micNote = " Your microphone is mixed in."; }
      catch (e) { console.warn("Microphone not added to meeting capture:", e); micNote = " Microphone access was denied, so only other participants are captured."; }
    }
    $("meetingTitle").textContent = "Meeting tab audio is being captured";
    $("meetingHint").textContent = "Keep the shared tab open." + micNote;
    $("panel-meeting").classList.add("active-recording");
    $("meetingStop").disabled = false;
    display.getTracks().forEach(t => t.addEventListener("ended", stopMeeting));
    await startAudio("meeting-tab", $("meetingLanguage").value, streams, "meetingTimer", "meetingLevel", stopMeeting);
  } catch (e) {
    setState("Meeting capture unavailable: " + e.message);
    $("meetingStop").disabled = true;
    state.streams.forEach(s => s.getTracks().forEach(t => t.stop()));
    if (state.displayStream) state.displayStream.getTracks().forEach(t => t.stop());
    state.displayStream = null;
    $("panel-meeting").classList.remove("active-recording");
    setBusy(false);
  }
};
$("meetingStop").onclick = stopMeeting;

window.addEventListener("beforeunload", e => {
  if (state.activeMode) { e.preventDefault(); e.returnValue = ""; }
});
window.addEventListener("pagehide", () => {
  try {
    if (state.ws) state.ws.close();
    state.streams.forEach(s => s.getTracks().forEach(t => t.stop()));
    if (state.displayStream) state.displayStream.getTracks().forEach(t => t.stop());
  } catch {}
});
