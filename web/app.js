const $ = id => document.getElementById(id);
const state = {
  file: null, ws: null, streams: [], displayStream: null, audioContext: null, processor: null, sources: [], mute: null,
  timer: null, startedAt: 0, transcript: "", summary: "", blocks: [], sampleRate: 48000, pcmQueue: [], queuedSamples: 0,
  activeMode: null, stopping: false, busy: false,
  user: null, noteId: null, noteTitle: "", notes: [], checks: {}, checksTimer: null, watchNoteId: null, notesPoll: null,
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

// ---------------------------------------------------------------- layout: capture sheet, sidebar, export menu

// The app bar tabs open the capture sheet under it; clicking the open tab again closes it.
function showCapture(tab) {
  const open = !!tab;
  $("captureSheet").hidden = !open;
  document.querySelectorAll(".source-tab").forEach(x => {
    const on = x.dataset.tab === tab;
    x.classList.toggle("active", on);
    x.setAttribute("aria-expanded", String(on));
  });
  document.querySelectorAll(".source-panel").forEach(x => x.classList.toggle("active", x.id === "panel-" + tab));
}
document.querySelectorAll(".source-tab").forEach(b => b.addEventListener("click", () => {
  showCapture(b.classList.contains("active") ? null : b.dataset.tab);
}));
function markLiveTab(mode) {
  const tab = { microphone: "mic", "meeting-tab": "meeting" }[mode];
  document.querySelectorAll(".source-tab").forEach(x => x.classList.toggle("live", x.dataset.tab === tab));
}

function setSidebar(open) { document.body.classList.toggle("sidebar-open", open); }
$("menuBtn").onclick = () => setSidebar(!document.body.classList.contains("sidebar-open"));
$("sidebarScrim").onclick = () => setSidebar(false);

function setExportMenu(open) {
  $("exportPop").hidden = !open;
  $("exportBtn").setAttribute("aria-expanded", String(open));
}
$("exportBtn").onclick = e => { e.stopPropagation(); setExportMenu($("exportPop").hidden); };
$("exportPop").addEventListener("click", () => setExportMenu(false));
document.addEventListener("click", e => { if (!e.target.closest(".export-menu")) setExportMenu(false); });
document.addEventListener("keydown", e => { if (e.key === "Escape") { setExportMenu(false); setSidebar(false); } });

$("newNote").onclick = () => {
  setSidebar(false);
  if (state.busy) { setState("Finish the current session before starting a new note", "error"); return; }
  resetResults();
  setState("Waiting for audio", "idle");
  showCapture("upload");
};
$("noteSearch").addEventListener("input", () => renderLibrary());

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

// Ticked tasks belong to the open note and are saved to the account (debounced).
function saveChecks() {
  if (!state.noteId) return;
  const noteId = state.noteId, checks = { ...state.checks };
  clearTimeout(state.checksTimer);
  state.checksTimer = setTimeout(() => {
    api(`/api/notes/${noteId}`, { method: "PATCH", json: { checks } }).catch(e => console.warn("Could not save ticks:", e));
  }, 600);
}

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
    $("summary").innerHTML = '<div class="empty-mark">✳</div><h3>Add audio to start this note</h3><p>Upload a recording, record with your microphone or capture a meeting from the bar above. The summary, decisions and next steps appear here.</p>';
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
  $("downloadHtml").disabled = !state.summary.trim();
  $("copyMd").disabled = empty;
}
function updateTranscript(t) {
  state.transcript = t || "";
  const box = $("transcript");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.textContent = state.transcript || "No speech recognized yet.";
  if (atBottom) box.scrollTop = box.scrollHeight;
  const count = state.transcript.trim() ? state.transcript.trim().split(/\s+/).length : 0;
  $("transcriptCount").textContent = count ? `${count} words` : "Nothing recognized yet";
  refreshExportButtons();
}
function updateSummary(t) { state.summary = t || ""; renderSummary(state.summary); refreshExportButtons(); }
function resetResults() {
  state.noteId = null; state.checks = {};
  setNoteTitle("");
  updateTranscript(""); updateSummary("");
  $("summaryUpdated").textContent = "Waiting for audio";
  setState("Preparing audio");
  renderLibrary();
}

function notesMarkdown() {
  const hasTitle = state.blocks[0] && state.blocks[0].type === "h" && state.blocks[0].level === 1;
  const parts = [];
  if (!hasTitle) parts.push("# " + (state.noteTitle || "Conversation notes"));
  parts.push(state.blocks.length ? blocksToMarkdown(state.blocks) : "_No summary available._");
  parts.push("---", "## Full transcript", state.transcript.trim() || "_No transcript available._");
  parts.push(`---\n_Exported ${new Date().toLocaleString()} · Review names, dates and numbers against the original audio._`);
  return parts.join("\n\n") + "\n";
}
function downloadFile(content, type, extension) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  const name = (state.noteTitle || "conversation-notes").replace(/[\\/:*?"<>|\s]+/g, "-").slice(0, 80);
  a.download = `${name}-${new Date().toISOString().slice(0, 10)}.${extension}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("downloadMd").onclick = () => downloadFile(notesMarkdown(), "text/markdown;charset=utf-8", "md");
$("downloadHtml").onclick = () => downloadFile(notesHtml(), "text/html;charset=utf-8", "html");

// A self-contained page (styles inline, no scripts) that anyone can open in a browser.
const SHARE_CSS = `
:root{--ink:#252923;--muted:#777a70;--line:#dfded5;--olive:#59664a;--olive-pale:#edf0e8;--paper:#f3f1eb}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.9 "DM Sans","Noto Sans Arabic",Tahoma,Arial,sans-serif}
.page{max-width:860px;margin:0 auto;padding:44px 24px 40px}
.meta{font-size:12px;color:var(--muted);margin:0 0 18px;letter-spacing:.3px}
.paper{background:#fffefa;border:1px solid var(--line);border-radius:6px;padding:30px 34px;box-shadow:0 10px 30px rgba(38,43,32,.05)}
h1,h2,h3,h4{font-family:"Newsreader","Noto Sans Arabic",Georgia,serif;line-height:1.3;color:#293126}
.summary-title{margin:0 0 22px;padding-bottom:16px;border-bottom:1px solid var(--line)}.summary-title h1{font-size:30px;margin:0}
.summary-section{padding:0 0 18px;margin:0 0 20px;border-bottom:1px solid #e8e6dd}.summary-section:last-child{border:0;margin:0;padding:0}
.summary-section>h2{display:flex;justify-content:space-between;align-items:baseline;gap:12px;font-size:22px;margin:0 0 10px;padding-bottom:9px;border-bottom:2px solid #e7e8df}
.task-progress{font:600 11px Arial,sans-serif;color:var(--olive);background:var(--olive-pale);border-radius:20px;padding:3px 10px}
p{margin:10px 0 14px}ul,ol{margin:10px 0 16px;padding-inline-start:28px}li{margin:7px 0}li::marker{color:var(--olive)}
strong{font-weight:800}code{font-family:Consolas,monospace;background:#efefe7;padding:1px 5px;border-radius:3px}
blockquote{margin:12px 0;padding:9px 15px;border-inline-start:3px solid #9aa58d;background:#f7f7f1;color:#60665a}hr{border:0;border-top:1px solid var(--line)}
[dir="rtl"]{direction:rtl;text-align:right;font-family:"Noto Sans Arabic",Tahoma,Arial,sans-serif;line-height:2}[dir="ltr"]{direction:ltr;text-align:left}
.action-section{background:#f8f8f3;border:1px solid #e2e4d9!important;border-radius:7px;padding:17px 20px 19px!important}
.task-list{list-style:none;padding:0}.task-item{margin:0;padding:8px 6px}.task-item+.task-item{border-top:1px solid #ebebe3}
.task-item label{display:flex;align-items:flex-start;gap:11px}.task-item input{width:17px;height:17px;flex:none;margin:6px 0 0;accent-color:var(--olive)}
.task-item input:checked+span{text-decoration:line-through;color:#888d81}
.note{font-size:12px;color:var(--muted);margin-top:16px}
@media print{body{background:#fff}.paper{border:0;box-shadow:none;padding:0}}
@media(max-width:600px){.paper{padding:20px 18px}}`;

function notesHtml() {
  const hasTitle = state.blocks[0] && state.blocks[0].type === "h" && state.blocks[0].level === 1;
  const title = hasTitle ? state.blocks[0].text : (state.noteTitle || "Conversation notes");
  const titleBlock = hasTitle ? "" : `<header class="summary-title"><h1${dirAttr(title)}>${inlineMd(title)}</h1></header>`;
  const summary = state.blocks.length ? renderBlocks(state.blocks) : "<p>No summary available.</p>";
  const rtl = hasArabic(title + state.summary.slice(0, 400));
  // Summary only: the raw transcript is not part of the shared page.
  return `<!doctype html>
<html lang="${rtl ? "ar" : "en"}" dir="${rtl ? "rtl" : "ltr"}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(plainText(title))}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Noto+Sans+Arabic:wght@400;600;700&display=swap">
<style>${SHARE_CSS}</style>
</head>
<body>
<main class="page">
<p class="meta" dir="ltr">CONVERSATION NOTES · ${esc(new Date().toLocaleDateString([], { dateStyle: "long" }))}</p>
<article class="paper">${titleBlock}${summary}</article>
<p class="note" dir="ltr">Generated automatically from a recording. Names, dates and numbers may contain recognition errors.</p>
</main>
</body>
</html>
`;
}
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

// ---------------------------------------------------------------- accounts

async function api(path, { json, ...opts } = {}) {
  if (json !== undefined) { opts.body = JSON.stringify(json); opts.headers = { "Content-Type": "application/json" }; }
  const r = await fetch(path, { cache: "no-store", ...opts });
  const data = await r.json().catch(() => ({}));
  // A wrong password on the sign-in form is a 401 too; that must not reset the form.
  if (r.status === 401 && !/^\/api\/(login|register)$/.test(path)) showAuth();
  if (!r.ok) throw Error(typeof data.detail === "string" ? data.detail : `Server returned ${r.status}`);
  return data;
}

let authMode = "login";
function setAuthMode(mode) {
  authMode = mode;
  const login = mode === "login";
  $("authKicker").textContent = login ? "Welcome back" : "New account";
  $("authTitle").textContent = login ? "Sign in to your notes" : "Create your account";
  $("authCopy").textContent = login
    ? "Your recordings, transcripts and summaries are saved to your account on this server."
    : "Choose a username (letters, numbers, _ . -) and a password of at least 8 characters.";
  $("authSubmit").textContent = login ? "Sign in" : "Create account";
  $("authSwitchText").textContent = login ? "New here?" : "Already have an account?";
  $("authSwitch").textContent = login ? "Create an account" : "Sign in instead";
  $("authPassword").autocomplete = login ? "current-password" : "new-password";
  $("authError").hidden = true;
}
$("authSwitch").onclick = () => setAuthMode(authMode === "login" ? "register" : "login");
$("authForm").onsubmit = async e => {
  e.preventDefault();
  $("authSubmit").disabled = true;
  $("authError").hidden = true;
  try {
    const user = await api(authMode === "login" ? "/api/login" : "/api/register", {
      method: "POST", json: { username: $("authUsername").value, password: $("authPassword").value },
    });
    $("authPassword").value = "";
    showApp(user);
  } catch (err) {
    $("authError").textContent = err.message;
    $("authError").hidden = false;
  } finally {
    $("authSubmit").disabled = false;
  }
};

function showAuth() {
  state.user = null;
  $("appView").hidden = true;
  $("accountBox").hidden = true;
  $("authView").hidden = false;
  $("authUsername").focus();
}
function showApp(user) {
  state.user = user;
  $("accountName").textContent = user.username;
  $("accountAvatar").textContent = user.username.slice(0, 1).toUpperCase();
  $("accountBox").hidden = false;
  $("authView").hidden = true;
  $("appView").hidden = false;
  // Reopen the note named in the address (#note=12) after a reload. Read it
  // before setNoteTitle below rewrites the address.
  const linked = Number((location.hash.match(/^#note=(\d+)$/) || [])[1]);
  if (!state.noteId) { updateTranscript(""); updateSummary(""); setNoteTitle(""); }
  refreshNotes();
  if (linked && !state.noteId) openNote(linked);
}
$("logoutBtn").onclick = async () => {
  if (state.activeMode) {
    if (!confirm("A live session is running. Stop it and sign out?")) return;
    await (state.activeMode === "microphone" ? stopMic() : stopMeeting());
  }
  await api("/api/logout", { method: "POST" }).catch(() => {});
  resetResults();
  setState("Waiting for audio", "idle");
  state.notes = [];
  renderLibrary();
  showAuth();
};

// ---------------------------------------------------------------- saved notes

const SOURCE_LABELS = { upload: "Upload", microphone: "Microphone", "meeting-tab": "Meeting", import: "Imported" };
const STATUS_LABELS = { processing: "Processing…", live: "Recording", failed: "Failed" };
function formatDate(ts) { return new Date(ts * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); }

function setNoteTitle(title) {
  state.noteTitle = title || "";
  history.replaceState(null, "", state.noteId ? `#note=${state.noteId}` : location.pathname);
  $("noteTitle").textContent = title || "New note";
  // The whole header (title, saved time, Rename) follows the title's language.
  $("noteTitle").closest(".note-head").dir = hasArabic(title) ? "rtl" : "ltr";
  $("renameNote").hidden = !state.noteId;
}

async function refreshNotes() {
  if (!state.user) return;
  try {
    state.notes = await api("/api/notes");
  } catch {
    $("libraryList").innerHTML = '<p class="library-empty">Could not load your notes. Check the server connection and reload.</p>';
    return;
  }
  const current = state.notes.find(n => n.id === state.noteId);
  if (current) setNoteTitle(current.title);
  renderLibrary();

  // An upload this page started has finished on the server: show the result.
  const watched = state.notes.find(n => n.id === state.watchNoteId);
  if (watched && watched.status !== "processing") {
    state.watchNoteId = null;
    $("uploadProgress").querySelector(".progress-track").hidden = true;
    $("uploadStatus").textContent = watched.status === "failed" ? "Processing failed. Open the note for details." : "Finished. Your notes are saved.";
    if (state.noteId === watched.id && !state.busy) openNote(watched.id, { scroll: false });
  }
  // Keep polling while the server is still working on something.
  clearTimeout(state.notesPoll);
  if (state.notes.some(n => n.status === "processing" || n.status === "live")) state.notesPoll = setTimeout(refreshNotes, 5000);
}

// Sidebar groups: notes are listed newest first, so each group is one contiguous run.
function dateGroup(ts) {
  const day = d => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(new Date()) - day(new Date(ts * 1000))) / 86400000);
  return days <= 0 ? "Today" : days === 1 ? "Yesterday" : days < 7 ? "Previous 7 days" : "Earlier";
}
function listTime(ts) {
  const d = new Date(ts * 1000);
  return dateGroup(ts) === "Today" || dateGroup(ts) === "Yesterday"
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { day: "numeric", month: "short" });
}

function renderLibrary() {
  if (!state.notes.length) {
    $("libraryList").innerHTML = '<p class="library-empty">No notes yet. Upload a recording or start recording, and the result is saved here.</p>';
    return;
  }
  const query = $("noteSearch").value.trim().toLowerCase();
  const notes = query ? state.notes.filter(n => (n.title + " " + (n.preview || "")).toLowerCase().includes(query)) : state.notes;
  if (!notes.length) {
    $("libraryList").innerHTML = `<p class="library-empty">No notes match “${esc(query)}”.</p>`;
    return;
  }
  let group = null;
  $("libraryList").innerHTML = notes.map(n => {
    const g = dateGroup(n.updated_at);
    const header = g !== group ? `<div class="library-group">${g}</div>` : "";
    group = g;
    const status = STATUS_LABELS[n.status] ? `<em class="note-status" data-status="${esc(n.status)}">${STATUS_LABELS[n.status]}</em>` : "";
    return `${header}
    <div class="library-item${n.id === state.noteId ? " active" : ""}">
      <button class="library-open" type="button" data-open="${n.id}" title="${esc(n.title)}">
        <b${dirAttr(n.title)}>${esc(n.title)}</b>
        <span class="library-meta"><span>${esc(SOURCE_LABELS[n.source] || n.source)}</span><time>${esc(listTime(n.updated_at))}</time>${status}</span>
      </button>
      <button class="library-delete" type="button" data-delete="${n.id}" aria-label="Delete ${esc(n.title)}" title="Delete">×</button>
    </div>`;
  }).join("");
}

$("libraryList").addEventListener("click", async e => {
  const open = e.target.closest("[data-open]");
  const del = e.target.closest("[data-delete]");
  if (open) openNote(Number(open.dataset.open));
  if (!del) return;
  const id = Number(del.dataset.delete);
  if (state.busy && id === state.noteId) { setState("This note is still being recorded", "error"); return; }
  const note = state.notes.find(n => n.id === id);
  if (!confirm(`Delete "${note ? note.title : "this note"}"? This cannot be undone.`)) return;
  try {
    await api(`/api/notes/${id}`, { method: "DELETE" });
    if (id === state.noteId) { resetResults(); setState("Note deleted", "idle"); }
    refreshNotes();
  } catch (err) {
    setState("Delete failed: " + err.message);
  }
});

async function openNote(id, { scroll = true } = {}) {
  if (state.busy) { setState("Finish the current session before opening another note", "error"); return; }
  try {
    const note = await api(`/api/notes/${id}`);
    state.noteId = note.id;
    state.checks = note.checks || {};
    setNoteTitle(note.title);
    updateTranscript(note.transcript);
    updateSummary(note.summary);
    $("summaryUpdated").textContent = "Saved " + formatDate(note.updated_at);
    if (note.status === "processing") {
      state.watchNoteId = note.id;
      setState("Processing on the server · you can close this page", "busy");
      if (!note.summary) $("summary").innerHTML = '<div class="empty-mark">⏳</div><h3>Still processing on the server.</h3><p>This note fills in automatically when it is ready. Closing the page does not stop it.</p>';
    } else if (note.status === "live") {
      setState("This session is still recording or finishing on the server", "busy");
    } else if (note.status === "failed") {
      setState("Processing failed", "error");
    } else {
      setState("Opened saved note", "idle");
    }
    if (note.error) $("summary").insertAdjacentHTML("beforeend", `<p class="summary-warning">${esc(note.error)}</p>`);
    renderLibrary();
    setSidebar(false);
    if (scroll) $("noteView").scrollTop = 0;
  } catch (e) {
    setState("Could not open note: " + e.message);
  }
}

$("importMd").onclick = () => $("importFile").click();
$("importFile").onchange = async () => {
  const file = $("importFile").files[0];
  $("importFile").value = "";
  if (!file) return;
  try {
    const { id } = await api("/api/notes/import", {
      method: "POST", json: { markdown: await file.text(), title: file.name.replace(/\.(md|markdown|txt)$/i, "") },
    });
    await refreshNotes();
    openNote(id);
  } catch (e) {
    setState("Import failed: " + e.message);
  }
};

$("renameNote").onclick = async () => {
  if (!state.noteId) return;
  const title = prompt("Rename note", state.noteTitle);
  if (!title || !title.trim()) return;
  try {
    await api(`/api/notes/${state.noteId}`, { method: "PATCH", json: { title } });
    setNoteTitle(title.trim());
    refreshNotes();
  } catch (e) {
    setState("Rename failed: " + e.message);
  }
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
  $("uploadStatus").textContent = "Uploading… keep this page open until the upload finishes.";
  resetResults();
  setState("Uploading recording", "busy");
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("language", $("uploadLanguage").value);
  fd.append("include_summary", "true");
  try {
    const r = await fetch("/transcribe", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({ detail: `Server returned ${r.status}` }));
    if (r.status === 401) showAuth();
    if (!r.ok) throw Error(data.detail || JSON.stringify(data));
    // The server now works on it in the background and saves it to the account.
    state.noteId = data.note_id;
    state.watchNoteId = data.note_id;
    $("uploadStatus").textContent = data.message || "Processing on the server. You can close this page.";
    setBusy(false);
    await refreshNotes();
    openNote(data.note_id, { scroll: false });
  } catch (e) {
    setState("Upload failed");
    $("uploadStatus").textContent = e.message;
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
  markLiveTab(mode);
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
    if (m.type === "session") { state.noteId = m.note_id; refreshNotes(); }
    else if (m.type === "transcript") { updateTranscript(state.transcript ? state.transcript + "\n" + m.text : m.text); setState("Listening · transcript updated", "live"); }
    else if (m.type === "summary") { updateSummary(m.summary); refreshNotes(); setState(state.stopping ? "Finishing notes" : "Listening · notes updated", state.stopping ? "busy" : "live"); }
    else if (m.type === "status") setState(m.message);
    else if (m.type === "error") setState("Issue: " + m.message);
    else if (m.type === "finished") {
      updateTranscript(m.transcript || state.transcript);
      setState("Session finished · notes saved", "idle");
      state.ws.close();
      refreshNotes();
    }
  };
  state.ws.onerror = () => setState("Live connection failed. Check server and network.");
  state.ws.onclose = e => {
    if (state.activeMode && !state.stopping) {
      // 4429: the server refused a second live session and already said why.
      if (e.code !== 4429) setState("Connection closed by the server");
      onUnexpectedClose();
    }
    if (e.code === 4401) showAuth();
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
  markLiveTab(null);
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

// ---------------------------------------------------------------- start

(async () => {
  try {
    const config = await fetch("/api/config").then(r => r.json());
    if (!config.allow_registration) $("authSwitchRow").hidden = true;
  } catch {}
  setAuthMode("login");
  try { showApp(await api("/api/me")); } catch { showAuth(); }
})();
