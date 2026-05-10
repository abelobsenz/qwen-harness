// qwen-ui — front-end. Vanilla, no build step.
// SSE chat client (fetch-based, since EventSource is GET-only).

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ===================== rich rendering pipeline =====================
// markdown · LaTeX (KaTeX) · code highlight · mermaid · image rewriting
// Loaded from CDN in index.html before this module. Each lib is checked
// at use-time so the UI degrades to plain text if any library failed.

const KATEX_DELIMS = [
  { left: "$$", right: "$$", display: true  },
  { left: "\\[", right: "\\]", display: true  },
  { left: "\\(", right: "\\)", display: false },
  { left: "$",  right: "$",  display: false },
];

// Configure marked once. Custom image renderer rewrites any path that
// isn't already a URL through /api/file so local png/jpg/svg can render.
//
// CRITICAL: register the KaTeX extension FIRST so math is extracted from
// the source during tokenization. Otherwise marked turns `$x_1 + y_2$`
// into `$x<em>1 + y</em>2$` (underscores become italics) and KaTeX has
// nothing left to parse.
(function configureMarked() {
  if (typeof marked === "undefined") return;
  marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });

  // KaTeX integration via marked-katex-extension. Recognises:
  //   $...$       inline math
  //   $$...$$     display math
  //   \(...\)     inline math (LaTeX-native)
  //   \[...\]     display math (LaTeX-native)
  // and produces fully-rendered HTML right inside the markdown tokens,
  // so neither markdown formatting nor DOMPurify's text walker can mangle it.
  if (typeof markedKatex === "function") {
    try {
      marked.use(markedKatex({
        throwOnError: false,
        nonStandard: true,   // accept `\(...\)` and `\[...\]` too
        output: "htmlAndMathml",
      }));
    } catch (e) { console.warn("marked-katex-extension setup failed:", e); }
  }

  const renderer = new marked.Renderer();
  const _origImage = renderer.image.bind(renderer);
  renderer.image = function (href, title, text) {
    if (href && !/^(https?:|data:|\/api\/file\?|blob:)/i.test(href)) {
      let p = href;
      if (p.startsWith("file://")) p = p.slice(7);
      href = "/api/file?path=" + encodeURIComponent(p);
    }
    return _origImage(href, title, text);
  };
  // Add target=_blank to external links so they don't replace our SPA.
  const _origLink = renderer.link.bind(renderer);
  renderer.link = function (href, title, text) {
    let html = _origLink(href, title, text);
    if (href && /^https?:\/\//i.test(href)) {
      html = html.replace(/^<a /, '<a target="_blank" rel="noopener noreferrer" ');
    }
    return html;
  };
  marked.use({ renderer });
})();

// Mermaid is opt-in (we call mermaid.run() manually after replacing
// `<pre><code class="language-mermaid">` blocks).
(function configureMermaid() {
  if (typeof mermaid === "undefined") return;
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  try {
    mermaid.initialize({
      startOnLoad: false,
      theme: dark ? "dark" : "default",
      securityLevel: "strict",
      fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
    });
  } catch (e) { console.warn("mermaid init failed", e); }
})();

// Theme management — explicit override of `prefers-color-scheme` via a
// localStorage flag. Defaults to "auto" (follow system); user can flip to
// "light" or "dark" with the topbar toggle.
const THEME_STORAGE_KEY = "qwen-ui-theme";
function currentTheme() {
  return localStorage.getItem(THEME_STORAGE_KEY) || "auto";
}
function effectiveTheme() {
  const t = currentTheme();
  if (t !== "auto") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function applyTheme() {
  const t = effectiveTheme();
  document.documentElement.dataset.theme = t;
  // Swap the highlight.js stylesheet href to match.
  const themeLink = document.getElementById("hljs-theme");
  if (themeLink) {
    const wantDark = t === "dark";
    const isDark = themeLink.href.includes("github-dark");
    if (wantDark && !isDark) {
      themeLink.href = themeLink.href.replace("/github.min.css", "/github-dark.min.css");
    } else if (!wantDark && isDark) {
      themeLink.href = themeLink.href.replace("/github-dark.min.css", "/github.min.css");
    }
  }
  // Update the toggle icon to hint the next state.
  const icon = document.getElementById("theme-toggle-icon");
  if (icon) {
    const next = currentTheme();
    icon.textContent = next === "dark" ? "☀" : next === "light" ? "🌙" : "🌗";
  }
}
applyTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
  if (currentTheme() === "auto") applyTheme();
});
function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const cur = currentTheme();
  const next = order[(order.indexOf(cur) + 1) % order.length];
  if (next === "auto") localStorage.removeItem(THEME_STORAGE_KEY);
  else localStorage.setItem(THEME_STORAGE_KEY, next);
  // Brief cross-fade — toggle a transition class for the duration of the
  // color swap, then drop it so we don't pay the transition cost forever.
  document.documentElement.classList.add("theme-transition");
  applyTheme();
  setTimeout(() => {
    document.documentElement.classList.remove("theme-transition");
  }, 220);
  toast(`theme: ${next}`, "info");
}
document.getElementById("theme-toggle")?.addEventListener("click", cycleTheme);

function renderMarkdownSafe(text) {
  if (!text) return "";
  if (typeof marked === "undefined") return escapeHtml(text);
  let html;
  try { html = marked.parse(text); }
  catch { return escapeHtml(text); }
  if (typeof DOMPurify !== "undefined") {
    // mathMl: keep KaTeX's <math>…</math> output intact
    // svg / svgFilters: keep mermaid's rendered <svg>
    html = DOMPurify.sanitize(html, {
      ADD_ATTR: ["target", "rel"],
      USE_PROFILES: { html: true, mathMl: true, svg: true, svgFilters: true },
    });
  }
  return html;
}

// Run after a node's innerHTML is set. Highlights code, rewrites images,
// attaches copy buttons, replaces mermaid blocks. (KaTeX is already done
// by the marked-katex-extension during markdown tokenization.)
async function enrichRenderedBlock(el) {
  if (!el) return;
  // Belt-and-suspenders KaTeX pass for any tool result or replayed text
  // that didn't go through the marked extension. Skips inside `<code>`,
  // `<pre>`, and the already-rendered `.katex` spans so it never
  // double-renders.
  if (typeof renderMathInElement === "function") {
    try {
      renderMathInElement(el, {
        delimiters: KATEX_DELIMS,
        throwOnError: false,
        errorColor: "var(--bad)",
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        ignoredClasses: ["katex", "katex-html", "katex-mathml"],
      });
    } catch (e) { console.warn("katex run failed", e); }
  }
  // Mermaid diagrams: marked emits `<pre><code class="language-mermaid">…`.
  // Convert each to a `<div class="mermaid">` and let mermaid render.
  const mermBlocks = el.querySelectorAll("pre code.language-mermaid");
  if (mermBlocks.length && typeof mermaid !== "undefined") {
    for (const code of mermBlocks) {
      const src = code.textContent;
      const wrap = elt("div", "mermaid");
      wrap.textContent = src;
      const pre = code.closest("pre");
      if (pre) pre.replaceWith(wrap); else code.replaceWith(wrap);
    }
    try { await mermaid.run({ querySelector: ".mermaid:not([data-processed])" }); }
    catch (e) { console.warn("mermaid render failed", e); }
  }
  // Code highlight + copy button on any remaining <pre><code>.
  if (typeof hljs !== "undefined") {
    el.querySelectorAll("pre code:not([data-hl])").forEach((c) => {
      try { hljs.highlightElement(c); } catch {}
      c.dataset.hl = "1";
    });
  }
  el.querySelectorAll("pre:not([data-copy])").forEach((pre) => {
    pre.dataset.copy = "1";
    pre.appendChild(makeCopyButton(() =>
      pre.querySelector("code")?.textContent ?? pre.textContent ?? ""
    ));
  });
  // Images: lazy-load + lightbox + show a placeholder on broken refs.
  el.querySelectorAll("img:not([data-lb])").forEach((img) => {
    img.dataset.lb = "1";
    img.loading = "lazy";
    img.classList.add("md-image");
    img.addEventListener("click", () => openLightbox(img.src, img.alt));
    img.addEventListener("error", () => img.classList.add("md-image-broken"));
  });
}

function makeCopyButton(getText) {
  const btn = elt("button", "code-copy");
  btn.type = "button";
  btn.textContent = "copy";
  btn.title = "copy to clipboard";
  btn.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    try {
      await navigator.clipboard.writeText(getText() || "");
      btn.textContent = "copied";
      btn.classList.add("ok");
    } catch {
      btn.textContent = "failed";
    }
    setTimeout(() => {
      btn.textContent = "copy";
      btn.classList.remove("ok");
    }, 1300);
  });
  return btn;
}

// Live (per-delta) markdown rendering, throttled to one paint per frame so
// long answers don't pin the main thread. KaTeX/highlight only run once
// at turn-end (they're expensive and partial input often won't parse).
function scheduleLiveRender(node) {
  if (!node || node._raf) return;
  node._raf = requestAnimationFrame(() => {
    node._raf = null;
    try {
      node.innerHTML = renderMarkdownSafe(node._buf || "");
    } catch {
      node.textContent = node._buf || "";
    }
  });
}

// Final pass — guaranteed fully-parsed markdown + KaTeX + highlight.
function finalizeRender(node) {
  if (!node) return;
  if (node._raf) { cancelAnimationFrame(node._raf); node._raf = null; }
  try {
    node.innerHTML = renderMarkdownSafe(node._buf ?? node.textContent ?? "");
  } catch {
    /* keep whatever live-render produced */
  }
  node._mdRendered = true;
  enrichRenderedBlock(node);
}

// Re-render every assistant text-run in place. Used after replayMessages
// once the session is loaded into the DOM.
window.__enrichAllAssistantTurns = function () {
  for (const node of document.querySelectorAll(".turn.assistant .text-run")) {
    if (!node._mdRendered) finalizeRender(node);
  }
};

// ----- lightbox -----

function openLightbox(src, alt) {
  const lb = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  if (!lb || !img || !src) return;
  img.src = src;
  img.alt = alt || "";
  lb.hidden = false;
}
function closeLightbox() {
  const lb = document.getElementById("lightbox");
  if (lb) lb.hidden = true;
}
document.addEventListener("click", (ev) => {
  const lb = document.getElementById("lightbox");
  if (!lb || lb.hidden) return;
  if (ev.target === lb || ev.target.classList?.contains("lightbox-close")) {
    closeLightbox();
  }
});

// Detect when a tool emitted a generated image artifact (matplotlib,
// `screencapture`, `convert`, etc) so we can inline it next to the tool
// widget. Heuristic: a single absolute path on its own line ending in
// a known image extension.
function detectGeneratedImagePath(result) {
  if (!result) return null;
  const lines = String(result).split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const m = line.match(/^["']?([\/~][^"'<>\s]+\.(?:png|jpe?g|gif|webp|svg|bmp))["']?$/i);
    if (m) return m[1];
    // also accept `Saved figure to /tmp/foo.png` style
    const m2 = line.match(/(?:saved (?:figure|image|file|plot)? ?(?:to|at|as)?\s*[:=]?\s*)([\/~][^"'<>\s]+\.(?:png|jpe?g|gif|webp|svg|bmp))/i);
    if (m2) return m2[1];
  }
  return null;
}

// ----------------------------- tab routing -----------------------------

function showTab(name) {
  if (name !== "chat" && name !== "agents") name = "chat";
  document.body.dataset.tab = name;
  for (const btn of $$(".tab")) {
    btn.setAttribute("aria-selected", String(btn.dataset.tab === name));
  }
  $("#view-chat").hidden   = name !== "chat";
  $("#view-agents").hidden = name !== "agents";
  if (location.hash !== `#${name}`) {
    history.replaceState({}, "", `#${name}`);
  }
}
for (const btn of $$(".tab")) {
  btn.addEventListener("click", () => showTab(btn.dataset.tab));
}
window.addEventListener("hashchange", () => showTab(location.hash.replace("#", "")));
showTab(location.hash.replace("#", "") || "chat");

// ----------------------------- toast helper ----------------------------

const _recentToasts = new Map();      // msg → expiry ts (ms)
function toast(msg, kind = "error") {
  // Dedupe: ignore the same message if we showed it within the last 3s.
  const now = Date.now();
  for (const [k, t] of _recentToasts) {
    if (t < now) _recentToasts.delete(k);
  }
  if (_recentToasts.has(msg)) return;
  _recentToasts.set(msg, now + 3000);

  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  if (kind === "warn") el.style.borderLeftColor = "var(--warn)";
  if (kind === "info") el.style.borderLeftColor = "var(--accent)";
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}
window.toast = toast;

// Map fetch errors to a friendlier short message that hints at the cause.
function fetchErrorMessage(err) {
  if (!err) return "unknown error";
  // TypeError "Failed to fetch" = network/connection error
  if (err.name === "TypeError" || /failed to fetch|networkerror/i.test(err.message)) {
    return "server unreachable — is qwen-ui still running?";
  }
  if (err.name === "AbortError") return "cancelled";
  return err.message || String(err);
}

// ----------------------------- health probe ----------------------------

const healthDot   = $("#health-dot");
const modelLabel  = $("#model-label");
const queueLabel  = $("#queue-label");

// Persistent "UI server is down" banner — shown when /api/health itself
// can't be reached (different from upstream-down, which still means the
// UI server is alive and just lost contact with inference).
function setUiServerDownBanner(down, reason = "") {
  let bar = document.getElementById("server-down-banner");
  if (down) {
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "server-down-banner";
      bar.className = "server-down-banner";
      bar.innerHTML = `
        <span class="banner-icon" aria-hidden="true">⚠</span>
        <span class="banner-text"><strong>UI server unreachable.</strong>
          The browser can't reach the qwen-ui process. Run
          <code>qwen-ui</code> in a terminal (or <code>qwen-ui restart</code>)
          to bring it back.</span>
        <span class="banner-stat" id="server-down-reason"></span>`;
      document.body.appendChild(bar);
    }
    const sub = document.getElementById("server-down-reason");
    if (sub) sub.textContent = reason || "";
  } else if (bar) {
    bar.remove();
  }
}

async function probeHealth() {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    setUiServerDownBanner(false);
    if (d.context_tokens) {
      CONTEXT_LIMIT_TOKENS = d.context_tokens;
      updateContextBar();
    }
    if (d.upstream === "up") {
      healthDot.dataset.state = "up";
      healthDot.title = `upstream up • ${d.url}`;
      const id = d.model || d.model_alias || "?";
      modelLabel.textContent = id.split("/").pop();
      modelLabel.title = id;
    } else {
      healthDot.dataset.state = "down";
      healthDot.title = `upstream down: ${d.error || "?"}`;
      modelLabel.textContent = "offline";
    }
    if (d.queue) {
      const total = (d.queue.high || 0) + (d.queue.normal || 0);
      const inflight = d.queue.busy ? 1 : 0;
      const waiting = total + inflight;
      if (waiting > 0) {
        queueLabel.hidden = false;
        queueLabel.textContent = `queue: ${waiting}`;
        queueLabel.dataset.state = waiting >= 4 ? "full" : "busy";
      } else {
        queueLabel.hidden = true;
      }
    }
  } catch (e) {
    healthDot.dataset.state = "down";
    healthDot.title = `health probe failed: ${e}`;
    modelLabel.textContent = "offline";
    // Distinguish UI-server-unreachable from upstream-down with a persistent banner.
    setUiServerDownBanner(true, e?.message || String(e));
  }
}
probeHealth();
setInterval(probeHealth, 5_000);

// ============================ CHAT TAB =================================

const transcript  = $("#transcript");
const composer    = $("#composer");
const input       = $("#composer-input");
const sendBtn     = $("#send-btn");

let currentSession = { id: null, messages: [] };
let activeTurn = null;          // see startAssistantTurn for shape
let abortController = null;
let userScrolled = false;       // pause auto-scroll if user scrolled up

// Pending attachments — files the user has dropped/picked but not yet sent.
// Each entry: { localId, file, status, server }   where `server` = the
// /api/upload response once uploaded.
let pendingAttachments = [];

// Cached context-window limit from /api/health; set by probeHealth.
let CONTEXT_LIMIT_TOKENS = 60000;
// Real token count reported by the upstream's stream_options.include_usage.
// Takes precedence over the local char-based estimate, which is systemati-
// cally inaccurate for tool-heavy turns (web_fetch returns 100k+ chars and
// dflash's tokenizer compresses them differently than English prose).
let lastReportedUsage = null;        // { prompt_tokens, completion_tokens, total_tokens }
// Tokens accrued by the in-flight stream — added to lastReportedUsage so the
// bar moves during a long generation instead of jumping at end-of-turn.
let inflightTokens = 0;

transcript.addEventListener("scroll", () => {
  userScrolled = (transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight) > 80;
});

function autogrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 240) + "px";
  // Send button stays enabled in two cases: there's text to send, OR a
  // stream is in flight (in which case it's the Stop button).
  sendBtn.disabled = !input.value.trim() && !abortController;
  updateLiveTokenHint();
}

// Live token-hint chip in the composer footer. Reads the typed text +
// pending attachments and shows an estimated token count so the user
// knows when they're approaching the chat-tab compaction threshold.
function updateLiveTokenHint() {
  const chip = document.getElementById("composer-token-hint");
  if (!chip) return;
  const text = input?.value || "";
  let pending = Math.ceil(text.length / 3.5);
  for (const a of (pendingAttachments || [])) {
    const chars = Number(a?.server?.text_chars) || 0;
    if (a?.status === "ready" && chars > 0) pending += Math.ceil(chars / 3.5);
  }
  if (pending === 0) {
    chip.hidden = true;
    return;
  }
  chip.hidden = false;
  chip.textContent = `~${fmtTokens(pending)} tokens`;
  // Color states: hint warm/danger only relative to the chat compact
  // threshold (60k) — that's the soft ceiling, NOT the model's 256k cap.
  const limit = (typeof CONTEXT_LIMIT_TOKENS !== "undefined" && CONTEXT_LIMIT_TOKENS) || 60000;
  const ratio = pending / limit;
  chip.dataset.state = ratio >= 0.5 ? "danger" : ratio >= 0.25 ? "warn" : "ok";
}
function setSendButtonStreaming(streaming) {
  if (streaming) {
    sendBtn.classList.add("stop");
    sendBtn.textContent = "■ Stop";
    sendBtn.title = "abort generation (Esc)";
    sendBtn.disabled = false;
    startThinkTimer();
  } else {
    sendBtn.classList.remove("stop");
    sendBtn.textContent = "Send";
    sendBtn.title = "send message (⌘↵)";
    sendBtn.disabled = !input.value.trim();
    stopThinkTimer();
  }
}

// "Thinking time" timer — counts elapsed seconds from when the user
// sends a prompt until the model finishes (or the user aborts). While
// active it ticks every second; after stop it lingers as "Xs" so the
// user can see what the previous turn cost.
let thinkTimerStart = 0;
let thinkTimerInterval = null;
const thinkTimerEl = document.getElementById("think-timer");

function _formatThinkTime(secs) {
  if (secs < 10)   return secs.toFixed(1) + "s";
  if (secs < 60)   return Math.round(secs) + "s";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}
function startThinkTimer() {
  if (!thinkTimerEl) return;
  thinkTimerStart = performance.now();
  thinkTimerEl.dataset.state = "active";
  thinkTimerEl.hidden = false;
  thinkTimerEl.textContent = "0.0s";
  if (thinkTimerInterval) clearInterval(thinkTimerInterval);
  thinkTimerInterval = setInterval(() => {
    const secs = (performance.now() - thinkTimerStart) / 1000;
    thinkTimerEl.textContent = _formatThinkTime(secs);
  }, 1000);
}
function stopThinkTimer() {
  if (!thinkTimerEl) return;
  if (thinkTimerInterval) {
    clearInterval(thinkTimerInterval);
    thinkTimerInterval = null;
  }
  // Final tally shown statically so the user can read it after the turn.
  if (thinkTimerStart > 0) {
    const secs = (performance.now() - thinkTimerStart) / 1000;
    thinkTimerEl.textContent = _formatThinkTime(secs);
    thinkTimerEl.dataset.state = "final";
    // Keep visible — it lingers until the next turn starts.
  }
  thinkTimerStart = 0;
}

// "Compacting..." indicator — visible while an auto-compact upstream
// call is in flight. The spinner + pulsing pill differentiates it from
// the regular think-timer (compaction can take 1-3 minutes).
const compactIndicatorEl = document.getElementById("compact-indicator");
function setCompacting(active) {
  if (!compactIndicatorEl) return;
  compactIndicatorEl.hidden = !active;
}
input.addEventListener("input", autogrow);
input.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    composer.requestSubmit();
  } else if (e.key === "Escape" && abortController) {
    e.preventDefault();
    abortController.abort();
  }
});

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  // Send button doubles as Stop while a stream is in flight.
  if (abortController) {
    abortController.abort();
    return;
  }
  const text = input.value.trim();
  // Allow send when there are attachments even if the text is empty —
  // the model can still react to "look at this PDF" without a question.
  if (!text && pendingAttachments.length === 0) return;
  await sendMessage(text);
});

// ============================ FILE ATTACHMENTS ============================
// Drag-and-drop and the paperclip button both feed a single `pendingAttachments`
// list. Files are uploaded to /api/upload immediately on selection so the user
// sees per-file progress; on send we only need to pass the upload IDs along
// with the chat-stream POST and the backend stitches the text into the user
// turn before forwarding to the model.

const attachInput     = $("#attach-input");
const attachBtn       = $("#attach-btn");
const attachmentsEl   = $("#attachments");
const droppableEl     = $("#composer-droppable");

attachBtn?.addEventListener("click", () => attachInput?.click());
attachInput?.addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  if (files.length) addPendingFiles(files);
  attachInput.value = "";
});

// Drag-and-drop (whole transcript area is a target so dropping anywhere works).
let dragDepth = 0;
function setDropping(on) {
  if (!droppableEl) return;
  droppableEl.hidden = !on;
  composer?.classList.toggle("dragging", on);
}
["dragenter", "dragover"].forEach(ev =>
  document.addEventListener(ev, (e) => {
    if (!Array.from(e.dataTransfer?.types || []).includes("Files")) return;
    e.preventDefault();
    if (ev === "dragenter") dragDepth++;
    setDropping(dragDepth > 0);
  })
);
document.addEventListener("dragleave", (e) => {
  if (!Array.from(e.dataTransfer?.types || []).includes("Files")) return;
  dragDepth = Math.max(0, dragDepth - 1);
  setDropping(dragDepth > 0);
});
document.addEventListener("drop", (e) => {
  if (!Array.from(e.dataTransfer?.types || []).includes("Files")) return;
  e.preventDefault();
  dragDepth = 0;
  setDropping(false);
  const files = Array.from(e.dataTransfer?.files || []);
  if (files.length) addPendingFiles(files);
});

function addPendingFiles(files) {
  for (const f of files) {
    const localId = "att-" + Math.random().toString(36).slice(2, 9);
    const entry = { localId, file: f, status: "uploading", server: null };
    pendingAttachments.push(entry);
    uploadOneAttachment(entry);
  }
  renderAttachmentChips();
  updateContextBar();
}

async function uploadOneAttachment(entry) {
  const fd = new FormData();
  fd.append("file", entry.file, entry.file.name);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !Array.isArray(d.files) || !d.files[0] || d.files[0].status >= 400) {
      throw new Error(d.files?.[0]?.error || d.error || `HTTP ${r.status}`);
    }
    entry.server = d.files[0];
    entry.status = "ready";
  } catch (err) {
    entry.status = "error";
    entry.error = fetchErrorMessage(err);
    toast(`upload failed: ${entry.file.name}: ${entry.error}`);
  }
  renderAttachmentChips();
  updateContextBar();
}

function renderAttachmentChips() {
  if (!attachmentsEl) return;
  if (pendingAttachments.length === 0) {
    attachmentsEl.hidden = true;
    attachmentsEl.innerHTML = "";
    autogrow();
    return;
  }
  attachmentsEl.hidden = false;
  attachmentsEl.innerHTML = "";
  for (const e of pendingAttachments) {
    const chip = document.createElement("span");
    chip.className = "att-chip att-" + e.status;
    const icon = document.createElement("span");
    icon.className = "att-icon";
    icon.textContent = e.status === "uploading" ? "⏳" : e.status === "error" ? "✕" : "📎";
    const name = document.createElement("span");
    name.className = "att-name";
    name.textContent = e.file.name;
    const meta = document.createElement("span");
    meta.className = "att-meta";
    if (e.status === "ready" && e.server) {
      const kb = (e.server.size / 1024).toFixed(1);
      const chars = e.server.text_chars
        ? `${(e.server.text_chars / 1000).toFixed(1)}k chars`
        : "binary";
      meta.textContent = `${kb} KB · ${chars}` + (e.server.truncated ? " (truncated)" : "");
    } else if (e.status === "uploading") {
      meta.textContent = "uploading…";
    } else if (e.status === "error") {
      meta.textContent = e.error || "failed";
    }
    const x = document.createElement("button");
    x.type = "button";
    x.className = "att-remove";
    x.textContent = "×";
    x.title = "remove attachment";
    x.addEventListener("click", () => {
      pendingAttachments = pendingAttachments.filter(a => a.localId !== e.localId);
      renderAttachmentChips();
      updateContextBar();
    });
    chip.append(icon, name, meta, x);
    attachmentsEl.appendChild(chip);
  }
  autogrow();
}

$("#new-chat-btn")?.addEventListener("click", () => {
  if (abortController) {
    toast("a turn is in progress — wait or press Esc to abort", "warn");
    return;
  }
  // Before clearing, distill the just-finished session into durable memory.
  // Fire-and-forget — the user shouldn't wait for it.
  extractSessionMemory(currentSession);
  newSession();
});

$("#remember-btn")?.addEventListener("click", () => {
  if (abortController) {
    toast("a turn is in progress — wait or press Esc to abort", "warn");
    return;
  }
  extractSessionMemory(currentSession, /*manual=*/true);
});

// ---------------------------- export-as-markdown ----------------------
// Walks currentSession.messages and serializes to a clean Markdown file
// with role headers, tool-call blocks, and code-fenced tool results so
// the user can paste a chat into a doc / share it / archive it offline.
$("#export-btn")?.addEventListener("click", () => {
  const lines = exportSessionAsMarkdown(currentSession);
  if (!lines) {
    toast("nothing to export — say hello first", "warn");
    return;
  }
  const blob = new Blob([lines], { type: "text/markdown;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 10);
  const safe = (currentSession.id || "chat").replace(/[^A-Za-z0-9_-]+/g, "_");
  a.download = `qwen-chat-${stamp}-${safe.slice(0, 16)}.md`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 800);
  toast("downloaded", "info");
});

function exportSessionAsMarkdown(session) {
  if (!session || !Array.isArray(session.messages) || session.messages.length === 0) {
    return null;
  }
  const out = [];
  const title = (session.messages.find(m => m.role === "user")?.content || "")
                 .split("\n")[0].slice(0, 80) || "qwen chat";
  out.push(`# ${title}`);
  if (session.id) out.push(`_session: ${session.id}_  ·  exported ${new Date().toISOString()}\n`);
  for (const m of session.messages) {
    if (m.role === "user") {
      out.push(`\n## You\n\n${m.content || "(empty)"}`);
    } else if (m.role === "assistant") {
      out.push(`\n## Qwen`);
      if (m.content) out.push(`\n${m.content}`);
      const calls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (const tc of calls) {
        const fn = tc.function || {};
        const argstr = typeof fn.arguments === "string"
                       ? fn.arguments : JSON.stringify(fn.arguments || {});
        out.push(`\n_called \`${fn.name || "?"}\` →_\n\n\`\`\`json\n${argstr}\n\`\`\``);
      }
    } else if (m.role === "tool") {
      out.push(`\n_tool result:_\n\n\`\`\`\n${(m.content || "").slice(0, 8000)}\n\`\`\``);
    }
  }
  return out.join("\n") + "\n";
}

// ---------------------------- quick-start prompts ----------------------
// Welcome screen has six buttons; clicking one fills the composer with
// the canned prompt so the user can edit before sending.
document.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".quick-start");
  if (!btn) return;
  const prompt = btn.dataset.prompt || "";
  if (!prompt) return;
  input.value = prompt;
  input.focus();
  // Move caret to end so the user can immediately edit / extend.
  input.setSelectionRange(prompt.length, prompt.length);
  autogrow();
});

// Distill the session into durable memory facts. Skipped silently for
// short sessions (< 4 messages) unless `manual` is true (in which case
// we surface the skip reason). The endpoint runs the model with thinking
// disabled — typical latency is 2–4s.
async function extractSessionMemory(session, manual = false) {
  if (!session || !session.id) {
    if (manual) toast("no session to extract from", "warn");
    return;
  }
  if (!manual && (session.messages?.length || 0) < 4) {
    return;  // too short, silent skip on auto-trigger
  }
  if (manual) toast("extracting durable facts… (a few seconds)", "info");
  try {
    const r = await fetch(
      `/api/sessions/${encodeURIComponent(session.id)}/extract`,
      { method: "POST" }
    );
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    const n = d.saved || 0;
    if (n > 0) {
      toast(`saved ${n} fact${n === 1 ? "" : "s"} to memory`, "info");
    } else if (manual) {
      const why = d.skipped || d.error || "no durable facts found";
      toast(`memory: ${why}`, "warn");
    }
  } catch (err) {
    if (manual) toast(`extract failed: ${fetchErrorMessage(err)}`);
  }
}

function newSession() {
  currentSession = { id: null, messages: [] };
  // Clear the server-reported usage — it belonged to the previous session.
  lastReportedUsage = null;
  inflightTokens = 0;
  transcript.innerHTML = `
    <div class="welcome">
      <h2>Start a conversation</h2>
      <p>Ask anything. The model has tools and can browse, edit files, and run code.</p>
    </div>`;
  highlightActiveSession();
  updateContextBar();
}

// ----------------------------- session sidebar -------------------------

const sessionList = $("#session-list");
const sessionSearchInput = $("#session-search");

// Cache the most recent /api/sessions response so the search box can
// filter without re-hitting the network each keystroke.
let _allSessions = [];
let _sessionFilter = "";

async function refreshSessionList() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) return;
    _allSessions = await r.json();
    renderSessionList();
  } catch (e) {
    // silent; sidebar is best-effort
  }
}

function _matchesSession(s, q) {
  if (!q) return true;
  q = q.toLowerCase();
  return ((s.title || "").toLowerCase().includes(q) ||
          (s.id || "").toLowerCase().includes(q));
}

function renderSessionList() {
  const items = _allSessions.filter(s => _matchesSession(s, _sessionFilter));
  if (!items.length) {
    sessionList.innerHTML = `<li class="empty">${
      _sessionFilter ? "no matches" : "no past sessions"
    }</li>`;
    return;
  }
  sessionList.innerHTML = "";
  for (const s of items) {
    const li = elt("li", "session-item");
    li.dataset.id = s.id;
    li.title = `${s.title}\n${s.n_messages} messages • ${s.updated_at || ""}`;
    const label = elt("span", "session-title");
    label.textContent = s.title;
    const meta = elt("span", "session-meta");
    meta.textContent = `${s.n_messages}`;
    meta.title = `${s.n_messages} messages`;
    const del = elt("button", "session-del icon-btn");
    del.type = "button";
    del.textContent = "×";
    del.title = "delete session";
    del.setAttribute("aria-label", `delete session: ${s.title}`);
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      deleteSession(s.id, s.title);
    });
    li.append(label, meta, del);
    li.addEventListener("click", () => loadSession(s.id));
    // Hover preview: lazy-fetch the session, render the first user msg
    // and last assistant response in a floating tooltip card.
    li.addEventListener("mouseenter", () => sessionPreviewOnHover(s, li));
    li.addEventListener("mouseleave", hideSessionPreview);
    sessionList.appendChild(li);
  }
  highlightActiveSession();
}

let _sessionPreviewTimer = null;
let _sessionPreviewEl = null;

function sessionPreviewOnHover(summary, anchorEl) {
  if (_sessionPreviewTimer) clearTimeout(_sessionPreviewTimer);
  _sessionPreviewTimer = setTimeout(async () => {
    let data;
    try {
      const r = await fetch(`/api/sessions/${encodeURIComponent(summary.id)}`);
      if (!r.ok) return;
      data = await r.json();
    } catch { return; }
    showSessionPreview(data, anchorEl);
  }, 350);  // small dwell-delay so flicking past doesn't fire requests
}

function hideSessionPreview() {
  if (_sessionPreviewTimer) {
    clearTimeout(_sessionPreviewTimer);
    _sessionPreviewTimer = null;
  }
  if (_sessionPreviewEl) {
    _sessionPreviewEl.remove();
    _sessionPreviewEl = null;
  }
}

function showSessionPreview(data, anchorEl) {
  if (!data || !anchorEl) return;
  hideSessionPreview();
  const messages = data.messages || [];
  const firstUser = messages.find(m => m.role === "user")?.content || "";
  const lastAssistant = [...messages].reverse().find(
    m => m.role === "assistant" && m.content
  )?.content || "";
  const card = document.createElement("div");
  card.className = "session-preview";
  const trim = (s, n) => {
    s = String(s).replace(/\s+/g, " ").trim();
    return s.length > n ? s.slice(0, n) + "…" : s;
  };
  card.innerHTML = `
    <div class="sp-title">${escapeHtml(trim(data.title || "(untitled)", 90))}</div>
    <div class="sp-block sp-user"><span class="sp-label">you</span>${escapeHtml(trim(firstUser, 220) || "(no message)")}</div>
    <div class="sp-block sp-asst"><span class="sp-label">qwen</span>${escapeHtml(trim(lastAssistant, 320) || "(no answer yet)")}</div>
    <div class="sp-foot"><span>${messages.length} message${messages.length === 1 ? "" : "s"}</span><span>${escapeHtml(data.updated_at || "")}</span></div>`;
  // Position to the right of the anchor; clip to viewport.
  const r = anchorEl.getBoundingClientRect();
  card.style.left = `${Math.min(r.right + 8, window.innerWidth - 380)}px`;
  card.style.top = `${Math.max(8, Math.min(r.top, window.innerHeight - 220))}px`;
  document.body.appendChild(card);
  _sessionPreviewEl = card;
}

sessionSearchInput?.addEventListener("input", (e) => {
  _sessionFilter = e.target.value.trim();
  renderSessionList();
});

async function deleteSession(id, title) {
  if (abortController) {
    toast("a turn is in progress — wait or press Esc to abort", "warn");
    return;
  }
  if (!confirm(`Delete session "${title || id}"?`)) return;
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    if (currentSession.id === id) newSession();
    await refreshSessionList();
  } catch (err) {
    toast(`delete failed: ${err.message}`);
  }
}

function highlightActiveSession() {
  for (const li of sessionList.children) {
    li.classList.toggle("active", li.dataset && li.dataset.id === currentSession.id);
  }
}

async function loadSession(id) {
  if (abortController) {
    toast("a turn is in progress — wait or press Esc to abort", "warn");
    return;
  }
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    currentSession = { id, messages: data.messages || [] };
    // Loaded a different session — drop the previous turn's usage stats so
    // the context bar shows an estimate (refines on the first new turn).
    lastReportedUsage = null;
    inflightTokens = 0;
    transcript.innerHTML = "";
    replayMessages(currentSession.messages);
    highlightActiveSession();
    updateContextBar();
  } catch (e) {
    toast(`could not load session: ${fetchErrorMessage(e)}`);
  }
}

function replayMessages(messages) {
  for (const m of messages) {
    if (m.role === "user") {
      appendUserTurn(m.content || "");
    } else if (m.role === "assistant") {
      // Reconstruct an assistant turn from saved state. The live append
      // helpers all use insertIntoBubble (which falls back to appendChild
      // when the cursor is gone), so it's safe to leave the cursor in
      // place during replay and only finalize at the end of the turn.
      startAssistantTurn();
      if (m.content) {
        appendMessageDelta(m.content);
        if (activeTurn?.textNode) activeTurn.textNode._closed = true;
      }
      const calls = m.tool_calls || [];
      for (const tc of calls) {
        let args = {};
        try { args = JSON.parse(tc.function?.arguments || "{}"); } catch {}
        appendToolCall({ id: tc.id, name: tc.function?.name, args });
      }
      // Finalize: drop cursor + dim any tool widgets still flagged "running"
      finalizeAllActive();
    } else if (m.role === "tool") {
      // Find the matching tool widget in the DOM and fill in the result.
      const turns = transcript.querySelectorAll(".turn.assistant");
      const lastTurn = turns[turns.length - 1];
      if (!lastTurn) continue;
      const dets = lastTurn.querySelectorAll("details.tool");
      for (const det of dets) {
        const out = det.querySelector(".tool-outcome");
        if (out && (out.classList.contains("run") || out.textContent === "cancelled")) {
          out.classList.remove("run", "err");
          out.classList.add("ok");
          out.textContent = "saved";
          const body = det.querySelector(".body");
          if (body) body.textContent = m.content || "";
          break;
        }
      }
    }
  }
  // Re-render markdown for replayed assistant text once the libs are loaded.
  if (window.__enrichAllAssistantTurns) window.__enrichAllAssistantTurns();
  scrollBottom();
}

refreshSessionList();

async function sendMessage(text) {
  // Drop welcome if it's there.
  const welcome = transcript.querySelector(".welcome");
  if (welcome) welcome.remove();

  // Wait for any in-flight uploads to finish before we send.
  const stillUploading = pendingAttachments.filter(a => a.status === "uploading");
  if (stillUploading.length) {
    toast(`waiting for ${stillUploading.length} upload(s)…`, "info");
    while (pendingAttachments.some(a => a.status === "uploading")) {
      await new Promise(r => setTimeout(r, 120));
    }
  }
  // Snapshot ready attachments and clear the staging area; failed ones drop.
  const readyAttachments = pendingAttachments.filter(a => a.status === "ready");
  const attachmentsForRequest = readyAttachments.map(a => ({
    id: a.server.id, filename: a.server.filename,
  }));
  // Build a UI-only display of the message: typed text + paperclip badges.
  const userBubbleParts = [];
  for (const a of readyAttachments) {
    const kb = (a.server.size / 1024).toFixed(1);
    userBubbleParts.push(`📎 **${a.server.filename}** · ${kb} KB`);
  }
  if (text) userBubbleParts.push(text);
  const displayText = userBubbleParts.join("\n\n");

  appendUserTurn(displayText || "(no text)");
  currentSession.messages.push({ role: "user", content: text || "" });

  input.value = "";
  pendingAttachments = [];
  renderAttachmentChips();
  autogrow();

  abortController = new AbortController();
  setSendButtonStreaming(true);

  streamChat(currentSession.messages, abortController.signal, attachmentsForRequest)
    .catch(err => {
      if (err.name !== "AbortError") {
        toast(`stream error: ${err.message}`);
      } else {
        appendSystemNote("aborted by user");
      }
    })
    .finally(() => {
      abortController = null;
      setSendButtonStreaming(false);
      finalizeAllActive();
    });
}

// ----------------------------- DOM helpers -----------------------------

function elt(tag, cls) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  return el;
}

function scrollBottom() { transcript.scrollTop = transcript.scrollHeight; }
function scrollBottomMaybe() { if (!userScrolled) scrollBottom(); }

function appendUserTurn(text) {
  const turn = elt("div", "turn user");
  const role = elt("div", "role"); role.textContent = "you";
  const bubble = elt("div", "bubble md");
  // Render as markdown so attached-file <details> blocks collapse and
  // user-typed code fences look nice. Falls back to plain text if any
  // lib is missing (renderMarkdownSafe handles that).
  bubble.innerHTML = renderMarkdownSafe(text || "");
  enrichRenderedBlock(bubble);
  turn.append(role, bubble);
  transcript.appendChild(turn);
  scrollBottom();
}

function appendSystemNote(text) {
  const turn = elt("div", "turn assistant");
  const role = elt("div", "role"); role.textContent = "system";
  const bubble = elt("div", "bubble"); bubble.style.opacity = ".6";
  bubble.textContent = `(${text})`;
  turn.append(role, bubble);
  transcript.appendChild(turn);
  scrollBottomMaybe();
}

// ----------------------------- assistant rendering ---------------------

function startAssistantTurn() {
  const turn = elt("div", "turn assistant");
  const role = elt("div", "role"); role.textContent = "qwen";
  const bubble = elt("div", "bubble");
  const cursor = elt("span", "cursor");
  bubble.appendChild(cursor);
  // Per-turn actions row (copy + regenerate). Hidden until hover via CSS.
  const actions = elt("div", "turn-actions");
  const copyBtn = elt("button", "turn-action");
  copyBtn.type = "button"; copyBtn.title = "copy message";
  copyBtn.textContent = "⧉ copy";
  copyBtn.addEventListener("click", async () => {
    const text = (activeTurn?.bubble || bubble)?.innerText || "";
    try {
      await navigator.clipboard.writeText(text.trim());
      copyBtn.textContent = "✓ copied";
      setTimeout(() => copyBtn.textContent = "⧉ copy", 1300);
    } catch { copyBtn.textContent = "× failed"; }
  });
  const regenBtn = elt("button", "turn-action");
  regenBtn.type = "button"; regenBtn.title = "re-run from the previous user message";
  regenBtn.textContent = "↻ regenerate";
  regenBtn.addEventListener("click", () => regenerateFromTurn(turn));
  actions.append(copyBtn, regenBtn);

  turn.append(role, bubble, actions);
  transcript.appendChild(turn);
  activeTurn = {
    turn,
    bubble,
    cursor,
    thinkingDetail: null,
    thinkingBody: null,
    thinkingSummary: null,
    textNode: null,    // current contiguous text run; closed by tool widget
    toolsById: new Map(),
  };
  scrollBottom();
  return activeTurn;
}

function regenerateFromTurn(turnEl) {
  if (abortController) {
    toast("a turn is in progress — wait or press Esc to abort", "warn");
    return;
  }
  // Walk currentSession.messages backwards, find the user message that
  // immediately preceded this assistant turn, truncate everything after,
  // and re-stream.
  const msgs = currentSession.messages;
  // The simplest contract: drop the trailing assistant + any tool messages
  // until we hit the most recent user message, keep that, and resend.
  let cut = msgs.length - 1;
  while (cut >= 0 && msgs[cut].role !== "user") cut--;
  if (cut < 0) {
    toast("no prior user message to regenerate from", "warn");
    return;
  }
  currentSession.messages = msgs.slice(0, cut + 1);
  // Remove DOM nodes for the truncated turns.
  let n = turnEl;
  while (n) {
    const next = n.nextSibling;
    n.remove();
    n = next;
  }
  // Stream a fresh response.
  abortController = new AbortController();
  setSendButtonStreaming(true);
  streamChat(currentSession.messages, abortController.signal)
    .catch(err => {
      if (err.name !== "AbortError") toast(`stream error: ${err.message}`);
      else appendSystemNote("aborted by user");
    })
    .finally(() => {
      abortController = null;
      setSendButtonStreaming(false);
      finalizeAllActive();
    });
}

function ensureActiveTurn() {
  if (!activeTurn) startAssistantTurn();
  return activeTurn;
}

// Insert a node before the active turn's cursor, falling back to appendChild
// when the cursor is no longer in the DOM (e.g. after replayMessages
// removes it for a finalized turn). This is what fixes the
// "node before which the new node is to be inserted is not a child" crash
// that hit anyone clicking an old session that included tool calls.
function insertIntoBubble(t, node) {
  if (t.cursor && t.cursor.parentNode === t.bubble) {
    t.bubble.insertBefore(node, t.cursor);
  } else {
    t.bubble.appendChild(node);
  }
}

function appendThinkingDelta(text) {
  if (!text) return;
  const t = ensureActiveTurn();
  if (!t.thinkingDetail) {
    const det = elt("details", "thinking");
    det.open = false;       // collapsed by default
    const sum = elt("summary"); sum.textContent = "thinking…";
    const body = elt("div", "body");
    det.append(sum, body);
    insertIntoBubble(t, det);
    t.thinkingDetail = det;
    t.thinkingBody = body;
    t.thinkingSummary = sum;
  }
  t.thinkingBody.append(text);
  scrollBottomMaybe();
}

function finalizeThinking(turn) {
  if (turn?.thinkingSummary) {
    const len = turn.thinkingBody.textContent.length;
    turn.thinkingSummary.textContent = `reasoning (${len.toLocaleString()} chars)`;
  }
}

function appendMessageDelta(text) {
  if (!text) return;
  const t = ensureActiveTurn();
  // Once we start emitting visible message text, finalize the thinking
  // header (so the user sees the rich label, not "thinking…" forever).
  if (t.thinkingDetail && !t._thinkingFinal) {
    finalizeThinking(t);
    t._thinkingFinal = true;
  }
  if (!t.textNode || t.textNode._closed) {
    // div, not span — markdown emits block elements (lists, headings, …).
    t.textNode = elt("div", "text-run md");
    t.textNode._buf = "";
    insertIntoBubble(t, t.textNode);
  }
  t.textNode._buf = (t.textNode._buf || "") + text;
  // Loop-guard surfacing: when the proxy emits its abort marker, swap
  // the message bubble's class so CSS can show a styled warning callout.
  // The proxy mirror in scripts/loop_guard_marker.py uses the same
  // detection contract: substring `[loop-guard:` AND one of the proxy's
  // specific suffix phrases. Without the suffix check, the model
  // legitimately mentioning the marker (e.g. answering "how does the
  // loop guard work?") would surface a false-positive warning bubble.
  // Round 15-17 fixed this in agent.py / agent_graph.py — same fix
  // here for parity.
  if (!t._loopGuardSurfaced && isProxyAbortMarker(t.textNode._buf)) {
    t._loopGuardSurfaced = true;
    if (t.bubble && t.bubble.classList) {
      t.bubble.classList.add("has-loop-guard-abort");
    }
    if (typeof toast === "function") {
      toast("Output stopped by loop guard — the model fell into a "
        + "repetition loop. Try rephrasing.", "warn");
    }
  }
  scheduleLiveRender(t.textNode);
  scrollBottomMaybe();
}

// High-precision detector for the proxy's loop-guard abort marker.
// Mirror of scripts/loop_guard_marker.is_proxy_abort_marker — kept in
// sync deliberately. Requires BOTH:
//   - the literal substring `[loop-guard:`
//   - one of the proxy's specific abort suffix phrases:
//       "output stopped early"      (non-streaming format)
//       "fell into a repetition loop" (streaming format)
// This avoids false positives on benign mentions (model explanations,
// echoed log/grep output, etc.) — see test_loop_guard_marker.py for
// the exhaustive case list this is paired against.
function isProxyAbortMarker(text) {
  if (!text || text.indexOf("[loop-guard:") === -1) return false;
  return /output stopped early|fell into a repetition loop/i.test(text);
}
// Expose for any future inline tests / consumers.
window.__isProxyAbortMarker = isProxyAbortMarker;

// Tool name → glyph. Used in the chat bubble so the user can scan a
// long thread and see at a glance what the model did. Anything not
// in this map gets a generic "🛠" — keeps the surface uniform.
const TOOL_ICONS = {
  // file ops
  read_file: "📖", list_files: "📂", grep: "🔎",
  write_file: "✏️", edit_file: "✏️", apply_patch: "📝",
  append_finding: "✚", write_file_verified: "✅",
  // shell / code
  bash: "▸", python_run: "🐍", python_reset: "♻", test_run: "🧪",
  notebook_edit: "📓", notebook_run: "📓",
  // web / research
  web_search: "🌐", web_fetch: "🌐", web_outline: "🗺",
  arxiv_search: "📚", arxiv_fetch: "📄",
  pdf_extract: "📄", github_repo: "🐙",
  doi_resolve: "🔖", csv_summary: "📊", now: "🕓",
  // memory
  memory_save: "🧠", memory_search: "🧠", memory_get: "🧠",
  memory_list: "🧠", memory_delete: "🧠",
  // agent / control
  explore: "🧭", subagent_implement: "🤖",
  enter_worktree: "🌿", exit_worktree: "🌿",
  todo_write: "☑", done: "🏁",
};

function toolIcon(name) {
  return TOOL_ICONS[name] || "🛠";
}

// Cheap heuristic: does this string look enough like markdown that
// running it through marked would improve the display? Avoids running
// markdown on raw output (file lists, JSON blobs, etc).
function _looksLikeMarkdown(s) {
  if (!s || typeof s !== "string" || s.length < 20) return false;
  // markdown signals: heading, bold emphasis, bullet list, table.
  if (/(^|\n)#{1,4}\s+\S/.test(s)) return true;
  if (/\*\*[^*\s][^*]{0,80}\*\*/.test(s)) return true;
  if (/(^|\n)[*\-+]\s+\S/.test(s) && /\n\s*[*\-+]\s+\S/.test(s)) return true;
  if (/(^|\n)\|.+\|\n\|[\s\-:|]+\|/.test(s)) return true;
  return false;
}

function appendToolCall({ id, name, args }) {
  const t = ensureActiveTurn();
  if (t.textNode) t.textNode._closed = true;

  const det = elt("details", "tool");
  det.dataset.tool = name || "?";
  det.open = false;

  const sum = elt("summary");
  const iconEl = elt("span", "tool-icon");
  iconEl.textContent = toolIcon(name);
  iconEl.setAttribute("aria-hidden", "true");
  const nameEl = elt("span", "tool-name"); nameEl.textContent = name;
  const argEl  = elt("span", "tool-arg");  argEl.textContent  = formatToolArg(name, args);
  argEl.title = JSON.stringify(args || {}, null, 2);  // hover for full args
  const outEl  = elt("span", "tool-outcome run");
  // Pulse spinner instead of plain "running" text — far less visual noise.
  outEl.innerHTML = '<span class="tool-spinner" aria-hidden="true"></span><span class="tool-status-label">running</span>';
  sum.append(iconEl, nameEl, argEl, outEl);

  const body = elt("div", "body");
  body.textContent = "";

  det.append(sum, body);
  insertIntoBubble(t, det);
  t.toolsById.set(id, { det, sum, body, outEl, argEl, name });
  scrollBottomMaybe();
}

function appendToolResult({ id, name, result, ok, cached, error, truncated, duration_ms }) {
  const t = ensureActiveTurn();
  let entry = t.toolsById.get(id);
  if (!entry) {
    // tool_call event was missed — synthesize a header for orphan results
    appendToolCall({ id, name, args: {} });
    entry = t.toolsById.get(id);
  }
  const { det, body, outEl } = entry;
  outEl.classList.remove("run");
  outEl.classList.add(ok ? "ok" : "err");
  const lines = (result || "").split("\n").length;
  let label;
  if (!ok) label = `error${error ? ": " + error.slice(0, 60) : ""}`;
  else if (cached) label = `cached • ${duration_ms}ms`;
  else label = `${lines} line${lines === 1 ? "" : "s"} • ${duration_ms}ms${truncated ? " • truncated" : ""}`;
  outEl.textContent = label;       // replaces the spinner+label markup
  // Brief flash class so the eye can catch which tool just finished
  // even when the user is scrolled near the bottom of a long thread.
  det.classList.add("just-finished");
  setTimeout(() => det.classList.remove("just-finished"), 750);

  body.textContent = result || "";

  // Diff highlighting for patches / edits / structured-write tool results.
  if ((name === "apply_patch" || name === "edit_file" || name === "write_file_verified")
      && /(^|\n)(\+\+\+|\-\-\-|@@)/.test(result || "")) {
    body.classList.add("diff");
    body.innerHTML = highlightDiff(result);
  }
  // For tool results that look like markdown (the new arxiv_fetch /
  // doi_resolve / github_repo / web_outline tools all return markdown),
  // render properly so headings + lists look right inside the tool body.
  // Heuristic: starts with `# `, contains `**bold**`, or has a `## section`.
  if (ok && !body.classList.contains("diff") && _looksLikeMarkdown(result)) {
    try {
      body.innerHTML = renderMarkdownSafe(result);
      body.classList.add("md");
      enrichRenderedBlock(body);
    } catch { /* keep textContent */ }
  }

  // If the tool produced (or named) an image file, inline-render it next
  // to the tool widget so the user doesn't have to hunt for the path.
  const imgPath = ok ? detectGeneratedImagePath(result) : null;
  if (imgPath) {
    const wrap = elt("div", "tool-image-wrap");
    const img = elt("img", "tool-image md-image");
    img.alt = imgPath;
    img.loading = "lazy";
    img.src = "/api/file?path=" + encodeURIComponent(imgPath);
    img.dataset.lb = "1";
    img.addEventListener("click", () => openLightbox(img.src, imgPath));
    img.addEventListener("error", () => img.classList.add("md-image-broken"));
    wrap.appendChild(img);
    const cap = elt("div", "tool-image-caption");
    cap.textContent = imgPath;
    wrap.appendChild(cap);
    det.insertBefore(wrap, body);
  }

  // Decide whether to show expanded:
  //   - errors: always
  //   - results <= 3 lines and short: yes
  //   - everything else: keep collapsed
  const expand = !ok || (lines <= 3 && (result || "").length <= 240);
  det.open = expand;
  scrollBottomMaybe();
}

function finalizeAllActive() {
  if (!activeTurn) return;
  if (activeTurn.cursor) activeTurn.cursor.remove();
  finalizeThinking(activeTurn);
  // Make sure the buffered markdown becomes its final rendered form even
  // if the stream was aborted before turn_end fired.
  activeTurn.bubble.querySelectorAll(".text-run").forEach(finalizeRender);
  // Mark any still-running tools (from an aborted turn) as cancelled
  for (const [, entry] of activeTurn.toolsById) {
    if (entry.outEl.classList.contains("run")) {
      entry.outEl.classList.remove("run");
      entry.outEl.classList.add("err");
      entry.outEl.textContent = "cancelled";
    }
  }
  activeTurn = null;
}

function endTurn(hadTools) {
  if (activeTurn?.cursor) activeTurn.cursor.remove();
  finalizeThinking(activeTurn);
  // Last chance to render markdown / KaTeX / highlight code.
  if (activeTurn) {
    activeTurn.bubble.querySelectorAll(".text-run").forEach(finalizeRender);
  }
  if (hadTools) {
    // Another assistant cycle is about to start — open a new turn.
    activeTurn = null;
    startAssistantTurn();
  } else {
    activeTurn = null;
  }
}

// ----------------------------- arg / diff helpers ----------------------

function formatToolArg(name, args) {
  if (!args || typeof args !== "object") return "";
  const key = args.path || args.file_path || args.command || args.query
              || args.url || args.pattern || args.task || args.code;
  if (key) return String(key).split("\n")[0].slice(0, 80);
  return JSON.stringify(args).slice(0, 80);
}

function highlightDiff(text) {
  // Track running line numbers from the hunk header so the gutter can
  // show file:line for every line. `(-)` rows tick the old-file counter,
  // `(+)` rows tick the new-file counter, context lines tick both.
  let oldNo = 0, newNo = 0;
  const escape = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const out = [];
  for (const raw of String(text).split("\n")) {
    let cls = "", oldN = "", newN = "";
    if (raw.startsWith("+++") || raw.startsWith("---")) {
      cls = "diff-file";
    } else if (raw.startsWith("@@")) {
      const m = raw.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (m) { oldNo = parseInt(m[1], 10); newNo = parseInt(m[2], 10); }
      cls = "diff-hunk";
    } else if (raw.startsWith("+") && !raw.startsWith("+++")) {
      cls = "diff-add"; newN = String(newNo); newNo++;
    } else if (raw.startsWith("-") && !raw.startsWith("---")) {
      cls = "diff-del"; oldN = String(oldNo); oldNo++;
    } else if (raw.length === 0 || raw.startsWith(" ")) {
      cls = "diff-ctx"; oldN = String(oldNo); newN = String(newNo);
      if (oldNo) oldNo++;
      if (newNo) newNo++;
    }
    out.push(
      `<span class="diff-line ${cls}">` +
        `<span class="diff-num diff-num-old">${oldN}</span>` +
        `<span class="diff-num diff-num-new">${newN}</span>` +
        `<span class="diff-text">${escape(raw)}</span>` +
      `</span>`);
  }
  return out.join("\n");
}

// ----------------------------- SSE client ------------------------------

async function streamChat(messages, signal, attachments = []) {
  startAssistantTurn();

  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      session_id: currentSession.id,
      attachments,
    }),
    signal,
  });

  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${txt.slice(0, 200)}`);
  }
  if (!resp.body) {
    throw new Error("no response body");
  }

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });

    // Parse SSE frames separated by blank lines (\n\n).
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleFrame(frame);
    }
  }
}

function handleFrame(frame) {
  let event = "message";
  let dataLine = null;
  for (const raw of frame.split("\n")) {
    if (raw.startsWith("event:")) event = raw.slice(6).trim();
    else if (raw.startsWith("data:")) dataLine = raw.slice(5).trim();
  }
  if (dataLine === null) return;
  let data;
  try { data = JSON.parse(dataLine); } catch { return; }

  switch (event) {
    case "started":
      currentSession.id = data.session_id;
      // New turn — reset the in-flight token counter; the next `usage`
      // event will replace it with an authoritative number from upstream.
      inflightTokens = 0;
      updateContextBar();
      break;
    case "thinking":
      appendThinkingDelta(data.delta || "");
      // Reasoning tokens count against the same context budget.
      inflightTokens += estimateTokens(data.delta || "");
      updateContextBar();
      break;
    case "message":
      appendMessageDelta(data.delta || "");
      inflightTokens += estimateTokens(data.delta || "");
      updateContextBar();
      break;
    case "tool_call":
      appendToolCall(data);
      // Tool args are usually short JSON, but they DO count.
      inflightTokens += 30 + estimateTokens(
        typeof data.args === "string" ? data.args : JSON.stringify(data.args || {})
      );
      updateContextBar();
      break;
    case "tool_result":
      appendToolResult(data);
      // Tool results are the BIGGEST source of estimator drift — web_fetch
      // can dump 100k+ chars in a single result. Count them as soon as we
      // see them so the bar reflects reality during a multi-tool turn.
      inflightTokens += estimateTokens(data.result || "");
      updateContextBar();
      break;
    case "usage":
      // Authoritative count from upstream's tokenizer. Replace estimate.
      lastReportedUsage = data;
      inflightTokens = 0;          // server already counted what's stored
      updateContextBar();
      break;
    case "turn_end":
      endTurn(!!data.had_tools);
      break;
    case "done":
      currentSession.messages = data.messages || currentSession.messages;
      // Refresh sidebar so the new/updated session appears immediately.
      refreshSessionList();
      updateContextBar();
      break;
    case "compaction":
      // Auto-compaction lifecycle: started → done|failed. Toggle the
      // "Compacting…" badge so the user knows why the response paused.
      // The think-timer keeps ticking through compaction since it's
      // wall-clock time the user is waiting.
      if (data.status === "started") {
        setCompacting(true);
      } else {
        setCompacting(false);
        if (data.status === "done" && typeof data.tokens_after === "number") {
          // After a successful compaction the new authoritative count is
          // tokens_after — feed it into the context bar so the bar snaps
          // to the new (smaller) value rather than waiting for the next
          // postflight usage event.
          lastReportedUsage = {
            prompt_tokens: data.tokens_after,
            completion_tokens: 0,
            total_tokens: data.tokens_after,
            source: "compaction",
          };
          inflightTokens = 0;
          updateContextBar();
        }
      }
      break;
    case "error":
      toast(data.message || "error");
      break;
    default:
      // unknown event — ignore
      break;
  }
}

// ============================ AGENTS TAB ==============================
//
// Backend contract:
//   GET    /api/agents                              list, with live status
//   POST   /api/agents                              create
//   GET    /api/agents/<id>                         details + recent runs
//   PATCH  /api/agents/<id>                         partial update
//   DELETE /api/agents/<id>                         delete (also cancels in-flight)
//   POST   /api/agents/<id>/run-now                 schedule immediate run
//   POST   /api/agents/<id>/cancel-run              SIGTERM in-flight (if running)
//   GET    /api/agents/<id>/runs                    list runs (most recent first)
//   GET    /api/agents/<id>/runs/<run_id>           single run (meta + events)
//   GET    /api/supervisor/status                   supervisor health/queue
//
// State held in memory:
//   agentList   — last-fetched array of summaries
//   selectedId  — agent currently shown in the detail panel (or null)
//   pollTimer   — setTimeout handle for the active polling cycle
// ----------------------------------------------------------------------

const agentModal = $("#agent-modal");
const agentListEl = $("#agent-list");
const agentDetailEl = $("#agent-detail");

let agentList = [];
let selectedId = null;
let selectedRunId = null;       // run currently expanded in the detail panel
let editingAgentId = null;      // when set, the modal is in "edit" mode
let pollTimer = null;

$("#new-agent-btn")?.addEventListener("click", () => openAgentModal());
$("#welcome-new-agent")?.addEventListener("click", () => openAgentModal());
$$("[data-close]", agentModal).forEach(b =>
  b.addEventListener("click", () => {
    editingAgentId = null;
    agentModal.close();
  })
);
// Also reset editing state when the dialog is closed via ESC.
agentModal.addEventListener("close", () => { editingAgentId = null; });

function openAgentModal(prefill = null) {
  const form = $("#agent-form");
  form.reset();
  editingAgentId = prefill?.id || null;
  $("#agent-modal-title").textContent = editingAgentId ? "Edit agent" : "New agent";
  // Toggle the submit-button label so users know which mode they're in.
  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.textContent = editingAgentId ? "Save" : "Create";

  if (prefill) {
    form.querySelector("[name=name]").value             = prefill.name || "";
    form.querySelector("[name=prompt]").value           = prefill.prompt || "";
    form.querySelector("[name=cwd]").value              = prefill.cwd || "~";
    form.querySelector("[name=interval_seconds]").value = prefill.interval_seconds ?? 600;
    form.querySelector("[name=enabled]").checked        = prefill.enabled !== false;
  } else {
    // Default cwd is the user's home — most agents will want this.
    const cwdInput = form.querySelector("[name=cwd]");
    if (cwdInput && !cwdInput.value) cwdInput.value = "~";
  }
  agentModal.showModal();
}

$("#agent-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const cwdRaw = (fd.get("cwd") || "").toString().trim();
  // Expand a leading ~ on the client — the backend also expands but the
  // server-side check needs an absolute path that exists.
  const cwd = cwdRaw === "~" ? "/Users/" + (await getUsernameOnce())
            : cwdRaw.startsWith("~/")
              ? "/Users/" + (await getUsernameOnce()) + cwdRaw.slice(1)
              : cwdRaw;
  const body = {
    name: (fd.get("name") || "").toString().trim(),
    prompt: (fd.get("prompt") || "").toString().trim(),
    cwd,
    interval_seconds: Number(fd.get("interval_seconds") || 600),
    enabled: form.querySelector("[name=enabled]")?.checked ?? true,
  };
  const isEdit = !!editingAgentId;
  try {
    const r = await fetch(
      isEdit ? `/api/agents/${editingAgentId}` : "/api/agents",
      {
        method: isEdit ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    agentModal.close();
    selectedId = d.id || editingAgentId;
    editingAgentId = null;
    await refreshAgents();
    await loadAgentDetail(selectedId);
    toast(isEdit ? "agent updated" : "agent created", "info");
  } catch (err) {
    toast(`${isEdit ? "update" : "create"} failed: ${err.message}`);
  }
});

let _username = null;
async function getUsernameOnce() {
  if (_username) return _username;
  // The UI server doesn't expose user info directly, but the home path
  // is in /api/health. Parse the username out of /Users/<name>.
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const m = (d.home || "").match(/^\/Users\/([^/]+)/);
    if (m) _username = m[1];
  } catch {}
  return _username || "me";
}

// ----- list rendering -----

function fmtRelative(ts) {
  if (!ts) return "—";
  const sec = Math.max(0, (Date.now() / 1000) - ts);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function fmtNext(ts) {
  if (!ts) return "—";
  const sec = ts - Date.now() / 1000;
  if (sec < 0) return "now";
  if (sec < 60) return `in ${Math.round(sec)}s`;
  if (sec < 3600) return `in ${Math.round(sec / 60)}m`;
  return `in ${Math.round(sec / 3600)}h`;
}

function statusDot(status) {
  // returns a class suffix for CSS coloring
  return ({
    running: "running",
    queued:  "queued",
    paused:  "paused",
    idle:    "idle",
  })[status] || "idle";
}

function renderAgentList() {
  if (!agentListEl) return;
  agentListEl.innerHTML = "";
  if (!agentList.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "no agents yet";
    agentListEl.appendChild(li);
    return;
  }
  for (const a of agentList) {
    const li = document.createElement("li");
    li.className = "agent-item" + (a.id === selectedId ? " selected" : "");
    li.dataset.agentId = a.id;
    const statusCls = statusDot(a.status);
    const lastRun = a.last_run_at ? fmtRelative(a.last_run_at) : "never";
    const lastStatus = a.last_run_status
      ? ` (${a.last_run_status})` : "";
    li.innerHTML = `
      <span class="status-pill ${statusCls}" title="${a.status}"></span>
      <div class="agent-meta">
        <div class="agent-name">${escapeHtml(a.name)}</div>
        <div class="agent-sub">${escapeHtml(a.status)} · last ${lastRun}${lastStatus}</div>
      </div>`;
    li.addEventListener("click", () => {
      if (selectedId !== a.id) selectedRunId = null;
      selectedId = a.id;
      renderAgentList();
      loadAgentDetail(a.id);
    });
    agentListEl.appendChild(li);
  }
}

async function refreshAgents() {
  try {
    const r = await fetch("/api/agents");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    agentList = await r.json();
    renderAgentList();
    updateAgentsHeaderStats();
  } catch (err) {
    console.warn("agents fetch failed", err);
  }
}

// ============== context window status bar ==============
// Token accounting strategy:
//   1) When we have an authoritative usage report from upstream (the
//      `usage` SSE event, populated by stream_options.include_usage), use
//      it directly — that's the dflash tokenizer's exact count, no
//      estimation needed.
//   2) During an in-flight stream, add `inflightTokens` (what we've seen
//      from delta/tool_call/tool_result events since `started`).
//   3) When there's no upstream count yet (fresh session, before the first
//      turn), fall back to a char-based estimate of the conversation.
//
// The character-to-token ratio depends heavily on content type:
//   English prose:   ~3.8 chars/tok    Code/JSON:  ~3.0    Math/LaTeX: ~2.5
// We pick 3.2 as a weighted average that errs on the slightly-conservative
// side (the bar tells you when you're getting close to the wall — better
// to nudge slightly high than to surprise you with sudden compaction).
const CHARS_PER_TOKEN = 3.2;

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / CHARS_PER_TOKEN);
}

function estimateConversationTokens() {
  let total = 0;
  // System prompt is ~700 tokens after the math/format rules were added.
  total += 700;
  for (const m of (currentSession?.messages || [])) {
    if (m.content) total += estimateTokens(m.content);
    // Tool calls — each one adds the JSON args + ~30 tokens of envelope
    // (function name, role markers, tool_call_id wrapping).
    if (Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const args = tc?.function?.arguments;
        total += 30 + estimateTokens(typeof args === "string" ? args : JSON.stringify(args || ""));
      }
    }
    // Tool result messages already contribute via m.content above; nothing
    // extra to add here.
  }
  // Pending (un-sent) attachments — their extracted text will be inlined
  // into the user message on the NEXT send, so reflect that future cost
  // in the bar today.
  for (const a of (pendingAttachments || [])) {
    const chars = Number(a?.server?.text_chars) || 0;
    if (a?.status === "ready" && chars > 0) {
      total += Math.ceil(chars / CHARS_PER_TOKEN);
    }
  }
  return total;
}

// The single source of truth for the bar. Returns the best number we have:
// upstream's authoritative usage when available, otherwise the local
// estimate. Adds in-flight tokens accumulated during an active stream.
function effectiveTokensUsed() {
  if (lastReportedUsage && Number(lastReportedUsage.prompt_tokens) >= 0) {
    // prompt_tokens is what's about to be re-sent next turn; that's the
    // total context cost of the conversation as the model sees it.
    return Math.max(0, Number(lastReportedUsage.prompt_tokens) || 0)
         + Math.max(0, Number(lastReportedUsage.completion_tokens) || 0)
         + Math.max(0, inflightTokens);
  }
  return estimateConversationTokens() + Math.max(0, inflightTokens);
}

const contextBarEl   = $("#context-bar");
const contextFillEl  = $("#context-bar-fill");
const contextLabelEl = $("#context-bar-label");

function fmtTokens(n) {
  if (n >= 100_000) return (n / 1000).toFixed(0) + "k";
  if (n >= 10_000)  return (n / 1000).toFixed(1) + "k";
  if (n >= 1_000)   return (n / 1000).toFixed(2) + "k";
  return String(n);
}

function updateContextBar() {
  // Keep the live composer-foot chip in sync whenever the persistent
  // context bar updates — same input data, same cadence.
  try { updateLiveTokenHint(); } catch {}
  if (!contextBarEl || !contextFillEl || !contextLabelEl) return;
  const used  = effectiveTokensUsed();
  const limit = CONTEXT_LIMIT_TOKENS || 60000;
  const pct   = Math.min(100, (used / limit) * 100);
  contextBarEl.hidden = false;
  contextFillEl.style.width = pct.toFixed(1) + "%";
  contextLabelEl.textContent = `${fmtTokens(used)} / ${fmtTokens(limit)}`;
  // Color thresholds: green < 50%, yellow 50–80%, red > 80%.
  let state = "ok";
  if (pct >= 80) state = "danger";
  else if (pct >= 50) state = "warn";
  contextBarEl.dataset.state = state;
  // Tell the user whether the number is server-reported or estimated.
  const isExact = !!lastReportedUsage && inflightTokens === 0;
  if (isExact) {
    contextBarEl.title =
      `${used.toLocaleString()} of ${limit.toLocaleString()} tokens used `
      + `(${pct.toFixed(1)}%, server-reported, exact)`;
  } else if (lastReportedUsage) {
    contextBarEl.title =
      `~${used.toLocaleString()} of ${limit.toLocaleString()} tokens used `
      + `(server-anchored prompt=${lastReportedUsage.prompt_tokens} `
      + `completion=${lastReportedUsage.completion_tokens} +${inflightTokens} live)`;
  } else {
    contextBarEl.title =
      `~${used.toLocaleString()} of ${limit.toLocaleString()} tokens used `
      + `(${pct.toFixed(1)}%, local estimate, refines on first turn)`;
  }
}

function updateAgentsHeaderStats() {
  const total   = agentList.length;
  const running = agentList.filter(a => a.status === "running").length;
  const queued  = agentList.filter(a => a.status === "queued").length;
  const paused  = agentList.filter(a => a.status === "paused").length;
  const active  = total - paused;

  // Tab badge: hidden if no agents, else shows total. Pulses when one is running.
  const badge = $("#agents-tab-badge");
  if (badge) {
    if (total === 0) {
      badge.hidden = true;
    } else {
      badge.hidden = false;
      badge.textContent = String(total);
      badge.title = `${total} agent${total === 1 ? "" : "s"}` +
                     (running ? `, ${running} running` : "") +
                     (queued ? `, ${queued} queued` : "") +
                     (paused ? `, ${paused} paused` : "");
      badge.dataset.state = running ? "running" : "idle";
    }
  }

  // Banner stat: live status summary inside the agents view banner.
  const stat = $("#agents-banner-stat");
  if (stat) {
    if (total === 0) {
      stat.hidden = true;
    } else {
      stat.hidden = false;
      if (running) {
        stat.textContent = `${running} running · ${active}/${total} active`;
        stat.dataset.state = "active";
      } else if (queued) {
        stat.textContent = `${queued} queued · ${active}/${total} active`;
        stat.dataset.state = "active";
      } else {
        stat.textContent = `${active}/${total} active`;
        stat.dataset.state = "idle";
      }
    }
  }
}

// ----- detail rendering -----

function renderEmptyDetail() {
  agentDetailEl.classList.add("empty");
  agentDetailEl.innerHTML = `
    <div class="welcome">
      <h2>Continuous agents</h2>
      <p>Pick an agent on the left, or create a new one.</p>
      <button class="primary-btn" id="welcome-new-agent">+ New agent</button>
    </div>`;
  $("#welcome-new-agent")?.addEventListener("click", openAgentModal);
}

function renderAgentDetail(a) {
  agentDetailEl.classList.remove("empty");
  const isRunning = a.status === "running";
  const isPaused  = a.status === "paused";
  const intervalMin = Math.round((a.interval_seconds || 0) / 60);
  agentDetailEl.innerHTML = `
    <header class="agent-header">
      <div>
        <h2>${escapeHtml(a.name)}</h2>
        <div class="agent-status-line">
          <span class="status-pill ${statusDot(a.status)}"></span>
          <span>${escapeHtml(a.status)}</span>
          <span class="dim">·</span>
          <span>every ${intervalMin}m</span>
          <span class="dim">·</span>
          <span>cwd: <code>${escapeHtml(a.cwd || "")}</code></span>
        </div>
        <div class="agent-status-line">
          <span class="dim">last run:</span>
          <span>${fmtRelative(a.last_run_at)}${a.last_run_status ? ` (${a.last_run_status})` : ""}</span>
          <span class="dim">·</span>
          <span class="dim">next:</span>
          <span>${fmtNext(a.next_run_at)}</span>
        </div>
      </div>
      <div class="agent-actions">
        ${isRunning
          ? `<button class="agent-btn warn" data-act="cancel">Cancel run</button>`
          : `<button class="agent-btn" data-act="run-now">Run now</button>`}
        <button class="agent-btn" data-act="${isPaused ? "resume" : "pause"}">
          ${isPaused ? "Resume" : "Pause"}
        </button>
        <button class="agent-btn" data-act="edit">Edit</button>
        <button class="agent-btn danger" data-act="delete">Delete</button>
      </div>
    </header>
    <section class="agent-prompt">
      <h3>Prompt</h3>
      <pre>${escapeHtml(a.prompt || "")}</pre>
    </section>
    <section class="agent-runs">
      <h3>Recent runs</h3>
      <ul class="runs-list" id="runs-list-${a.id}">
        ${(a.recent_runs || []).length === 0
          ? '<li class="empty">no runs yet</li>'
          : a.recent_runs.map(r => `
              <li class="run-item" data-run-id="${r.run_id}">
                <span class="status-pill ${statusDot(r.status === 'ok' ? 'idle' : r.status === 'running' ? 'running' : 'paused')}"></span>
                <span class="run-id">${escapeHtml(r.run_id || "")}</span>
                <span class="dim">${escapeHtml(r.status || "")}</span>
                <span class="dim">${fmtRelative(r.started_at)}</span>
              </li>`).join("")}
      </ul>
      <div class="run-detail" id="run-detail" hidden></div>
    </section>`;

  // wire action buttons
  agentDetailEl.querySelector('[data-act=run-now]')?.addEventListener("click", () => agentAction(a.id, "run-now"));
  agentDetailEl.querySelector('[data-act=cancel]')?.addEventListener("click", () => agentAction(a.id, "cancel-run"));
  agentDetailEl.querySelector('[data-act=pause]')?.addEventListener("click", () => agentPatch(a.id, { enabled: false }));
  agentDetailEl.querySelector('[data-act=resume]')?.addEventListener("click", () => agentPatch(a.id, { enabled: true }));
  agentDetailEl.querySelector('[data-act=edit]')?.addEventListener("click", () => openAgentModal({
    id: a.id,
    name: a.name,
    prompt: a.prompt,
    cwd: a.cwd,
    interval_seconds: a.interval_seconds,
    enabled: a.enabled !== false,
  }));
  agentDetailEl.querySelector('[data-act=delete]')?.addEventListener("click", () => {
    if (!confirm(`Delete agent "${a.name}"? This also wipes its run history.`)) return;
    agentDelete(a.id);
  });

  // wire run rows
  for (const li of agentDetailEl.querySelectorAll(".run-item")) {
    li.addEventListener("click", () => {
      selectedRunId = li.dataset.runId;
      // Visually mark the row right away — the next poll tick will
      // re-apply this from selectedRunId.
      for (const sib of agentDetailEl.querySelectorAll(".run-item")) {
        sib.classList.toggle("selected", sib === li);
      }
      loadRunDetail(a.id, li.dataset.runId);
    });
  }
}

async function loadAgentDetail(aid) {
  if (!aid) {
    renderEmptyDetail();
    return;
  }
  try {
    const r = await fetch(`/api/agents/${aid}`);
    if (r.status === 404) {
      selectedId = null;
      selectedRunId = null;
      renderEmptyDetail();
      return;
    }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const a = await r.json();
    renderAgentDetail(a);
    // The poll loop calls us every 4s; re-renders blow away the run-detail
    // panel. If the user had a run open before the re-render, restore it.
    if (selectedRunId) {
      // Highlight the selected run in the list
      const runLi = agentDetailEl.querySelector(`.run-item[data-run-id="${CSS.escape(selectedRunId)}"]`);
      if (runLi) runLi.classList.add("selected");
      loadRunDetail(aid, selectedRunId);
    }
  } catch (err) {
    toast(`load failed: ${err.message}`);
  }
}

async function loadRunDetail(aid, rid) {
  const target = $("#run-detail");
  if (!target) return;
  target.hidden = false;
  target.innerHTML = `<div class="run-loading">loading ${rid}…</div>`;
  try {
    const r = await fetch(`/api/agents/${aid}/runs/${rid}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    const meta = d.meta || {};
    const events = d.events || [];
    target.innerHTML = `
      <div class="run-meta-line">
        <span><strong>${escapeHtml(rid)}</strong></span>
        <span class="dim">status: ${escapeHtml(meta.status || "?")}</span>
        <span class="dim">exit: ${meta.exit_code ?? "?"}</span>
        ${meta.started_at && meta.ended_at
          ? `<span class="dim">duration: ${(meta.ended_at - meta.started_at).toFixed(1)}s</span>`
          : ""}
      </div>
      <details open>
        <summary>${events.length} event${events.length === 1 ? "" : "s"}</summary>
        <ol class="run-events">
          ${events.map(ev => renderRunEvent(ev)).join("")}
        </ol>
      </details>`;
  } catch (err) {
    target.innerHTML = `<div class="error">load failed: ${escapeHtml(err.message)}</div>`;
  }
}

function fmtEventTs(ts) {
  // Events come from agent.py session logs (ts is an ISO string like
  // "2026-05-04T16:51:38") and from supervisor metas (ts is a Unix
  // float). Accept both, fall through to "" when unparseable.
  if (ts === null || ts === undefined || ts === "") return "";
  let d;
  if (typeof ts === "number") {
    d = new Date(ts * 1000);
  } else {
    d = new Date(String(ts));
  }
  if (isNaN(d.getTime())) return "";
  return d.toLocaleTimeString();
}

function renderRunEvent(ev) {
  const kind = ev.kind || ev.event || "?";
  const tsStr = fmtEventTs(ev.ts || ev.t);
  let body = "";
  if (kind === "assistant") {
    const calls = Array.isArray(ev.tool_calls) ? ev.tool_calls : [];
    if (calls.length) {
      body = calls.map(tc => {
        const fn = tc.function || {};
        return `<code class="tool-call">${escapeHtml(fn.name || "?")}(${escapeHtml(short(fn.arguments || "", 80))})</code>`;
      }).join(" ");
    }
    if (ev.content) {
      body += `<div class="ev-text">${escapeHtml(short(ev.content, 400))}</div>`;
    }
    if (ev.reasoning) {
      body += `<details class="ev-reasoning"><summary>thinking…</summary><pre>${escapeHtml(short(ev.reasoning, 1500))}</pre></details>`;
    }
    if (!body) body = `<span class="dim">(empty)</span>`;
  } else if (kind === "tool_result") {
    body = `<code>${escapeHtml(ev.name || "?")}</code> → <span>${escapeHtml(short(ev.content || "", 240))}</span>`;
  } else if (kind === "user") {
    body = `<div class="ev-text">${escapeHtml(short(ev.content || "", 400))}</div>`;
  } else if (kind === "session_start") {
    const model = ev.model ? `model: <code>${escapeHtml(String(ev.model).split("/").pop())}</code>` : "";
    const tools = Array.isArray(ev.tools) ? `${ev.tools.length} tool${ev.tools.length === 1 ? "" : "s"}` : "";
    body = `<span class="dim">${[model, tools].filter(Boolean).join(" · ")}</span>`;
  } else if (kind === "session_end") {
    body = `<span class="dim">end of session</span>`;
  } else {
    // Unknown kind — show JSON minus already-displayed fields.
    const trimmed = { ...ev };
    delete trimmed.kind; delete trimmed.event; delete trimmed.ts; delete trimmed.t;
    body = `<code class="ev-raw">${escapeHtml(short(JSON.stringify(trimmed), 240))}</code>`;
  }
  return `<li class="ev ev-${kind}"><span class="ev-kind">${escapeHtml(kind)}</span><span class="ev-ts dim">${tsStr}</span><div class="ev-body">${body}</div></li>`;
}

function short(s, n) { s = String(s); return s.length > n ? s.slice(0, n) + "…" : s; }
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ----- mutations -----

async function agentAction(aid, action) {
  try {
    const r = await fetch(`/api/agents/${aid}/${action}`, { method: "POST" });
    if (!r.ok && r.status !== 202) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    toast(action === "run-now" ? "queued" : "cancel sent", "info");
    await refreshAgents();
    if (selectedId === aid) await loadAgentDetail(aid);
  } catch (err) {
    toast(`${action} failed: ${err.message}`);
  }
}

async function agentPatch(aid, patch) {
  try {
    const r = await fetch(`/api/agents/${aid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    await refreshAgents();
    if (selectedId === aid) await loadAgentDetail(aid);
  } catch (err) {
    toast(`update failed: ${err.message}`);
  }
}

async function agentDelete(aid) {
  try {
    const r = await fetch(`/api/agents/${aid}`, { method: "DELETE" });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    if (selectedId === aid) {
      selectedId = null;
      selectedRunId = null;
      renderEmptyDetail();
    }
    await refreshAgents();
    toast("agent deleted", "info");
  } catch (err) {
    toast(`delete failed: ${err.message}`);
  }
}

// ----- polling lifecycle -----

function startAgentsPolling() {
  stopAgentsPolling();
  const tick = async () => {
    if (document.body.dataset.tab !== "agents") {
      stopAgentsPolling();
      return;
    }
    await refreshAgents();
    if (selectedId) {
      // Only re-fetch detail if status might have changed (cheap, low frequency).
      // Skip if a modal is open to avoid layout thrash.
      if (!agentModal.open) await loadAgentDetail(selectedId);
    }
    pollTimer = setTimeout(tick, 4000);
  };
  pollTimer = setTimeout(tick, 0);
}

function stopAgentsPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}

// Hook into tab switches.
const _origShowTab = showTab;
window.showTab = function(name) {
  _origShowTab(name);
  if (name === "agents") {
    startAgentsPolling();
  } else {
    stopAgentsPolling();
  }
};
// re-bind tab buttons through the wrapper
for (const btn of $$(".tab")) {
  btn.removeEventListener("click", () => {});
  btn.onclick = () => window.showTab(btn.dataset.tab);
}
window.addEventListener("hashchange", () => window.showTab(location.hash.replace("#", "")));
// Kick off if we landed on the agents tab via URL.
if (document.body.dataset.tab === "agents") {
  startAgentsPolling();
} else {
  // Still do a one-off fetch so the sidebar count is right.
  refreshAgents();
}

// ----------------------------- keyboard shortcuts -----------------------
//   ⌘/Ctrl+K     — focus the composer (or focus search if you add one)
//   ⌘/Ctrl+L     — start a fresh chat (matches the terminal idiom)
//   ⌘/Ctrl+/     — show shortcut help
//   Esc          — close lightbox / abort streaming (already wired)
document.addEventListener("keydown", (ev) => {
  const mod = ev.metaKey || ev.ctrlKey;
  // Skip when the user is typing in a contenteditable / input that isn't
  // the composer (e.g. agent modal fields). The composer captures its
  // own keys above.
  if (mod && ev.key === "k" && !ev.shiftKey) {
    ev.preventDefault();
    showTab("chat");
    input?.focus();
    return;
  }
  if (mod && ev.key === "l" && !ev.shiftKey) {
    if (document.activeElement === input) return;  // ⌘L in textarea = clear line
    ev.preventDefault();
    if (!abortController) newSession();
    input?.focus();
    return;
  }
  if (mod && ev.key === "/") {
    ev.preventDefault();
    toggleShortcutHelp();
    return;
  }
  if (mod && ev.shiftKey && (ev.key === "d" || ev.key === "D")) {
    ev.preventDefault();
    cycleTheme();
    return;
  }
  if (ev.key === "Escape") {
    const lb = document.getElementById("lightbox");
    if (lb && !lb.hidden) { closeLightbox(); return; }
  }
});

function toggleShortcutHelp() {
  let dlg = document.getElementById("shortcut-help");
  if (dlg) { dlg.open ? dlg.close() : dlg.showModal(); return; }
  dlg = document.createElement("dialog");
  dlg.id = "shortcut-help";
  dlg.className = "modal";
  dlg.innerHTML = `
    <header class="modal-head"><h3>Shortcuts</h3>
      <button class="icon-btn" data-close>×</button></header>
    <div class="modal-body shortcut-list">
      <div><kbd>⌘K</kbd> focus composer</div>
      <div><kbd>⌘L</kbd> new chat</div>
      <div><kbd>⌘⇧D</kbd> cycle theme (auto / light / dark)</div>
      <div><kbd>⌘/</kbd> this help</div>
      <div><kbd>⌘↵</kbd> send message</div>
      <div><kbd>Esc</kbd> abort streaming · close dialogs</div>
    </div>`;
  document.body.appendChild(dlg);
  dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
  dlg.showModal();
}

// ============================================================================
// GRAPHS panel — list, design, run; lives in the Agents tab sidebar.
// ============================================================================

const graphListEl = document.querySelector("#graph-list");
const graphModal = document.querySelector("#graph-modal");
const graphForm = document.querySelector("#graph-form");
const graphRunModal = document.querySelector("#graph-run-modal");
const graphRunForm = document.querySelector("#graph-run-form");
const graphDesignStatus = document.querySelector("#graph-design-status");
const graphDesignBtn = document.querySelector("#graph-design-btn");

let _graphs = [];
// In-flight + recent runs by graph name. State preserved across modal close
// so the user can click away, the run keeps streaming, and reopening
// re-attaches with the live event list. Status moves running → done | failed
// | canceled. We keep the LAST run per graph for replay; no history depth.
const _runs = new Map();

function _runState(name) { return _runs.get(name); }
function _setRun(name, run) { _runs.set(name, run); _renderRunBadgeFor(name); }

function _renderRunBadgeFor(name) {
  // Update just the matching list row's badge if rendered.
  const li = graphListEl?.querySelector(`[data-graph="${name}"]`);
  if (!li) return;
  const badgeEl = li.querySelector(".run-badge");
  if (badgeEl) badgeEl.remove();
  const run = _runState(name);
  if (!run) return;
  const badge = document.createElement("span");
  badge.className = `run-badge run-badge-${run.status}`;
  if (run.status === "running") {
    const completed = run.events.filter(e => e.kind === "node_end" || e.kind === "node_skipped").length;
    const total = run.totalNodes || 0;
    badge.innerHTML = `<span class="dot running"></span> ${completed}/${total || "?"}`;
  } else if (run.status === "done") {
    badge.innerHTML = `<span class="dot done"></span> ${run.wall_s ?? "?"}s`;
  } else if (run.status === "canceled") {
    badge.innerHTML = `<span class="dot skipped"></span> canceled`;
  } else {
    badge.innerHTML = `<span class="dot err"></span> failed`;
  }
  li.querySelector(".graph-meta").appendChild(badge);
}

async function loadGraphs() {
  if (!graphListEl) return;
  try {
    const r = await fetch("/api/graphs");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    _graphs = d.graphs || [];
  } catch (e) {
    graphListEl.innerHTML = `<li class="empty">load failed: ${e.message}</li>`;
    return;
  }
  if (!_graphs.length) {
    graphListEl.innerHTML = `<li class="empty">no graphs — click + to design one</li>`;
    return;
  }
  graphListEl.innerHTML = "";
  for (const g of _graphs) {
    const li = document.createElement("li");
    li.className = "graph-list-item";
    li.dataset.graph = g.name;
    const rmBtn = `<button class="graph-rm" title="delete graph" aria-label="delete graph: ${escapeHTML(g.name)}" data-name="${escapeHTML(g.name)}">×</button>`;
    if (g.error) {
      li.innerHTML = `<div class="graph-name">${escapeHTML(g.name)} ${rmBtn}</div>
                      <div class="graph-meta error">load error</div>`;
    } else {
      const nodeCount = (g.nodes || []).length;
      const tools = new Set();
      for (const n of (g.nodes || [])) for (const t of (n.tools || [])) tools.add(t);
      li.innerHTML = `<div class="graph-name">${escapeHTML(g.name)} ${rmBtn}</div>
                      <div class="graph-meta">${nodeCount} nodes${tools.size ? ` · ${tools.size} tools` : ""}</div>`;
      li.addEventListener("click", (ev) => {
        if (ev.target.classList.contains("graph-rm")) return;
        openGraphRun(g);
      });
    }
    const rm = li.querySelector(".graph-rm");
    rm?.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const name = ev.currentTarget.dataset.name;
      // If the graph is currently running on the server, ask the user
      // first — deletion of the file won't kill the in-flight run on the
      // server (it has the module already loaded), but it'll be confusing
      // if results come back for a graph that no longer exists locally.
      const inflight = _runs.get(name);
      if (inflight && inflight.status === "running") {
        if (!confirm(`Graph "${name}" is currently running. Cancel the run and delete it?`)) return;
        try { inflight.controller?.abort(); } catch {}
        inflight.status = "canceled";
        _setRun(name, inflight);
      } else {
        if (!confirm(`Delete graph "${name}"? The Python file will be removed from examples/.`)) return;
      }
      try {
        const r = await fetch("/api/graphs/delete", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({name}),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          toast(`delete failed: ${d.error || r.status}`, "warn");
          return;
        }
        _runs.delete(name);
        loadGraphs();
      } catch (err) {
        toast(`delete failed: ${err.message}`, "warn");
      }
    });
    graphListEl.appendChild(li);
    _renderRunBadgeFor(g.name);
  }
}

// Light tick to keep "running" badges live (animation alone isn't visible
// when only the dot pulses; the N/M counter changes with each node end).
setInterval(() => {
  for (const [name, run] of _runs) {
    if (run.status === "running") _renderRunBadgeFor(name);
  }
}, 800);

function escapeHTML(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function openGraphRun(graph) {
  graphRunModal.dataset.graph = graph.name;
  document.querySelector("#graph-run-title").textContent = `Run · ${graph.name}`;
  // Render summary
  const summaryEl = document.querySelector("#graph-run-summary");
  const nodes = graph.nodes || [];
  summaryEl.innerHTML = `
    <div class="graph-detail-head">
      <strong>${escapeHTML(graph.name)}</strong>
      <span class="muted">${nodes.length} nodes · entry: ${(graph.entry_inputs||[]).join(", ") || "—"}</span>
    </div>
    <div class="graph-diagram" id="graph-run-diagram"></div>
    <details class="graph-node-list-wrap">
      <summary class="muted">node details (${nodes.length})</summary>
      <ol class="graph-node-list">
        ${nodes.map(n => `
          <li>
            <span class="node-name">${escapeHTML(n.name)}</span>
            <span class="muted">${escapeHTML(n.role || "")}</span>
            ${(n.tools || []).length ? `<span class="muted">tools: ${(n.tools||[]).join(", ")}</span>` : ""}
          </li>`).join("")}
      </ol>
    </details>`;
  // Render diagram async (mermaid is async).
  renderGraphDiagram(document.querySelector("#graph-run-diagram"), graph);
  // Render input form (pre-fill from last run's inputs if present)
  const inputsEl = document.querySelector("#graph-run-inputs");
  inputsEl.innerHTML = "";
  const prev = _runState(graph.name);
  for (const inp of (graph.entry_inputs || [])) {
    const wrap = document.createElement("label");
    wrap.className = "graph-input";
    const seed = prev?.inputs?.[inp] || "";
    wrap.innerHTML = `<span>${escapeHTML(inp)}</span>
      <input type="text" name="${escapeHTML(inp)}" required placeholder="${escapeHTML(inp)}…" value="${escapeHTML(seed)}">`;
    inputsEl.appendChild(wrap);
  }
  const eventsEl = document.querySelector("#graph-run-events");
  const outputEl = document.querySelector("#graph-run-output");
  // Configure the action button + replay any prior/in-flight events
  const runBtn = document.querySelector("#graph-run-btn");
  const cancelBtn = _ensureCancelButton();
  if (prev && prev.status === "running") {
    runBtn.textContent = "Run";
    runBtn.disabled = true;
    cancelBtn.hidden = false;
    eventsEl.hidden = false;
    outputEl.hidden = true;
    eventsEl.innerHTML = "";
    for (const ev of prev.events) _renderEventInto(eventsEl, ev);
    // Subscribe so subsequent events keep flowing into THIS modal too:
    prev.modalEventsEl = eventsEl;
    prev.modalOutputEl = outputEl;
  } else if (prev && prev.outputs) {
    runBtn.textContent = "Re-run";
    runBtn.disabled = false;
    cancelBtn.hidden = true;
    eventsEl.hidden = false;
    outputEl.hidden = false;
    eventsEl.innerHTML = "";
    for (const ev of prev.events) _renderEventInto(eventsEl, ev);
    outputEl.innerHTML = `<h4>Last outputs (${prev.wall_s ?? "?"}s · ${prev.status})</h4><pre>${escapeHTML(
      JSON.stringify(prev.outputs || {}, null, 2).slice(0, 6000))}</pre>`;
  } else {
    runBtn.textContent = "Run";
    runBtn.disabled = false;
    cancelBtn.hidden = true;
    eventsEl.hidden = true;
    outputEl.hidden = true;
  }
  graphRunModal.showModal();
}

function _ensureCancelButton() {
  let btn = document.querySelector("#graph-run-cancel");
  if (btn) return btn;
  btn = document.createElement("button");
  btn.type = "button";
  btn.id = "graph-run-cancel";
  btn.className = "warn-btn";
  btn.textContent = "Cancel";
  btn.hidden = true;
  btn.addEventListener("click", () => {
    const name = graphRunModal.dataset.graph;
    const run = _runState(name);
    if (!run || run.status !== "running") return;
    try { run.controller?.abort(); } catch {}
    run.status = "canceled";
    _setRun(name, run);
    btn.hidden = true;
    document.querySelector("#graph-run-btn").disabled = false;
  });
  // Insert before the existing Run button in the modal footer.
  const foot = graphRunForm.querySelector(".modal-foot") || graphRunForm.lastElementChild;
  foot?.insertBefore(btn, document.querySelector("#graph-run-btn"));
  return btn;
}

function _renderEventInto(eventsEl, ev) {
  if (ev.kind === "node_start") {
    if (eventsEl.querySelector(`[data-node="${ev.node}"]`)) return;
    const row = document.createElement("div");
    row.className = "graph-event node-start";
    row.dataset.node = ev.node;
    row.innerHTML = `<span class="dot running"></span>
      <strong>${escapeHTML(ev.node)}</strong>
      <span class="muted">${escapeHTML(ev.role || "")}${(ev.inputs||[]).length ? " · " + (ev.inputs||[]).join(", ") : ""}</span>`;
    eventsEl.appendChild(row);
  } else if (ev.kind === "node_end") {
    const row = eventsEl.querySelector(`[data-node="${ev.node}"]`);
    if (!row) return;
    const stats = ev.stats || {};
    row.querySelector(".dot")?.classList.replace("running", "done");
    const parts = [];
    if (stats.wall_s != null) parts.push(`${stats.wall_s}s`);
    if (stats.n_tool_calls) parts.push(`${stats.n_tool_calls} tools`);
    if (stats.cache_hits) parts.push(`${stats.cache_hits} cached`);
    if (stats.max_msgs_tokens) parts.push(`peak ${stats.max_msgs_tokens}t`);
    if (stats.steps) parts.push(`${stats.steps}st`);
    if (stats.skipped_upstream_error) parts.push("skipped (upstream err)");
    if (stats.map_n) parts.push(`map×${stats.map_n}`);
    if (stats.batched_map) parts.push("batched_map");
    if (stats.finalize_nudged) parts.push("nudged");
    if (!row.querySelector(".graph-event-stats")) {
      const meta = document.createElement("span");
      meta.className = "muted graph-event-stats";
      meta.textContent = parts.length ? ` · ${parts.join(" · ")}` : "";
      row.appendChild(meta);
    }
    if (!row.querySelector(".graph-event-output")) {
      const outs = ev.outputs || {};
      const outKeys = Object.keys(outs).filter(k => !k.startsWith("_"));
      if (outKeys.length) {
        const previewLine = document.createElement("div");
        previewLine.className = "graph-event-output";
        previewLine.innerHTML = outKeys.slice(0, 3).map(k => {
          let v = outs[k];
          if (typeof v !== "string") v = JSON.stringify(v);
          v = String(v).replace(/\s+/g, " ").trim();
          if (v.length > 100) v = v.slice(0, 100) + "…";
          return `<span class="o-key">${escapeHTML(k)}:</span> <span class="o-val">${escapeHTML(v)}</span>`;
        }).join(" · ");
        row.appendChild(previewLine);
      }
    }
  } else if (ev.kind === "node_skipped") {
    if (eventsEl.querySelector(`[data-node="${ev.node}"]`)) return;
    const row = document.createElement("div");
    row.className = "graph-event node-skipped";
    row.dataset.node = ev.node;
    row.innerHTML = `<span class="dot skipped"></span>
      <strong>${escapeHTML(ev.node)}</strong>
      <span class="muted">skipped — ${escapeHTML(ev.reason||"")}</span>`;
    eventsEl.appendChild(row);
  } else if (ev.kind === "graph_end") {
    // visual marker
  } else if (ev.kind === "error") {
    const e = document.createElement("div");
    e.className = "graph-event err";
    e.textContent = `error: ${ev.message || ""}`;
    eventsEl.appendChild(e);
  }
}

if (graphRunForm) {
  graphRunForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const graphName = graphRunModal.dataset.graph;
    const inputs = {};
    for (const i of graphRunForm.querySelectorAll("input[name]")) inputs[i.name] = i.value;
    const graphSpec = _graphs.find(g => g.name === graphName);
    const totalNodes = (graphSpec?.nodes || []).length;
    const eventsEl = document.querySelector("#graph-run-events");
    const outputEl = document.querySelector("#graph-run-output");
    const runBtn = document.querySelector("#graph-run-btn");
    const cancelBtn = _ensureCancelButton();
    eventsEl.hidden = false;
    outputEl.hidden = true;
    eventsEl.innerHTML = "";
    runBtn.disabled = true;
    cancelBtn.hidden = false;

    const controller = new AbortController();
    const startedAt = Date.now();
    const run = {
      name: graphName,
      inputs,
      events: [],
      outputs: null,
      status: "running",
      controller,
      totalNodes,
      startedAt,
      modalEventsEl: eventsEl,
      modalOutputEl: outputEl,
    };
    _setRun(graphName, run);

    runStreamingGraph(run).catch(err => {
      // Errors are surfaced inside the runner; this is just a guard.
      console.error("graph runner crashed", err);
    });
  });
}

async function runStreamingGraph(run) {
  const eventsRef = () => run.modalEventsEl; // re-evaluated so reopens reattach
  const outputRef = () => run.modalOutputEl;
  const runBtn = document.querySelector("#graph-run-btn");
  const cancelBtn = document.querySelector("#graph-run-cancel");
  try {
    const resp = await fetch("/api/graph/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({graph: run.name, inputs: run.inputs}),
      signal: run.controller.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, nl);
        buf = buf.slice(nl + 2);
        let event = "message", data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        let payload = {};
        try { payload = JSON.parse(data); } catch {}
        const ev = {kind: event, ...payload};
        run.events.push(ev);
        const eventsEl = eventsRef();
        if (eventsEl) _renderEventInto(eventsEl, ev);
        if (event === "done") {
          run.outputs = payload.outputs || {};
          run.wall_s = payload.wall_s;
        } else if (event === "node_end" || event === "node_skipped") {
          _renderRunBadgeFor(run.name);
        }
      }
    }
    run.status = "done";
    const outEl = outputRef();
    if (outEl) {
      outEl.hidden = false;
      outEl.innerHTML = `<h4>Outputs <span class="muted">(${run.wall_s ?? "?"}s)</span></h4><pre>${escapeHTML(
        JSON.stringify(run.outputs || {}, null, 2).slice(0, 6000))}</pre>`;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      run.status = "canceled";
    } else {
      run.status = "failed";
      run.error = err.message;
      const eventsEl = eventsRef();
      if (eventsEl) {
        const e = document.createElement("div");
        e.className = "graph-event err";
        e.textContent = `stream failed: ${err.message}`;
        eventsEl.appendChild(e);
      }
    }
  } finally {
    _setRun(run.name, run);
    if (graphRunModal.dataset.graph === run.name) {
      runBtn.disabled = false;
      runBtn.textContent = "Re-run";
      if (cancelBtn) cancelBtn.hidden = true;
    }
  }
}

// ---- Graph designer: preview → edit → save flow ----
//
// Two-phase UX so the user owns the final architecture:
//   1. user describes the graph in plain English
//   2. server returns a proposed spec (nodes/edges/tools)
//   3. user edits anything they want in the preview modal
//   4. saving validates the spec server-side and writes the file
//
// The user can also build a graph from scratch via the manual builder modal
// (gear button), which skips phase 1 entirely.

const ALLOWED_TOOLS = [
  "web_search", "web_fetch", "now", "make_table",
  "read_file", "list_files", "grep", "csv_summary",
  "github_repo", "arxiv_search",
];
const OUTPUT_TAGS = ["t", "j", "l", "n", "b", "kv"];
// Friendly labels for the output-tag dropdown — the model-facing letters
// ("t", "j", etc) come from the AGFMT spec and are inscrutable to users.
const OUTPUT_TAG_LABEL = {
  t:  "text",
  j:  "JSON",
  l:  "list",
  n:  "number",
  b:  "bool (yes/no)",
  kv: "key-value",
};

// ---- Mermaid rendering of a graph spec --------------------------------
// Translates a {nodes, edges} spec into a Mermaid `flowchart LR` block.
// Used by the graph run modal (visualizes the saved graph) and the live
// preview at the top of the build/edit modals (re-rendered on every edit).

function _mermaidId(name) {
  // Mermaid node IDs must match `[A-Za-z][A-Za-z0-9_]*`. Our node names
  // are already snake_case so this is just a no-op safety net.
  return String(name).replace(/[^A-Za-z0-9_]/g, "_") || "n";
}

function _mermaidEscape(s) {
  // Escape characters that break mermaid label parsing inside `["..."]`.
  return String(s).replace(/[\\]/g, "\\\\").replace(/"/g, "&quot;")
                  .replace(/\n/g, "<br/>");
}

function specToMermaid(spec) {
  if (!spec || !Array.isArray(spec.nodes) || !spec.nodes.length) {
    return "graph LR\n  empty[\"(no nodes)\"]";
  }
  const lines = ["flowchart LR"];
  for (const n of spec.nodes) {
    const id = _mermaidId(n.name);
    const role = n.role || "specialist";
    const tools = (n.tools || []).slice(0, 3).join(", ");
    const toolStr = (n.tools || []).length
      ? `<br/><span style='font-size:11px;opacity:0.7'>🔧 ${_mermaidEscape(tools)}${(n.tools||[]).length > 3 ? "…" : ""}</span>`
      : "";
    const mapStr = n.map_over
      ? `<br/><span style='font-size:11px;opacity:0.7'>↻ map: ${_mermaidEscape(n.map_over)}</span>`
      : "";
    const label = `<b>${_mermaidEscape(n.name)}</b><br/><span style='font-size:11px;opacity:0.6'>${_mermaidEscape(role)}</span>${toolStr}${mapStr}`;
    lines.push(`  ${id}["${label}"]`);
  }
  for (const e of (spec.edges || [])) {
    const src = _mermaidId(e.src);
    const dst = _mermaidId(e.dst);
    if (e.when) {
      // dashed conditional edge with the gate as the label
      lines.push(`  ${src} -. "${_mermaidEscape(e.when)}" .-> ${dst}`);
    } else {
      lines.push(`  ${src} --> ${dst}`);
    }
  }
  // Style: highlight nodes that have tools vs synthesis nodes
  for (const n of spec.nodes) {
    const id = _mermaidId(n.name);
    if ((n.tools || []).length) {
      lines.push(`  style ${id} fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000`);
    } else {
      lines.push(`  style ${id} fill:#fff8e1,stroke:#f57c00,stroke-width:1px,color:#000`);
    }
  }
  return lines.join("\n");
}

let _mermaidRunCounter = 0;
async function renderGraphDiagram(targetEl, spec) {
  if (!targetEl) return;
  if (typeof mermaid === "undefined") {
    targetEl.innerHTML = `<div class="muted">Mermaid not loaded — diagram unavailable.</div>`;
    return;
  }
  const code = specToMermaid(spec);
  targetEl.dataset.mermaidSrc = code;
  const id = `mmd-${++_mermaidRunCounter}-${Date.now()}`;
  try {
    const {svg} = await mermaid.render(id, code);
    targetEl.innerHTML = svg;
  } catch (e) {
    targetEl.innerHTML = `<details><summary class="muted">diagram render failed: ${escapeHTML(e.message || String(e))}</summary><pre>${escapeHTML(code)}</pre></details>`;
  }
}

// Carries the original NL description across phases so the "Regenerate"
// button in the preview/edit modal can re-call the designer with the same
// input. The spec is not stored here — it lives in the modal's DOM until save.
let _designContext = {description: ""};

if (graphForm) {
  graphForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const description = graphForm.querySelector("textarea[name=description]").value.trim();
    if (description.length < 8) { toast("description too short"); return; }
    graphDesignStatus.hidden = false;
    graphDesignStatus.classList.remove("error");
    graphDesignStatus.textContent = "designing graph… (~10-30s)";
    graphDesignBtn.disabled = true;
    try {
      const r = await fetch("/api/graphs/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({description}),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.ok) {
        graphDesignStatus.classList.add("error");
        graphDesignStatus.textContent = `failed: ${d.error || ("HTTP " + r.status)}`;
        return;
      }
      _designContext.description = description;
      graphModal.close();
      openPreviewEditor(d.spec, d.code, description);
    } catch (err) {
      graphDesignStatus.classList.add("error");
      graphDesignStatus.textContent = `failed: ${err.message}`;
    } finally {
      graphDesignBtn.disabled = false;
    }
  });
}

document.querySelector("#new-graph-btn")?.addEventListener("click", () => {
  document.querySelector("#graph-design-status").hidden = true;
  graphForm.reset();
  graphModal.showModal();
});
document.querySelector("#welcome-new-graph")?.addEventListener("click", () => {
  document.querySelector("#new-graph-btn")?.click();
});
document.querySelector("#welcome-build-graph")?.addEventListener("click", () => {
  document.querySelector("#build-graph-btn")?.click();
});

// ---- Manual builder: ⚙ button next to + ----
const buildModal = document.querySelector("#graph-build-modal");
const buildForm = document.querySelector("#graph-build-form");
const buildStatus = document.querySelector("#gb-status");
document.querySelector("#build-graph-btn")?.addEventListener("click", () => {
  // Seed with a single skeleton node so the user has something to edit.
  document.querySelector("#gb-name").value = "";
  document.querySelector("#gb-desc").value = "";
  buildStatus.hidden = true;
  buildStatus.classList.remove("error");
  const nodesEl = document.querySelector("#gb-nodes");
  const edgesEl = document.querySelector("#gb-edges");
  nodesEl.innerHTML = "";
  edgesEl.innerHTML = "";
  nodesEl.appendChild(_renderNodeCard({
    name: "step1", role: "specialist",
    goal: "Describe what this node should do.",
    inputs: [], outputs: [["result", "t"]],
    tools: null, max_steps: 6,
    map_over: null, map_item_key: null,
    batch_map: false, extra_instructions: "",
  }));
  buildModal.showModal();
  _maybeRefreshLivePreview();
});

document.querySelector("#gb-add-node")?.addEventListener("click", (e) => {
  e.preventDefault();
  const nodesEl = document.querySelector("#gb-nodes");
  const i = nodesEl.children.length + 1;
  nodesEl.appendChild(_renderNodeCard({
    name: `step${i}`, role: "specialist",
    goal: "Describe what this node should do.",
    inputs: [], outputs: [["result", "t"]],
    tools: null, max_steps: 6,
    map_over: null, map_item_key: null,
    batch_map: false, extra_instructions: "",
  }));
  _maybeRefreshLivePreview();
});

document.querySelector("#gb-add-edge")?.addEventListener("click", (e) => {
  e.preventDefault();
  const edgesEl = document.querySelector("#gb-edges");
  edgesEl.appendChild(_renderEdgeRow("", "", "", _collectNodeNames("#gb-nodes")));
  _maybeRefreshLivePreview();
});

if (buildForm) {
  buildForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.querySelector("#gb-name").value.trim();
    const desc = document.querySelector("#gb-desc").value.trim();
    if (!/^[a-z][a-z0-9_]*$/.test(name)) {
      _setStatus(buildStatus, "name must be snake_case (lowercase, digits, underscore)", true);
      return;
    }
    const nodes = _collectNodes("#gb-nodes");
    if (!nodes.length) {
      _setStatus(buildStatus, "at least one node is required", true);
      return;
    }
    const edges = _collectEdges("#gb-edges", nodes.map(n => n.name));
    const spec = {name, nodes, edges};
    _setStatus(buildStatus, "saving graph…", false);
    try {
      const r = await fetch("/api/graphs/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({spec, description: desc}),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.ok) {
        _setStatus(buildStatus, `failed: ${d.error || ("HTTP " + r.status)}`, true);
        return;
      }
      _setStatus(buildStatus, `saved ${d.name} → ${d.path}`, false);
      await loadGraphs();
      setTimeout(() => { buildModal.close(); buildStatus.hidden = true; }, 700);
    } catch (err) {
      _setStatus(buildStatus, `save failed: ${err.message}`, true);
    }
  });
}

// ---- Preview/edit modal ----
const previewModal = document.querySelector("#graph-preview-modal");
const previewForm = document.querySelector("#graph-preview-form");
const previewStatus = document.querySelector("#gpv-status");

function openPreviewEditor(spec, codePreview, description) {
  document.querySelector("#gpv-name").value = spec.name || "";
  const nodesEl = document.querySelector("#gpv-nodes");
  const edgesEl = document.querySelector("#gpv-edges");
  nodesEl.innerHTML = "";
  edgesEl.innerHTML = "";
  for (const n of (spec.nodes || [])) {
    nodesEl.appendChild(_renderNodeCard(n));
  }
  const nodeNames = (spec.nodes || []).map(n => n.name);
  for (const e of (spec.edges || [])) {
    edgesEl.appendChild(_renderEdgeRow(e.src, e.dst, e.when || "", nodeNames));
  }
  document.querySelector("#gpv-code").textContent = codePreview || "";
  previewStatus.hidden = true;
  previewStatus.classList.remove("error");
  _designContext.description = description || "";
  previewModal.showModal();
  _maybeRefreshLivePreview();
}

document.querySelector("#gpv-add-node")?.addEventListener("click", (e) => {
  e.preventDefault();
  const nodesEl = document.querySelector("#gpv-nodes");
  const i = nodesEl.children.length + 1;
  nodesEl.appendChild(_renderNodeCard({
    name: `node${i}`, role: "specialist",
    goal: "Describe what this node should do.",
    inputs: [], outputs: [["result", "t"]],
    tools: null, max_steps: 6,
    map_over: null, map_item_key: null,
    batch_map: false, extra_instructions: "",
  }));
  _maybeRefreshLivePreview();
});

document.querySelector("#gpv-add-edge")?.addEventListener("click", (e) => {
  e.preventDefault();
  const edgesEl = document.querySelector("#gpv-edges");
  edgesEl.appendChild(_renderEdgeRow("", "", "", _collectNodeNames("#gpv-nodes")));
  _maybeRefreshLivePreview();
});

document.querySelector("#gpv-regen")?.addEventListener("click", async (e) => {
  e.preventDefault();
  const desc = _designContext.description;
  if (!desc) { toast("no original description to regenerate from"); return; }
  _setStatus(previewStatus, "regenerating…", false);
  try {
    const r = await fetch("/api/graphs/preview", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({description: desc}),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !d.ok) {
      _setStatus(previewStatus, `regenerate failed: ${d.error || r.status}`, true);
      return;
    }
    openPreviewEditor(d.spec, d.code, desc);
  } catch (err) {
    _setStatus(previewStatus, `regenerate failed: ${err.message}`, true);
  }
});

if (previewForm) {
  previewForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.querySelector("#gpv-name").value.trim();
    if (!/^[a-z][a-z0-9_]*$/.test(name)) {
      _setStatus(previewStatus, "name must be snake_case", true);
      return;
    }
    const nodes = _collectNodes("#gpv-nodes");
    if (!nodes.length) {
      _setStatus(previewStatus, "at least one node is required", true);
      return;
    }
    const edges = _collectEdges("#gpv-edges", nodes.map(n => n.name));
    const spec = {name, nodes, edges};
    _setStatus(previewStatus, "saving graph…", false);
    try {
      const r = await fetch("/api/graphs/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({spec, description: _designContext.description}),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.ok) {
        _setStatus(previewStatus, `failed: ${d.error || r.status}`, true);
        return;
      }
      _setStatus(previewStatus, `saved ${d.name} → ${d.path}`, false);
      await loadGraphs();
      setTimeout(() => { previewModal.close(); previewStatus.hidden = true; }, 700);
    } catch (err) {
      _setStatus(previewStatus, `save failed: ${err.message}`, true);
    }
  });
}

// ---- helpers for rendering and collecting node/edge editor cards ----
function _setStatus(el, msg, isErr) {
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
  el.classList.toggle("error", !!isErr);
}

function _collectNodeNames(sel) {
  return Array.from(document.querySelectorAll(`${sel} .gpv-node-card .gpv-name`))
    .map(i => i.value.trim()).filter(Boolean);
}

function _renderOutputRow(name, tag) {
  // One output = name input + dropdown showing the friendly label.
  const tagOpts = OUTPUT_TAGS.map(t =>
    `<option value="${t}" ${t === (tag || "t") ? "selected" : ""}>${OUTPUT_TAG_LABEL[t] || t}</option>`
  ).join("");
  const row = document.createElement("div");
  row.className = "gpv-output-row";
  row.innerHTML = `
    <input class="gpv-output-name" type="text" value="${escapeHTML(name || "")}"
           placeholder="output name" required>
    <select class="gpv-output-tag" title="output type">${tagOpts}</select>
    <button type="button" class="gpv-output-rm" title="remove output" tabindex="-1">×</button>
  `;
  row.querySelector(".gpv-output-rm").addEventListener("click", () => {
    // Don't allow removing the last output — node must have ≥1.
    const list = row.parentElement;
    if (list && list.querySelectorAll(".gpv-output-row").length > 1) row.remove();
    _maybeRefreshLivePreview();
  });
  row.addEventListener("input", _maybeRefreshLivePreview);
  return row;
}

function _renderNodeCard(node) {
  const card = document.createElement("li");
  card.className = "gpv-node-card";
  const tools = node.tools || [];
  const outputs = (node.outputs || []).length
    ? node.outputs : [["result", "t"]];
  const inputs = (node.inputs || []).join(", ");
  const toolBoxes = ALLOWED_TOOLS.map(t => `
    <label><input type="checkbox" class="gpv-tool" value="${t}"
      ${tools.includes(t) ? "checked" : ""}> ${t}</label>`).join("");
  card.innerHTML = `
    <button type="button" class="gpv-node-rm" title="remove node">×</button>
    <label>
      <span>Name <small class="muted">(snake_case)</small></span>
      <input class="gpv-name" type="text" value="${escapeHTML(node.name || "")}"
             pattern="[a-z][a-z0-9_]*" required>
    </label>
    <label>
      <span>Role <small class="muted">(short label, shown in graph)</small></span>
      <input class="gpv-role" type="text" value="${escapeHTML(node.role || "specialist")}">
    </label>
    <label class="gpv-row-full">
      <span>Goal <small class="muted">(this becomes the agent's prompt — be specific)</small></span>
      <textarea class="gpv-goal" rows="2" required>${escapeHTML(node.goal || "")}</textarea>
    </label>
    <label>
      <span>Inputs <small class="muted">(comma-separated names from upstream nodes)</small></span>
      <input class="gpv-inputs" type="text" value="${escapeHTML(inputs)}"
             placeholder="topic, year">
    </label>
    <label class="gpv-row-full">
      <span>Outputs <small class="muted">(at least one; type tells downstream nodes what to expect)</small></span>
      <div class="gpv-outputs-list"></div>
      <button type="button" class="gpv-output-add">+ add output</button>
    </label>
    <label>
      <span>Max steps <small class="muted">(2 for synthesis, 6–8 with tools)</small></span>
      <input class="gpv-max-steps" type="number" min="1" max="12"
             value="${node.max_steps || 6}">
    </label>
    <label>
      <span>Map over <small class="muted">(if set, run once per item in this list-typed input)</small></span>
      <input class="gpv-map-over" type="text" value="${escapeHTML(node.map_over || "")}"
             placeholder="leave blank for normal node">
    </label>
    <label class="gpv-row-full">
      <span>Tools <small class="muted">(only checked tools are available to this node)</small></span>
      <div class="gpv-tools-grid">${toolBoxes}</div>
    </label>
    <label class="gpv-row-full">
      <span>Extra instructions <small class="muted">(optional — appended to the prompt)</small></span>
      <textarea class="gpv-extra" rows="2">${escapeHTML(node.extra_instructions || "")}</textarea>
    </label>
  `;
  // Populate the outputs list with row-per-output.
  const outsList = card.querySelector(".gpv-outputs-list");
  for (const o of outputs) {
    const [n, t] = Array.isArray(o)
      ? [o[0], o[1]]
      : (typeof o === "string" && o.includes(":")
          ? o.split(":") : [o, "t"]);
    outsList.appendChild(_renderOutputRow(n, t));
  }
  card.querySelector(".gpv-output-add").addEventListener("click", () => {
    outsList.appendChild(_renderOutputRow("", "t"));
    _maybeRefreshLivePreview();
  });
  card.querySelector(".gpv-node-rm").addEventListener("click", () => {
    card.remove();
    _maybeRefreshLivePreview();
  });
  // Refresh live diagram on any field change.
  card.addEventListener("input", _maybeRefreshLivePreview);
  return card;
}

function _renderEdgeRow(src, dst, when, nodeNames) {
  const row = document.createElement("li");
  row.className = "gpv-edge-row";
  const opts = (names) => names.map(n =>
    `<option value="${escapeHTML(n)}">${escapeHTML(n)}</option>`).join("");
  row.innerHTML = `
    <select class="gpv-edge-src" title="source node"><option value="">(from)</option>${opts(nodeNames)}</select>
    <span class="gpv-edge-arrow" aria-hidden="true">→</span>
    <select class="gpv-edge-dst" title="target node"><option value="">(to)</option>${opts(nodeNames)}</select>
    <input class="gpv-edge-when" type="text"
           placeholder="optional condition, e.g. category=='news'"
           title="Python expression evaluated against upstream outputs; edge only fires when true"
           value="${escapeHTML(when || "")}">
    <button type="button" class="gpv-edge-rm" title="remove edge" tabindex="-1">×</button>
  `;
  if (src) row.querySelector(".gpv-edge-src").value = src;
  if (dst) row.querySelector(".gpv-edge-dst").value = dst;
  row.querySelector(".gpv-edge-rm").addEventListener("click", () => {
    row.remove();
    _maybeRefreshLivePreview();
  });
  // Live-preview hook on any change.
  row.addEventListener("change", _maybeRefreshLivePreview);
  row.addEventListener("input", _maybeRefreshLivePreview);
  return row;
}

// Live preview — debounced re-render of the build/edit modal's diagram.
let _previewTimer = null;
function _maybeRefreshLivePreview() {
  clearTimeout(_previewTimer);
  _previewTimer = setTimeout(() => {
    // Detect which modal is open and grab its node/edge selectors.
    const buildOpen = document.querySelector("#graph-build-modal[open]");
    const previewOpen = document.querySelector("#graph-preview-modal[open]");
    const target = buildOpen ? "#gb" : (previewOpen ? "#gpv" : null);
    if (!target) return;
    const diagramEl = document.querySelector(`${target}-diagram`);
    if (!diagramEl) return;
    const nodes = _collectNodes(`${target}-nodes`);
    const edges = _collectEdges(`${target}-edges`, nodes.map(n => n.name));
    renderGraphDiagram(diagramEl, {
      name: document.querySelector(`${target}-name`)?.value || "preview",
      nodes, edges,
    });
  }, 300);
}

function _collectNodes(sel) {
  const cards = document.querySelectorAll(`${sel} .gpv-node-card`);
  const out = [];
  for (const c of cards) {
    const name = c.querySelector(".gpv-name").value.trim();
    if (!name) continue;
    // Read output rows: each row is [name input, tag select].
    const outputs = [];
    const outRows = c.querySelectorAll(".gpv-output-row");
    for (const r of outRows) {
      const oname = r.querySelector(".gpv-output-name").value.trim();
      if (!oname) continue;
      const tag = r.querySelector(".gpv-output-tag").value;
      outputs.push([oname, OUTPUT_TAGS.includes(tag) ? tag : "t"]);
    }
    // Backward-compat fallback: an older form may still have the legacy
    // `.gpv-outputs` text field. Parse it if no per-row outputs were found.
    if (!outputs.length) {
      const legacy = c.querySelector(".gpv-outputs");
      if (legacy && legacy.value.trim()) {
        for (const s of legacy.value.split(",").map(x => x.trim()).filter(Boolean)) {
          const [n, t] = s.split(":").map(x => (x || "").trim());
          outputs.push([n, OUTPUT_TAGS.includes(t) ? t : "t"]);
        }
      }
    }
    const tools = Array.from(c.querySelectorAll(".gpv-tool:checked")).map(i => i.value);
    const inputs = c.querySelector(".gpv-inputs").value
      .split(",").map(s => s.trim()).filter(Boolean);
    const mapOver = c.querySelector(".gpv-map-over").value.trim();
    out.push({
      name,
      role: c.querySelector(".gpv-role").value.trim() || "specialist",
      goal: c.querySelector(".gpv-goal").value.trim(),
      inputs,
      outputs,
      tools: tools.length ? tools : null,
      max_steps: parseInt(c.querySelector(".gpv-max-steps").value, 10) || 6,
      map_over: mapOver || null,
      map_item_key: null,
      batch_map: false,
      extra_instructions: c.querySelector(".gpv-extra").value.trim(),
    });
  }
  return out;
}

function _collectEdges(sel, nodeNames) {
  const rows = document.querySelectorAll(`${sel} .gpv-edge-row`);
  const seen = new Set(nodeNames);
  const out = [];
  for (const r of rows) {
    const src = r.querySelector(".gpv-edge-src").value;
    const dst = r.querySelector(".gpv-edge-dst").value;
    const when = r.querySelector(".gpv-edge-when").value.trim() || null;
    if (!src || !dst) continue;
    if (!seen.has(src) || !seen.has(dst)) continue;
    out.push({src, dst, when});
  }
  return out;
}

// ============================================================================
// MCPs panel — list, register, unregister
// ============================================================================

const mcpListEl = document.querySelector("#mcp-list");
const mcpModal = document.querySelector("#mcp-modal");
const mcpForm = document.querySelector("#mcp-form");

async function loadMCPs() {
  if (!mcpListEl) return;
  try {
    const r = await fetch("/api/mcps");
    const d = await r.json();
    const mcps = d.mcps || [];
    if (!mcps.length) {
      mcpListEl.innerHTML = `<li class="empty">none registered</li>`;
      return;
    }
    mcpListEl.innerHTML = "";
    for (const m of mcps) {
      const li = document.createElement("li");
      li.className = "mcp-list-item";
      const tools = (m.tools || []).map(t => t.name).filter(Boolean);
      li.innerHTML = `
        <div class="mcp-name">${escapeHTML(m.name)}</div>
        <div class="mcp-meta">
          <span class="muted">${escapeHTML(m.url)}</span>
          ${tools.length ? `<span class="muted"> · ${tools.join(", ")}</span>` : ""}
        </div>
        <button class="mcp-rm" title="unregister" data-name="${escapeHTML(m.name)}">×</button>`;
      li.querySelector(".mcp-rm").addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const name = ev.currentTarget.dataset.name;
        if (!confirm(`Unregister MCP ${name}?`)) return;
        await fetch("/api/mcps/unregister", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({name}),
        });
        loadMCPs();
      });
      mcpListEl.appendChild(li);
    }
  } catch (e) {
    mcpListEl.innerHTML = `<li class="empty">load failed: ${e.message}</li>`;
  }
}

document.querySelector("#new-mcp-btn")?.addEventListener("click", () => mcpModal.showModal());

if (mcpForm) {
  mcpForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(mcpForm);
    let toolsArr = [];
    const toolsRaw = fd.get("tools");
    if (toolsRaw && String(toolsRaw).trim()) {
      try { toolsArr = JSON.parse(String(toolsRaw)); }
      catch (err) { toast(`invalid tools JSON: ${err.message}`, "warn"); return; }
    }
    const body = {name: fd.get("name"), url: fd.get("url"), tools: toolsArr};
    const r = await fetch("/api/mcps/register", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      toast(`register failed: ${d.error || r.status}`, "warn");
      return;
    }
    mcpModal.close();
    mcpForm.reset();
    loadMCPs();
  });
}

// ============================================================================
// Outputs archive — list of every graph run, grouped by date, lazy-paged.
// Each run is a tiny summary in the sidebar; click opens a detail modal
// with full diagram + inputs + outputs + events.
// ============================================================================

const archiveListEl = document.querySelector("#archive-list");
const archiveFilterEl = document.querySelector("#archive-filter");
const archiveLoadMoreBtn = document.querySelector("#archive-load-more");
const archiveModal = document.querySelector("#archive-modal");

let _archiveCursor = null;       // pagination: ISO string of oldest seen
let _archiveAllRuns = [];        // accumulated across pages
let _archiveCurrentRun = null;   // full record cached for re-run/delete

function _fmtDate(iso) {
  if (!iso) return "?";
  // Extract YYYY-MM-DD from "YYYY-MM-DDTHH:MM:SS"
  return iso.slice(0, 10);
}

function _fmtTime(iso) {
  if (!iso) return "";
  return iso.slice(11, 19);
}

function _renderArchiveSidebar() {
  if (!archiveListEl) return;
  // Apply current filter.
  const q = (archiveFilterEl?.value || "").trim().toLowerCase();
  const visible = q
    ? _archiveAllRuns.filter(r => (r.graph || "").toLowerCase().includes(q))
    : _archiveAllRuns;
  if (!visible.length) {
    archiveListEl.innerHTML = `<li class="empty">${q ? "no matches" : "no runs yet"}</li>`;
    return;
  }
  // Group by date for visual collapse — rendered as flat list with date
  // headings between groups.
  archiveListEl.innerHTML = "";
  let lastDate = null;
  for (const r of visible) {
    const d = _fmtDate(r.started_at);
    if (d !== lastDate) {
      const hdr = document.createElement("li");
      hdr.className = "archive-date-head";
      hdr.textContent = d;
      archiveListEl.appendChild(hdr);
      lastDate = d;
    }
    const li = document.createElement("li");
    li.className = "archive-item";
    li.dataset.runId = r.run_id;
    const ok = r.ok ? "" : ' archive-item-failed';
    li.innerHTML = `
      <div class="archive-item-row${ok}">
        <span class="archive-item-graph">${escapeHTML(r.graph || "?")}</span>
        <span class="archive-item-time">${_fmtTime(r.started_at)}</span>
      </div>
      <div class="archive-item-meta muted">
        ${r.ok ? `${r.n_nodes || 0} nodes` : "failed"}
        ${r.wall_s != null ? ` · ${r.wall_s}s` : ""}
      </div>`;
    li.addEventListener("click", () => openArchiveRun(r.run_id));
    archiveListEl.appendChild(li);
  }
}

async function loadArchive(reset = true) {
  if (!archiveListEl) return;
  if (reset) {
    _archiveCursor = null;
    _archiveAllRuns = [];
    archiveListEl.innerHTML = `<li class="empty">loading…</li>`;
  }
  const params = new URLSearchParams();
  if (_archiveCursor) params.set("since", _archiveCursor);
  params.set("limit", "50");
  try {
    const r = await fetch(`/api/graph-runs?${params}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    _archiveAllRuns.push(...(d.runs || []));
    _archiveCursor = d.next_cursor;
    _renderArchiveSidebar();
    if (archiveLoadMoreBtn) {
      archiveLoadMoreBtn.hidden = !d.next_cursor;
    }
  } catch (e) {
    archiveListEl.innerHTML = `<li class="empty">load failed: ${escapeHTML(e.message)}</li>`;
  }
}

archiveFilterEl?.addEventListener("input", () => _renderArchiveSidebar());
archiveLoadMoreBtn?.addEventListener("click", () => loadArchive(false));
document.querySelector("#archive-refresh-btn")?.addEventListener("click", () => loadArchive(true));

// ---- Archive detail rendering helpers --------------------------------
// Outputs come back as `{node_name: {output_name: value, ...}}` from the
// graph executor. Render each node as a card; render each output by its
// declared tag (text→markdown, JSON→pretty json, list→bulleted, etc.) so
// the structure that the user defined in the builder shows up here too.

function _archiveTagLabel(tag) {
  return OUTPUT_TAG_LABEL[tag] || tag || "?";
}

// Best-effort detect a tag from a value when the graph spec isn't loaded.
function _inferTag(v) {
  if (v == null) return "t";
  if (typeof v === "boolean") return "b";
  if (typeof v === "number") return "n";
  if (Array.isArray(v)) return "l";
  if (typeof v === "object") {
    // `kv` heuristic: shallow object of string-ish values; otherwise `j`.
    const vals = Object.values(v);
    return vals.length > 0 && vals.every(x => x == null || typeof x !== "object")
      ? "kv" : "j";
  }
  return "t";
}

function _formatOutputValue(value, tag) {
  // Returns an HTML string. All user-content paths go through escapeHTML
  // or renderMarkdownSafe (which DOMPurifies).
  if (value == null) {
    return `<span class="muted">(null)</span>`;
  }
  if (tag === "t") {
    // Text → markdown. Long content gets a collapsible.
    const s = String(value);
    if (s.length > 1200) {
      return `<details class="archive-long"><summary class="muted">${s.length} chars · click to expand</summary>` +
             `<div class="markdown-body">${renderMarkdownSafe(s)}</div></details>`;
    }
    return `<div class="markdown-body">${renderMarkdownSafe(s)}</div>`;
  }
  if (tag === "n") {
    return `<code class="archive-num">${escapeHTML(String(value))}</code>`;
  }
  if (tag === "b") {
    const v = !!value;
    return `<span class="archive-bool ${v ? "yes" : "no"}">${v ? "yes" : "no"}</span>`;
  }
  if (tag === "l") {
    if (!Array.isArray(value)) {
      return `<pre class="archive-json">${escapeHTML(JSON.stringify(value, null, 2))}</pre>`;
    }
    if (!value.length) return `<span class="muted">(empty list)</span>`;
    const items = value.map(x => {
      if (x == null) return `<li class="muted">(null)</li>`;
      if (typeof x === "object") {
        return `<li><pre class="archive-json">${escapeHTML(JSON.stringify(x, null, 2))}</pre></li>`;
      }
      return `<li>${escapeHTML(String(x))}</li>`;
    }).join("");
    return `<ul class="archive-list-items">${items}</ul>`;
  }
  if (tag === "kv") {
    if (!value || typeof value !== "object") {
      return `<pre class="archive-json">${escapeHTML(JSON.stringify(value, null, 2))}</pre>`;
    }
    const rows = Object.entries(value).map(([k, v]) => {
      const valHtml = (v != null && typeof v === "object")
        ? `<pre class="archive-json archive-kv-nested">${escapeHTML(JSON.stringify(v, null, 2))}</pre>`
        : escapeHTML(String(v));
      return `<tr><th>${escapeHTML(k)}</th><td>${valHtml}</td></tr>`;
    }).join("");
    return `<table class="archive-kv">${rows}</table>`;
  }
  // tag === "j" or unknown → pretty JSON
  let pretty;
  try {
    pretty = JSON.stringify(value, null, 2);
  } catch {
    pretty = String(value);
  }
  return `<pre class="archive-json">${escapeHTML(pretty)}</pre>`;
}

function _outputTagsForNode(graphSpec, nodeName) {
  // Return {output_name: tag} from the graph spec if available.
  if (!graphSpec || !Array.isArray(graphSpec.nodes)) return {};
  const node = graphSpec.nodes.find(n => n.name === nodeName);
  if (!node || !Array.isArray(node.outputs)) return {};
  const m = {};
  for (const o of node.outputs) {
    if (Array.isArray(o)) m[o[0]] = o[1];
    else if (typeof o === "string" && o.includes(":")) {
      const [n, t] = o.split(":");
      m[n] = t;
    } else m[String(o)] = "t";
  }
  return m;
}

function _renderArchiveInputsInto(target, inputs) {
  if (!inputs || !Object.keys(inputs).length) {
    target.innerHTML = `<div class="muted">(no inputs)</div>`;
    return;
  }
  // Inputs from the run modal are typically text; render compact key-row.
  const rows = Object.entries(inputs).map(([k, v]) => {
    const display = (typeof v === "object")
      ? `<pre class="archive-json">${escapeHTML(JSON.stringify(v, null, 2))}</pre>`
      : escapeHTML(String(v));
    return `<tr><th>${escapeHTML(k)}</th><td>${display}</td></tr>`;
  }).join("");
  target.innerHTML = `<table class="archive-kv archive-inputs-table">${rows}</table>`;
}

function _renderArchiveOutputsInto(target, outputs, graphSpec) {
  // outputs is `{node_name: {output_name: value}}` (or sometimes flat).
  if (!outputs || !Object.keys(outputs).length) {
    target.innerHTML = `<div class="muted">(no outputs)</div>`;
    return;
  }
  // Detect shape: graph executor returns nested {node: {field: val}}.
  // If anyone passes a flat dict, wrap it in a "results" pseudo-node so
  // the rendering path is uniform.
  const looksNested = Object.values(outputs).every(v =>
    v != null && typeof v === "object" && !Array.isArray(v));
  const nested = looksNested ? outputs : {results: outputs};

  // Order nodes by graphSpec order if available; else insertion order.
  let nodeNames = Object.keys(nested);
  if (graphSpec && Array.isArray(graphSpec.nodes)) {
    const specOrder = graphSpec.nodes.map(n => n.name);
    nodeNames = [
      ...specOrder.filter(n => n in nested),
      ...nodeNames.filter(n => !specOrder.includes(n)),
    ];
  }

  const cards = nodeNames.map(nodeName => {
    const nodeOutputs = nested[nodeName] || {};
    const tagMap = _outputTagsForNode(graphSpec, nodeName);
    const role = (graphSpec?.nodes?.find(n => n.name === nodeName) || {}).role || "";
    const outputRows = Object.entries(nodeOutputs).map(([oname, value]) => {
      const tag = tagMap[oname] || _inferTag(value);
      return `
        <div class="archive-output">
          <div class="archive-output-head">
            <span class="archive-output-name">${escapeHTML(oname)}</span>
            <span class="archive-output-tag" title="output type">${escapeHTML(_archiveTagLabel(tag))}</span>
          </div>
          <div class="archive-output-value">${_formatOutputValue(value, tag)}</div>
        </div>`;
    }).join("");
    return `
      <section class="archive-node-card">
        <header class="archive-node-head">
          <span class="archive-node-name">${escapeHTML(nodeName)}</span>
          ${role ? `<span class="archive-node-role muted">${escapeHTML(role)}</span>` : ""}
          <span class="archive-node-count muted">${Object.keys(nodeOutputs).length} output${Object.keys(nodeOutputs).length === 1 ? "" : "s"}</span>
        </header>
        <div class="archive-node-body">${outputRows || `<div class="muted">(no outputs)</div>`}</div>
      </section>`;
  }).join("");

  target.innerHTML = cards;
  // Enrich any markdown bodies (highlight code, render math, etc).
  for (const md of target.querySelectorAll(".markdown-body")) {
    enrichRenderedBlock(md).catch(() => {});
  }
}

function _renderArchiveEventsInto(target, events) {
  if (!events || !events.length) {
    target.innerHTML = `<div class="muted">(no events)</div>`;
    return;
  }
  // Group consecutive events from the same node so the timeline is
  // readable instead of a wall of `[event] node …` lines.
  const rows = events.map(e => {
    const kind = e.kind || "event";
    const node = e.node || "";
    const msg = e.message || "";
    const dur = (e.duration_ms != null) ? `${e.duration_ms}ms` : "";
    return `
      <tr class="archive-event-row archive-event-${escapeHTML(kind)}">
        <td class="archive-event-kind">${escapeHTML(kind)}</td>
        <td class="archive-event-node">${escapeHTML(node)}</td>
        <td class="archive-event-msg">${escapeHTML(String(msg).slice(0, 400))}</td>
        <td class="archive-event-dur muted">${escapeHTML(dur)}</td>
      </tr>`;
  }).join("");
  target.innerHTML = `<table class="archive-events-table">${rows}</table>`;
}

async function openArchiveRun(runId) {
  if (!archiveModal) return;
  document.querySelector("#archive-title").textContent = `Run · loading…`;
  document.querySelector("#archive-meta").textContent = "";
  document.querySelector("#archive-inputs").innerHTML = "";
  document.querySelector("#archive-outputs").innerHTML = "";
  document.querySelector("#archive-events").innerHTML = "";
  document.querySelector("#archive-events-count").textContent = "";
  document.querySelector("#archive-outputs-count").textContent = "";
  document.querySelector("#archive-diagram").innerHTML = "";
  archiveModal.showModal();
  try {
    const r = await fetch(`/api/graph-runs/${encodeURIComponent(runId)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const rec = await r.json();
    _archiveCurrentRun = rec;
    document.querySelector("#archive-title").textContent =
      `Run · ${rec.graph} · ${rec.run_id}`;
    const okBadge = rec.ok ? `<span class="ok-badge">ok</span>`
                            : `<span class="bad-badge">failed</span>`;
    document.querySelector("#archive-meta").innerHTML = `
      ${okBadge}
      <span><strong>graph:</strong> ${escapeHTML(rec.graph || "?")}</span>
      <span><strong>started:</strong> ${escapeHTML(rec.started_at || "?")}</span>
      ${rec.wall_s != null ? `<span><strong>wall:</strong> ${rec.wall_s}s</span>` : ""}
      ${rec.error ? `<div class="archive-error">${escapeHTML(rec.error)}</div>` : ""}
    `;
    const matchingGraph = (typeof _graphs !== "undefined")
      ? _graphs.find(g => g.name === rec.graph) : null;
    if (matchingGraph) {
      renderGraphDiagram(document.querySelector("#archive-diagram"),
                          matchingGraph);
    } else {
      document.querySelector("#archive-diagram").innerHTML =
        `<div class="muted">graph file no longer in examples/ — diagram unavailable</div>`;
    }
    _renderArchiveInputsInto(
      document.querySelector("#archive-inputs"), rec.inputs);
    _renderArchiveOutputsInto(
      document.querySelector("#archive-outputs"), rec.outputs, matchingGraph);
    const nNodes = rec.outputs ? Object.keys(rec.outputs).length : 0;
    document.querySelector("#archive-outputs-count").textContent =
      nNodes ? `(${nNodes} node${nNodes === 1 ? "" : "s"})` : "";
    const events = rec.events || [];
    document.querySelector("#archive-events-count").textContent =
      events.length ? `(${events.length})` : "";
    _renderArchiveEventsInto(
      document.querySelector("#archive-events"), events);
  } catch (e) {
    document.querySelector("#archive-meta").innerHTML =
      `<div class="archive-error">load failed: ${escapeHTML(e.message)}</div>`;
  }
}

document.querySelector("#archive-rerun-btn")?.addEventListener("click", () => {
  const rec = _archiveCurrentRun;
  if (!rec) return;
  archiveModal.close();
  // Find the graph in the loaded list and open the run modal pre-filled
  // with the archived inputs.
  const g = _graphs?.find(x => x.name === rec.graph);
  if (!g) {
    toast(`graph "${rec.graph}" no longer exists`, "warn");
    return;
  }
  // Stash inputs into _runs so openGraphRun seeds them.
  _setRun(g.name, {inputs: rec.inputs || {}, status: "ready", events: []});
  openGraphRun(g);
});

document.querySelector("#archive-delete-btn")?.addEventListener("click", async () => {
  const rec = _archiveCurrentRun;
  if (!rec) return;
  if (!confirm(`Delete archived run ${rec.run_id}?`)) return;
  try {
    const r = await fetch(`/api/graph-runs/${encodeURIComponent(rec.run_id)}`,
                          {method: "DELETE"});
    if (!r.ok) {
      toast(`delete failed: ${r.status}`, "warn");
      return;
    }
    archiveModal.close();
    loadArchive(true);
  } catch (e) {
    toast(`delete failed: ${e.message}`, "warn");
  }
});

// Auto-refresh the panels when the agents tab is opened.
const _agentsObserver = new MutationObserver(() => {
  if (document.body.dataset.tab === "agents") {
    loadGraphs();
    loadMCPs();
    loadArchive(true);
  }
});
_agentsObserver.observe(document.body, {attributes: true, attributeFilter: ["data-tab"]});
if (document.body.dataset.tab === "agents") {
  loadGraphs();
  loadMCPs();
  loadArchive(true);
}

// Close modals via [data-close] buttons within the new dialogs.
for (const dlg of [graphModal, graphRunModal, mcpModal, previewModal, buildModal, archiveModal]) {
  if (!dlg) continue;
  for (const closeBtn of dlg.querySelectorAll("[data-close]")) {
    closeBtn.addEventListener("click", () => dlg.close());
  }
}

console.info("qwen-ui app loaded");
