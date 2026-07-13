// ── State ──────────────────────────────────────────────────────────────────
const SLOT_COUNT = 5;
let configData = null;
let abortController = null;

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const statusBadge = $("#statusBadge");
const openrouterBadge = $("#openrouterBadge");
const ollamaBadge = $("#ollamaBadge");
const settingsToggle = $("#settingsToggle");
const settingsBody = $("#settingsBody");
const openrouterKey = $("#openrouterKey");
const openrouterKeyHint = $("#openrouterKeyHint");
const ollamaUrl = $("#ollamaUrl");
const slotTimeoutInput = $("#slotTimeout");
const saveSettingsBtn = $("#saveSettings");
const saveMsg = $("#saveMsg");
const toggleKeyVis = $("#toggleKeyVis");
const slotsGrid = $("#slotsGrid");
const promptInput = $("#promptInput");
const systemPromptInput = $("#systemPrompt");
const sendBtn = $("#sendBtn");
const streamBtn = $("#streamBtn");
const synthBtn = $("#synthBtn");
const responsesGrid = $("#responsesGrid");
const synthPanel = $("#synthPanel");
const synthModelName = $("#synthModelName");
const synthLatency = $("#synthLatency");
const synthContent = $("#synthContent");
const synthSlot = $("#synthSlot");
const synthModeToggle = $("#synthModeToggle");
const synthSystemPrompt = $("#synthSystemPrompt");
const privateApiKey = $("#privateApiKey");
const privateApiKeyHint = $("#privateApiKeyHint");
const toggleApiKeyVisBtn = $("#toggleApiKeyVis");
const synthModeBadge = $("#synthModeBadge");
const apiKeyBadge = $("#apiKeyBadge");
const toolsContext7Toggle = $("#toolsContext7Toggle");
const context7Key = $("#context7Key");
const context7KeyHint = $("#context7KeyHint");
const toolsWebToggle = $("#toolsWebToggle");
const webSearchBackend = $("#webSearchBackend");
const searxngUrl = $("#searxngUrl");
const braveKey = $("#braveKey");
const braveKeyHint = $("#braveKeyHint");
const tavilyKey = $("#tavilyKey");
const tavilyKeyHint = $("#tavilyKeyHint");
const toolsMaxIterations = $("#toolsMaxIterations");
const synthTools = $("#synthTools");
const synthWarning = $("#synthWarning");

// ── API fetch (adds the private API key when the server requires one) ──────
function _storedApiKey() {
  return localStorage.getItem("fusionApiKey") || "";
}

async function apiFetch(url, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const key = _storedApiKey();
  if (key) headers["Authorization"] = "Bearer " + key;
  let resp = await fetch(url, { ...opts, headers });
  if (resp.status === 401) {
    const entered = prompt("This Fusion App server requires its private API key:");
    if (entered && entered.trim()) {
      localStorage.setItem("fusionApiKey", entered.trim());
      headers["Authorization"] = "Bearer " + entered.trim();
      resp = await fetch(url, { ...opts, headers });
    }
  }
  return resp;
}

// ── Tools helpers ──────────────────────────────────────────────────────────
function _updateBackendRows() {
  const backend = webSearchBackend.value;
  $("#searxngUrlRow").style.display = backend === "searxng" ? "" : "none";
  $("#braveKeyRow").style.display = backend === "brave" ? "" : "none";
  $("#tavilyKeyRow").style.display = backend === "tavily" ? "" : "none";
}

function _setKeyHint(hintEl, isSet) {
  hintEl.textContent = isSet ? "✓ Key saved" : "";
  hintEl.className = isSet ? "key-hint set" : "key-hint";
}

// Build a collapsible tool-call trace element. Tool names/args/results come
// from external sources — DOM APIs only, never innerHTML interpolation.
function buildToolTraceEl(entries, { open = false } = {}) {
  const details = document.createElement("details");
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = `🔧 Tool calls (${entries.length})`;
  details.appendChild(summary);
  for (const t of entries) {
    const row = document.createElement("div");
    row.className = "tool-row";
    const head = document.createElement("div");
    head.className = "tool-row-head";
    const name = document.createElement("span");
    name.className = "tool-name";
    name.textContent = t.name || "?";
    const args = document.createElement("span");
    args.className = "tool-args";
    args.textContent = JSON.stringify(t.arguments ?? {});
    const ms = document.createElement("span");
    ms.className = "tool-ms";
    ms.textContent = t.ms != null ? `${t.ms}ms` : "";
    head.append(name, args, ms);
    const result = document.createElement("div");
    result.className = "tool-result";
    result.textContent = t.result_preview || "";
    row.append(head, result);
    details.appendChild(row);
  }
  return details;
}

function renderToolTrace(entries, { open = false } = {}) {
  if (!entries || !entries.length) {
    synthTools.style.display = "none";
    synthTools.textContent = "";
    return;
  }
  synthTools.style.display = "block";
  synthTools.textContent = "";
  synthTools.appendChild(buildToolTraceEl(entries, { open }));
}

// Per-slot variants (draft slots can tool-call on every path too)
function renderSlotToolTrace(panel, entries, { open = false } = {}) {
  const box = panel.querySelector(".slot-tool-trace");
  if (!box) return;
  if (!entries || !entries.length) {
    box.style.display = "none";
    box.textContent = "";
    return;
  }
  box.style.display = "block";
  box.textContent = "";
  box.appendChild(buildToolTraceEl(entries, { open }));
}

function showSlotWarning(panel, text) {
  const badge = panel.querySelector(".slot-warning");
  if (!badge) return;
  if (text) {
    badge.textContent = `⚠ ${text}`;
    badge.style.display = "";
  } else {
    badge.style.display = "none";
  }
}

function clearAllSlotToolUI() {
  document.querySelectorAll(".response-panel").forEach((panel) => {
    renderSlotToolTrace(panel, []);
    showSlotWarning(panel, null);
  });
}

function showSynthWarning(text) {
  if (text) {
    synthWarning.textContent = `⚠ ${text}`;
    synthWarning.style.display = "";
  } else {
    synthWarning.style.display = "none";
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function _updateSynthModeBadge() {
  const isActive = synthModeToggle.checked && parseInt(synthSlot.value) >= 0;
  if (isActive) {
    synthModeBadge.textContent = "⚡ Synth: ON";
    synthModeBadge.className = "key-badge synth-active";
    synthModeBadge.style.display = "";
  } else {
    synthModeBadge.style.display = "none";
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  buildSlotCards();
  buildResponsePanels();
  await loadConfig();

  // If no OpenRouter key is set, auto-expand settings on first visit
  if (!configData?.openrouter_key_set) {
    settingsBody.classList.add("open");
  }

  settingsToggle.addEventListener("click", () => {
    settingsBody.classList.toggle("open");
  });

  toggleKeyVis.addEventListener("click", () => {
    const t = openrouterKey;
    t.type = t.type === "password" ? "text" : "password";
  });

  // Click hint → focus the key input
  openrouterKeyHint.addEventListener("click", () => {
    settingsBody.classList.add("open");
    openrouterKey.focus();
  });

  // Private API key hint click → focus
  privateApiKeyHint.addEventListener("click", () => {
    settingsBody.classList.add("open");
    privateApiKey.focus();
  });

  // Toggle private API key visibility
  toggleApiKeyVisBtn.addEventListener("click", () => {
    const t = privateApiKey;
    t.type = t.type === "password" ? "text" : "password";
  });

  // Synth mode toggle updates badge immediately
  synthModeToggle.addEventListener("change", _updateSynthModeBadge);
  synthSlot.addEventListener("change", _updateSynthModeBadge);

  webSearchBackend.addEventListener("change", _updateBackendRows);
  saveSettingsBtn.addEventListener("click", saveConfig);
  sendBtn.addEventListener("click", () => sendPrompt(false));
  streamBtn.addEventListener("click", () => sendPrompt(true));
  synthBtn.addEventListener("click", sendSynth);

  // Enter to send (Shift+Enter for newline)
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(false);
    }
  });
});

// ── Load config ────────────────────────────────────────────────────────────
async function loadConfig() {
  statusBadge.textContent = "● Loading…";
  statusBadge.className = "status-badge";
  try {
    const resp = await apiFetch("/api/config");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    configData = await resp.json();

    // API key status
    if (configData.openrouter_key_set) {
      openrouterBadge.textContent = "🔑 OR: set";
      openrouterBadge.className = "key-badge set";
      openrouterKey.placeholder = configData.openrouter_key + " (saved)";
      openrouterKeyHint.textContent = "✓ Key saved";
      openrouterKeyHint.className = "key-hint set";
    } else {
      openrouterBadge.textContent = "🔑 OR: not set";
      openrouterBadge.className = "key-badge unset";
      openrouterKey.placeholder = "sk-or-…";
      openrouterKeyHint.textContent = "Not set — Click here ↑";
      openrouterKeyHint.className = "key-hint";
    }
    // Don't fill the password field — user must type to change
    openrouterKey.value = "";

    // Synth slot selector
    synthSlot.value = String(configData.synth_slot ?? -1);

    // Synth mode toggle
    synthModeToggle.checked = configData.synth_mode || false;
    _updateSynthModeBadge();

    // Synth system prompt
    if (configData.synth_system_prompt) {
      synthSystemPrompt.value = configData.synth_system_prompt;
    }

    // Private API key
    if (configData.private_api_key_set) {
      apiKeyBadge.textContent = "🔐 API: set";
      apiKeyBadge.className = "key-badge set";
      privateApiKey.placeholder = "•••••••• (saved)";
      privateApiKeyHint.textContent = "✓ Private key saved";
      privateApiKeyHint.className = "key-hint set";
    } else {
      apiKeyBadge.textContent = "🔐 API: not set";
      apiKeyBadge.className = "key-badge unset";
      privateApiKey.placeholder = "sk-…";
      privateApiKeyHint.textContent = "Not set — set for agent auth";
      privateApiKeyHint.className = "key-hint";
    }
    privateApiKey.value = "";

    // Ollama badge
    ollamaUrl.value = configData.ollama_base_url;
    ollamaBadge.textContent = `🖥 Ollama: ${configData.ollama_base_url}`;
    ollamaBadge.className = "key-badge set";

    // Global slot timeout
    slotTimeoutInput.value = configData.slot_timeout ?? "";

    configData.slots.forEach((slot, i) => {
      const card = document.querySelector(`.slot-card[data-slot="${i}"]`);
      if (!card) return;
      const toggle = card.querySelector(".toggle-switch input");
      const provider = card.querySelector(".slot-provider");
      const modelInput = card.querySelector(".slot-model-input");
      const urlOverride = card.querySelector(".slot-url-override");

      toggle.checked = slot.enabled;
      card.classList.toggle("enabled", slot.enabled);
      provider.value = slot.provider;
      modelInput.value = slot.model;
      if (urlOverride) {
        urlOverride.value = slot.base_url_override || "";
      }
      const slotTimeout = card.querySelector(".slot-timeout");
      if (slotTimeout) {
        slotTimeout.value = slot.timeout ?? "";
      }
      const slotTools = card.querySelector(".slot-tools");
      if (slotTools) {
        slotTools.checked = !!slot.tools_enabled;
      }
      // Auto-expand Advanced section if any override is set
      if (slot.base_url_override || slot.timeout || slot.tools_enabled) {
        const advBody = card.querySelector(".slot-advanced-body");
        const advToggle = card.querySelector(".slot-advanced-toggle");
        if (advBody) advBody.style.display = "block";
        if (advToggle) advToggle.textContent = "▾ Advanced";
      }

      // Update response panel header
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) {
        panel.classList.toggle("disabled", !slot.enabled);
        panel.querySelector(".model-name").textContent = slot.model || "—";
      }
    });

    // Tools settings
    const tools = configData.tools || {};
    toolsContext7Toggle.checked = tools.context7_enabled !== false;
    toolsWebToggle.checked = tools.web_enabled !== false;
    webSearchBackend.value = tools.web_search_backend || "duckduckgo";
    searxngUrl.value = tools.searxng_base_url || "";
    toolsMaxIterations.value = tools.max_iterations ?? "";
    _setKeyHint(context7KeyHint, !!tools.context7_api_key_set);
    _setKeyHint(braveKeyHint, !!tools.brave_api_key_set);
    _setKeyHint(tavilyKeyHint, !!tools.tavily_api_key_set);
    context7Key.value = "";
    braveKey.value = "";
    tavilyKey.value = "";
    _updateBackendRows();

    updateAllDisabledStates();
    statusBadge.textContent = "● Online";
    statusBadge.className = "status-badge online";
  } catch (e) {
    statusBadge.textContent = "● Offline";
    statusBadge.className = "status-badge offline";
  }
}

// ── Save config ────────────────────────────────────────────────────────────
const CONFIG_FIELD_LABELS = {
  base_url_override: "URL override",
  ollama_base_url: "Ollama URL",
  openrouter_key: "OpenRouter key",
  private_api_key: "Private API key",
  synth_slot: "Synth slot",
  synth_system_prompt: "Synth system prompt",
  slot_timeout: "Slot timeout",
  timeout: "timeout",
};

// Turn a 422 response into a short human-readable message,
// e.g. "Slot 0 URL override: must be an http(s) URL with a host"
async function formatConfigError(resp) {
  const text = await resp.text();
  try {
    const detail = JSON.parse(text).detail;
    if (Array.isArray(detail)) {
      return detail.map((err) => {
        const loc = err.loc.filter((p) => p !== "body");
        let field = loc.map((p) => CONFIG_FIELD_LABELS[p] || p).join(" ");
        if (loc[0] === "slots" && typeof loc[1] === "number") {
          field = `Slot ${loc[1]} ${CONFIG_FIELD_LABELS[loc[2]] || loc[2] || ""}`.trim();
        }
        return `${field}: ${err.msg.replace(/^Value error, /, "")}`;
      }).join("; ");
    }
    if (typeof detail === "string") return detail.replace(/\s+/g, " ");
  } catch (_) {
    // not JSON — fall through to raw text
  }
  return text;
}

async function saveConfig() {
  saveMsg.textContent = "Saving…";
  const slots = [];
  for (let i = 0; i < SLOT_COUNT; i++) {
    const card = document.querySelector(`.slot-card[data-slot="${i}"]`);
    const toggle = card.querySelector(".toggle-switch input");
    const provider = card.querySelector(".slot-provider");
    const modelInput = card.querySelector(".slot-model-input");
    const urlOverride = card.querySelector(".slot-url-override");
    const slotData = {
      provider: provider.value,
      model: modelInput.value,
      enabled: toggle.checked,
    };
    const urlVal = urlOverride ? urlOverride.value.trim() : "";
    if (urlVal) {
      slotData.base_url_override = urlVal;
    } else {
      slotData.base_url_override = null;
    }
    const timeoutEl = card.querySelector(".slot-timeout");
    const timeoutVal = timeoutEl ? timeoutEl.value.trim() : "";
    slotData.timeout = timeoutVal ? parseFloat(timeoutVal) : null;
    const toolsEl = card.querySelector(".slot-tools");
    slotData.tools_enabled = toolsEl ? toolsEl.checked : false;
    slots.push(slotData);
  }

  // Build body — only include key if user typed something new
  const body = {
    ollama_base_url: ollamaUrl.value,
    synth_slot: parseInt(synthSlot.value, 10),
    synth_mode: synthModeToggle.checked,
    slots,
  };

  const globalTimeout = slotTimeoutInput.value.trim();
  if (globalTimeout) {
    body.slot_timeout = parseFloat(globalTimeout);
  }

  // Include synth system prompt if user modified it
  const promptVal = synthSystemPrompt.value.trim();
  if (promptVal) {
    body.synth_system_prompt = promptVal;
  }

  if (openrouterKey.value.trim()) {
    body.openrouter_key = openrouterKey.value.trim();
  }
  if (privateApiKey.value.trim()) {
    body.private_api_key = privateApiKey.value.trim();
  }

  // Tools: partial update — secret fields only when the user typed a new
  // value, so saved keys are never clobbered by empty/masked inputs.
  body.tools = {
    context7_enabled: toolsContext7Toggle.checked,
    web_enabled: toolsWebToggle.checked,
    web_search_backend: webSearchBackend.value,
    searxng_base_url: searxngUrl.value.trim(),
  };
  const maxIter = toolsMaxIterations.value.trim();
  if (maxIter) {
    body.tools.max_iterations = parseInt(maxIter, 10);
  }
  if (context7Key.value.trim()) {
    body.tools.context7_api_key = context7Key.value.trim();
  }
  if (braveKey.value.trim()) {
    body.tools.brave_api_key = braveKey.value.trim();
  }
  if (tavilyKey.value.trim()) {
    body.tools.tavily_api_key = tavilyKey.value.trim();
  }

  try {
    const resp = await apiFetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await formatConfigError(resp));
    saveMsg.textContent = "✓ Saved";
    // Remember the new private key so subsequent requests stay authorized
    if (privateApiKey.value.trim()) {
      localStorage.setItem("fusionApiKey", privateApiKey.value.trim());
    }
    openrouterKey.value = "";
    privateApiKey.value = ""; // clear so user knows it's saved
    await loadConfig(); // reload to refresh badges
    setTimeout(() => { saveMsg.textContent = ""; }, 2000);
  } catch (e) {
    saveMsg.textContent = "✗ " + e.message;
  }
}

// ── Build slot cards ───────────────────────────────────────────────────────
function buildSlotCards() {
  slotsGrid.innerHTML = "";
  for (let i = 0; i < SLOT_COUNT; i++) {
    const div = document.createElement("div");
    div.className = "slot-card";
    div.dataset.slot = i;

    div.innerHTML = `
      <div class="slot-header">
        <label>Slot ${i}</label>
        <label class="toggle-switch">
          <input type="checkbox" ${i < 2 ? "checked" : ""} />
          <span class="toggle-slider"></span>
        </label>
      </div>
      <select class="slot-provider">
        <option value="openrouter" ${i === 0 ? "selected" : ""}>OpenRouter</option>
        <option value="ollama" ${i === 1 ? "selected" : ""}>Ollama</option>
      </select>
      <input class="slot-model-input" type="text" placeholder="e.g. gpt-4o, llama3.2"
             value="${i === 0 ? "openai/gpt-4o" : i === 1 ? "llama3.2" : ""}" spellcheck="false" />
      <div class="slot-advanced">
        <button class="slot-advanced-toggle" type="button">▸ Advanced</button>
        <div class="slot-advanced-body" style="display:none;">
          <input class="slot-url-override" type="text" placeholder="e.g. http://192.168.1.10:11434" spellcheck="false" />
          <span class="slot-url-hint">Only needed for Ollama on a different machine. Leave blank to use the global URL from Settings.</span>
          <input class="slot-timeout" type="number" min="1" step="1" placeholder="Timeout (seconds)" />
          <span class="slot-url-hint">Blank = global Slot Timeout from Settings. Ollama requests are hard-capped at 1200s by the HTTP client.</span>
          <label class="slot-tools-label"><input class="slot-tools" type="checkbox" /> 🔧 Tools (fact-checking)</label>
          <span class="slot-url-hint">Lets this slot verify claims via docs lookup, web search, and URL checks. Intended for the synth slot; adds up to Max Tool Rounds × timeout.</span>
        </div>
      </div>
      <button class="model-refresh-btn" data-slot="${i}">↻ Fetch models</button>
    `;

    // Toggle handler
    const toggle = div.querySelector(".toggle-switch input");
    toggle.addEventListener("change", () => {
      div.classList.toggle("enabled", toggle.checked);
      updateAllDisabledStates();
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) panel.classList.toggle("disabled", !toggle.checked);
    });

    // Provider change → update model list
    const provider = div.querySelector(".slot-provider");
    provider.addEventListener("change", () => {
      const modelInput = div.querySelector(".slot-model-input");
      modelInput.value = "";
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) panel.querySelector(".model-name").textContent = "—";
    });

    // Model input change → update response header
    const modelInput = div.querySelector(".slot-model-input");
    modelInput.addEventListener("change", () => {
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) panel.querySelector(".model-name").textContent = modelInput.value || "—";
    });
    modelInput.addEventListener("input", () => {
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) panel.querySelector(".model-name").textContent = modelInput.value || "—";
    });

    // Refresh models button
    const refreshBtn = div.querySelector(".model-refresh-btn");
    refreshBtn.addEventListener("click", () => refreshModels(i));

    // Advanced toggle (URL override)
    const advToggle = div.querySelector(".slot-advanced-toggle");
    const advBody = div.querySelector(".slot-advanced-body");
    advToggle.addEventListener("click", () => {
      const isOpen = advBody.style.display !== "none";
      advBody.style.display = isOpen ? "none" : "block";
      advToggle.textContent = isOpen ? "▸ Advanced" : "▾ Advanced";
    });

    slotsGrid.appendChild(div);
  }
}

// ── Build response panels ──────────────────────────────────────────────────
function buildResponsePanels() {
  responsesGrid.innerHTML = "";
  for (let i = 0; i < SLOT_COUNT; i++) {
    const div = document.createElement("div");
    div.className = "response-panel";
    div.dataset.slot = i;

    div.innerHTML = `
      <div class="response-header">
        <span class="slot-label">Slot ${i}</span>
        <span class="model-name">—</span>
        <span class="latency"></span>
        <span class="warning-badge slot-warning" style="display:none;"></span>
      </div>
      <div class="slot-tool-trace" style="display:none;"></div>
      <div class="response-content">
        <span class="placeholder">Awaiting prompt…</span>
      </div>
    `;

    responsesGrid.appendChild(div);
  }
}

// ── Refresh models for a slot ──────────────────────────────────────────────
async function refreshModels(slotIndex) {
  const card = document.querySelector(`.slot-card[data-slot="${slotIndex}"]`);
  const provider = card.querySelector(".slot-provider").value;
  const refreshBtn = card.querySelector(".model-refresh-btn");
  const modelInput = card.querySelector(".slot-model-input");

  refreshBtn.textContent = "↻ Loading…";
  refreshBtn.disabled = true;

  try {
    const urlOverride = card.querySelector(".slot-url-override");
    const overrideVal = urlOverride ? urlOverride.value.trim() : "";
    let modelsUrl = `/api/models/${provider}`;
    if (provider === "ollama" && overrideVal) {
      modelsUrl += `?base_url=${encodeURIComponent(overrideVal)}`;
    }
    const resp = await apiFetch(modelsUrl);
    const data = await resp.json();

    if (data.error) {
      refreshBtn.textContent = "✗ " + data.error;
      setTimeout(() => { refreshBtn.textContent = "↻ Fetch models"; refreshBtn.disabled = false; }, 2000);
      return;
    }

    // Build model list and show in dropdown via datalist
    const models = provider === "openrouter" ? data.models.map(m => m.id) : data.models.map(m => m.name);
    if (models.length === 0) {
      refreshBtn.textContent = "✗ No models found";
    } else {
      refreshBtn.textContent = `✓ ${models.length} models`;
      // Show a simple picker: reuse the select if one already exists.
      // Model IDs come from external servers — never interpolate them
      // into HTML; build options via DOM APIs.
      const currentVal = modelInput.value;
      let select = card.querySelector(".slot-model-select");
      if (!select) {
        select = document.createElement("select");
        select.className = "slot-model-select";
        select.style.cssText = "width:100%;padding:6px 8px;margin-bottom:6px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);font-size:12px;";
        select.addEventListener("change", () => {
          modelInput.value = select.value;
          const panel = document.querySelector(`.response-panel[data-slot="${slotIndex}"]`);
          if (panel) panel.querySelector(".model-name").textContent = select.value || "—";
        });
        modelInput.parentNode.insertBefore(select, modelInput);
        modelInput.style.display = "none";
      }
      select.textContent = "";
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === currentVal) opt.selected = true;
        select.appendChild(opt);
      }
    }
  } catch (e) {
    refreshBtn.textContent = "✗ Error";
  }

  setTimeout(() => {
    if (!refreshBtn.disabled) return;
    refreshBtn.textContent = "↻ Fetch models";
    refreshBtn.disabled = false;
  }, 3000);
}

// ── Send prompt ────────────────────────────────────────────────────────────
async function sendPrompt(streaming) {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  // Clear previous responses
  for (let i = 0; i < SLOT_COUNT; i++) {
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    panel.querySelector(".response-content").innerHTML = "";
    panel.querySelector(".latency").textContent = "";
  }

  // Clear synth panel
  synthPanel.style.display = "none";
  synthContent.innerHTML = "";

  // Cancel any prior streaming
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  if (streaming) {
    sendStreaming(prompt);
  } else {
    sendSingleShot(prompt);
  }
}

// Show a waiting placeholder in every enabled panel so a slow request
// doesn't look like a hang
function markPanelsWaiting() {
  for (let i = 0; i < SLOT_COUNT; i++) {
    const card = document.querySelector(`.slot-card[data-slot="${i}"]`);
    const toggle = card?.querySelector(".toggle-switch input");
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    if (toggle?.checked && panel) {
      panel.querySelector(".response-content").innerHTML =
        '<span class="placeholder">⏳ Waiting for response…</span>';
    }
  }
}

// ── Single-shot (parallel) ─────────────────────────────────────────────────
async function sendSingleShot(prompt) {
  sendBtn.disabled = true;
  sendBtn.textContent = "◉ Sending…";
  streamBtn.disabled = true;
  markPanelsWaiting();

  const body = {
    prompt,
    system_prompt: systemPromptInput.value || null,
  };

  try {
    const resp = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });

    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    // With synth mode ON, /api/chat relays the synthesize() shape whose
    // responses are keyed "Slot N (model)" — normalize to slot_N keys
    for (const [k, v] of Object.entries(data)) {
      const m = k.match(/^Slot (\d)/);
      if (m) data[`slot_${m[1]}`] = v;
    }

    for (let i = 0; i < SLOT_COUNT; i++) {
      const key = `slot_${i}`;
      const item = data[key];
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (!panel) continue;
      const contentDiv = panel.querySelector(".response-content");
      const latency = panel.querySelector(".latency");

      if (item === null || item === undefined) {
        const isSynthSlot = data.synthesis && i === parseInt(synthSlot.value, 10);
        contentDiv.innerHTML = isSynthSlot
          ? '<span class="placeholder">Synthesizer — see output below</span>'
          : '<span class="placeholder">Disabled</span>';
        latency.textContent = "";
        continue;
      }
      // Show the model that actually generated this response, so a
      // stale backend config can't masquerade as the UI's selection
      if (item.model) {
        panel.querySelector(".model-name").textContent = item.model;
      }
      if (item.error) {
        contentDiv.innerHTML = `<span class="error">Error: ${escapeHtml(item.error)}</span>`;
        latency.textContent = `⏱ ${item.latency_ms.toFixed(0)}ms`;
      } else {
        contentDiv.textContent = item.content;
        latency.textContent = `⏱ ${item.latency_ms.toFixed(0)}ms`;
      }
      renderSlotToolTrace(panel, item.tool_trace || []);
      showSlotWarning(panel, item.warning);
    }

    // Synth mode response
    const synthResult = data["synthesis"];
    if (synthResult) {
      synthPanel.style.display = "block";
      synthModelName.textContent = synthResult.model || "synth";
      synthLatency.textContent = synthResult.latency_ms ? `⏱ ${synthResult.latency_ms.toFixed(0)}ms` : "";
      if (synthResult.error) {
        synthContent.innerHTML = `<span class="error">Error: ${escapeHtml(synthResult.error)}</span>`;
      } else {
        synthContent.textContent = synthResult.content;
      }
      renderToolTrace(synthResult.tool_trace || data.tool_trace);
      showSynthWarning(synthResult.warning);
    } else {
      synthPanel.style.display = "none";
      renderToolTrace([]);
      showSynthWarning(null);
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      showError(e.message);
    }
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "► Send to All";
    streamBtn.disabled = false;
  }
}

// ── Streaming ──────────────────────────────────────────────────────────────
async function sendStreaming(prompt, forceSynth = false) {
  sendBtn.disabled = true;
  streamBtn.disabled = true;
  streamBtn.textContent = "◉ Streaming…";

  // Set all enabled slots to "Waiting…" with cursor
  for (let i = 0; i < SLOT_COUNT; i++) {
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    if (!panel.classList.contains("disabled")) {
      panel.querySelector(".response-content").innerHTML = '<span class="placeholder">Waiting…</span>';
    }
  }

  // The synth slot doesn't stream drafts — label it so it doesn't sit on "Waiting…"
  const synthIdx = parseInt(synthSlot.value, 10);
  const expectSynth = (forceSynth || synthModeToggle.checked) && synthIdx >= 0;
  if (expectSynth) {
    const synthSlotPanel = document.querySelector(`.response-panel[data-slot="${synthIdx}"]`);
    if (synthSlotPanel) {
      synthSlotPanel.querySelector(".response-content").innerHTML =
        '<span class="placeholder">Synthesizer — runs after drafts finish</span>';
    }
  }

  const body = {
    prompt,
    system_prompt: systemPromptInput.value || null,
    synth: forceSynth,
  };

  try {
    const resp = await apiFetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });

    if (!resp.ok) throw new Error(await resp.text());

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const fullTexts = {};
    let synthText = "";
    const liveTrace = [];
    const slotTraces = {};
    renderToolTrace([]);
    showSynthWarning(null);
    clearAllSlotToolUI();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          const eventType = line.slice(7).trim();
          // Next line should be data:
          continue;
        }
        if (line.startsWith("data: ")) {
          const payload = line.slice(6).trim();
          if (!payload) continue;

          try {
            const event = JSON.parse(payload);
            const slotIdx = event.slot;

            if (event.type === "synth_start") {
              // Drafts done — synthesis starting. A thinking model may be
              // silent for a while before its first visible token.
              synthModelName.textContent = event.synth_model || "synth";
              synthLatency.textContent = "";
              synthContent.innerHTML =
                '<span class="placeholder">🧠 Synthesizing… (thinking models may be silent before the first token)</span>';
              synthPanel.style.display = "block";
              synthPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
              continue;
            }

            if (event.type === "synth_token") {
              synthText += event.content;
              synthContent.innerHTML = escapeHtml(synthText) + '<span class="cursor"></span>';
              synthContent.scrollTop = synthContent.scrollHeight;
              continue;
            }

            if (event.type === "tool_call") {
              if (event.slot != null && !event.synth) {
                // Draft slot fact-checking — live per-slot trace
                const traces = (slotTraces[event.slot] ||= []);
                traces.push(event);
                const panel = document.querySelector(
                  `.response-panel[data-slot="${event.slot}"]`
                );
                if (panel) {
                  renderSlotToolTrace(panel, traces, { open: true });
                  panel.querySelector(".response-content").innerHTML =
                    `<span class="placeholder">🔧 Fact-checking (${traces.length})…</span>`;
                }
                continue;
              }
              // Synth slot fact-checking — show the trace live for liveness
              liveTrace.push(event);
              renderToolTrace(liveTrace, { open: true });
              synthContent.innerHTML =
                `<span class="placeholder">🔧 Fact-checking with tools (${liveTrace.length} call${liveTrace.length > 1 ? "s" : ""})…</span>`;
              continue;
            }

            if (event.type === "synth") {
              // Final synth result (also arrives after synth_token streaming)
              synthModelName.textContent = event.synth_model || "synth";
              synthLatency.textContent = event.latency_ms ? `⏱ ${event.latency_ms.toFixed(0)}ms` : "";
              if (event.error) {
                synthContent.innerHTML = `<span class="error">Error: ${escapeHtml(event.error)}</span>`;
              } else {
                synthContent.textContent = event.content;
              }
              renderToolTrace(event.tool_trace || liveTrace);
              showSynthWarning(event.warning);
              synthPanel.style.display = "block";
              continue;
            }

            if (slotIdx < 0 || slotIdx >= SLOT_COUNT) continue;
            const panel = document.querySelector(`.response-panel[data-slot="${slotIdx}"]`);
            if (!panel) continue;
            const contentDiv = panel.querySelector(".response-content");
            const latency = panel.querySelector(".latency");

            if (event.type === "token") {
              if (!fullTexts[slotIdx]) fullTexts[slotIdx] = "";
              fullTexts[slotIdx] += event.content;
              contentDiv.innerHTML = escapeHtml(fullTexts[slotIdx]) + '<span class="cursor"></span>';
              contentDiv.scrollTop = contentDiv.scrollHeight;
            } else if (event.type === "done") {
              fullTexts[slotIdx] = event.full_content || fullTexts[slotIdx] || "";
              contentDiv.textContent = fullTexts[slotIdx];
              latency.textContent = `⏱ ${event.latency_ms.toFixed(0)}ms`;
              if (event.tool_trace?.length) {
                renderSlotToolTrace(panel, event.tool_trace);
              }
              showSlotWarning(panel, event.warning);
            } else if (event.type === "error") {
              contentDiv.innerHTML = `<span class="error">Error: ${escapeHtml(event.error)}</span>`;
              if (event.tool_trace?.length) {
                renderSlotToolTrace(panel, event.tool_trace);
              }
            }
          } catch (e) {
            // skip malformed JSON
          }
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      showError(e.message);
    }
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "► Send to All";
    streamBtn.disabled = false;
    streamBtn.textContent = "◉ Stream";
  }
}

// ── Synthesize ─────────────────────────────────────────────────────────────
async function sendSynth() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  // Hide previous synth panel
  synthPanel.style.display = "none";

  // Clear slot responses
  for (let i = 0; i < SLOT_COUNT; i++) {
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    panel.querySelector(".response-content").innerHTML = "";
    panel.querySelector(".latency").textContent = "";
  }
  clearAllSlotToolUI();

  // Cancel any prior in-flight request
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  synthBtn.disabled = true;
  synthBtn.textContent = "⚡ Synthesizing…";

  try {
    // Stream drafts and synthesis so long runs always show progress
    // instead of a silent multi-minute wait.
    await sendStreaming(prompt, true);
  } finally {
    synthBtn.disabled = false;
    synthBtn.textContent = "⚡ Synthesize";
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function updateAllDisabledStates() {
  for (let i = 0; i < SLOT_COUNT; i++) {
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    if (!panel) continue;
    const card = document.querySelector(`.slot-card[data-slot="${i}"]`);
    const toggle = card?.querySelector(".toggle-switch input");
    panel.classList.toggle("disabled", !toggle?.checked);
  }
}

function showError(msg) {
  for (let i = 0; i < SLOT_COUNT; i++) {
    const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
    if (panel && !panel.classList.contains("disabled")) {
      panel.querySelector(".response-content").innerHTML =
        `<span class="error">${escapeHtml(msg)}</span>`;
    }
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
