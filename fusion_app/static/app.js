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
        // Auto-expand Advanced section if a URL override is set
        if (slot.base_url_override) {
          const advBody = card.querySelector(".slot-advanced-body");
          const advToggle = card.querySelector(".slot-advanced-toggle");
          if (advBody) advBody.style.display = "block";
          if (advToggle) advToggle.textContent = "▾ Advanced";
        }
      }

      // Update response panel header
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (panel) {
        panel.classList.toggle("disabled", !slot.enabled);
        panel.querySelector(".model-name").textContent = slot.model || "—";
      }
    });

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
    slots.push(slotData);
  }

  // Build body — only include key if user typed something new
  const body = {
    ollama_base_url: ollamaUrl.value,
    synth_slot: parseInt(synthSlot.value, 10),
    synth_mode: synthModeToggle.checked,
    slots,
  };

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
      </div>
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
    } else {
      synthPanel.style.display = "none";
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
async function sendStreaming(prompt) {
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

  const body = {
    prompt,
    system_prompt: systemPromptInput.value || null,
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

            if (event.type === "synth") {
              // Synth result after streaming in synth mode
              synthModelName.textContent = event.synth_model || "synth";
              synthLatency.textContent = event.latency_ms ? `⏱ ${event.latency_ms.toFixed(0)}ms` : "";
              if (event.error) {
                synthContent.innerHTML = `<span class="error">Error: ${escapeHtml(event.error)}</span>`;
              } else {
                synthContent.textContent = event.content;
              }
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
            } else if (event.type === "error") {
              contentDiv.innerHTML = `<span class="error">Error: ${escapeHtml(event.error)}</span>`;
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

  // Cancel any prior in-flight request
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  synthBtn.disabled = true;
  synthBtn.textContent = "⚡ Synthesizing…";
  sendBtn.disabled = true;
  streamBtn.disabled = true;
  markPanelsWaiting();

  const body = {
    prompt,
    system_prompt: systemPromptInput.value || null,
  };

  try {
    const resp = await apiFetch("/api/synth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });

    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    // Show individual responses
    for (const [label, item] of Object.entries(data.responses)) {
      // Find the slot index from the label "Slot N (model)"
      const match = label.match(/Slot (\d)/);
      if (!match) continue;
      const i = parseInt(match[1], 10);
      const panel = document.querySelector(`.response-panel[data-slot="${i}"]`);
      if (!panel) continue;
      const content = panel.querySelector(".response-content");
      const latency = panel.querySelector(".latency");

      if (item.model) {
        panel.querySelector(".model-name").textContent = item.model;
      }
      if (item.error) {
        content.innerHTML = `<span class="error">Error: ${escapeHtml(item.error)}</span>`;
        latency.textContent = `⏱ ${item.latency_ms.toFixed(0)}ms`;
      } else {
        content.textContent = item.content;
        latency.textContent = `⏱ ${item.latency_ms.toFixed(0)}ms`;
      }
    }

    // The synth slot answers below, not as a panelist — clear its waiting state
    const synthSlotIdx = parseInt(synthSlot.value, 10);
    const synthSlotPanel = document.querySelector(`.response-panel[data-slot="${synthSlotIdx}"]`);
    if (synthSlotPanel) {
      synthSlotPanel.querySelector(".response-content").innerHTML =
        '<span class="placeholder">Synthesizer — see output below</span>';
    }

    // Show synth output
    const synthesis = data.synthesis;
    if (synthesis.error) {
      synthContent.innerHTML = `<span class="error">Error: ${escapeHtml(synthesis.error)}</span>`;
    } else {
      synthContent.textContent = synthesis.content;
    }
    synthModelName.textContent = `${synthesis.model}`;
    synthLatency.textContent = synthesis.error ? "" : `⏱ ${synthesis.latency_ms.toFixed(0)}ms`;
    synthPanel.style.display = "block";
    synthPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (e) {
    if (e.name !== "AbortError") {
      synthContent.innerHTML = `<span class="error">Synth failed: ${escapeHtml(e.message)}</span>`;
      synthPanel.style.display = "block";
    }
  } finally {
    synthBtn.disabled = false;
    synthBtn.textContent = "⚡ Synthesize";
    sendBtn.disabled = false;
    streamBtn.disabled = false;
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
