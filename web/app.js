/**
 * Cross-Market Financial Quant AI Chatbot Frontend (`web/app.js`).
 *
 * Full conversational LLM interface with:
 * - Multi-turn conversational context memory (stored in sessions)
 * - Cascading Filing Type dropdown (SET TH vs US SEC standards)
 * - Real-time SSE Agent execution (token streaming, tool traces, citations)
 * - 1-Click inline bilingual translation (TH <-> EN)
 * - Dynamic corpus stock universe & upload management
 */

// Global State
let currentSessionId = null;
let contextMenuSessionId = null;
let sessionsList = [];
let cachedTickers = [];
let activeChunksMap = {}; // Maps messageId -> retrieved chunks

// DOM Elements
const sessionsListEl = document.getElementById("sessionsList");
const sessionCountBadgeEl = document.getElementById("sessionCountBadge");
const stockCountBadgeEl = document.getElementById("stockCountBadge");
const activeStocksBadgesEl = document.getElementById("activeStocksBadges");
const currentChatTitleEl = document.getElementById("currentChatTitle");
const chatMessagesEl = document.getElementById("chatMessages");
const welcomeBannerEl = document.getElementById("welcomeBanner");
const chatScrollAreaEl = document.getElementById("chatScrollArea");
const chatInputEl = document.getElementById("chatInput");
const sendBtnEl = document.getElementById("sendBtn");
const typingIndicatorEl = document.getElementById("typingIndicator");
const uploadModalEl = document.getElementById("uploadModal");
const uploadFilingTypeEl = document.getElementById("uploadFilingType");
const uploadMarketEl = document.getElementById("uploadMarket");
const uploadCurrencyEl = document.getElementById("uploadCurrency");
const uploadAccountingStandardEl = document.getElementById("uploadAccountingStandard");
const filingTypeHintEl = document.getElementById("filingTypeHint");
const sourcesModalEl = document.getElementById("sourcesModal");
const sourcesContentEl = document.getElementById("sourcesContent");

// ============================================================================
// 1. Cascading Filing Type & Market Logic
// ============================================================================
const FILING_STANDARDS = {
  TH: {
    types: [
      { value: "56-1", label: "Form 56-1 One Report (TFRS) [สำหรับรายงานประจำปี]" },
      { value: "QUARTERLY_FINANCIALS", label: "งบการเงินรายไตรมาส (TFRS) [สำหรับ Q1, Q2, Q3]" },
      { value: "MDA", label: "รายงานคำอธิบายของฝ่ายจัดการ (MD&A)" },
    ],
    currency: "THB",
    accounting: "TFRS",
    hint: "มาตรฐาน TFRS (TAS 38 R&D Capitalization, TFRS 16 Leases, THB Reporting)",
  },
  US: {
    types: [
      { value: "10-K", label: "SEC Form 10-K (US GAAP / IFRS) [สำหรับรายงานประจำปี]" },
      { value: "10-Q", label: "SEC Form 10-Q (US GAAP / IFRS) [สำหรับรายงานรายไตรมาส]" },
      { value: "20-F", label: "SEC Form 20-F (IFRS / Foreign Issuer) [สำหรับบริษัทต่างชาติ]" },
      { value: "8-K", label: "SEC Form 8-K (Current Report / Earnings Release) [รายงานด่วน/ผลประกอบการ]" },
    ],
    currency: "USD",
    accounting: "US_GAAP",
    hint: "มาตรฐาน US SEC (ASC 730 R&D Expensing, ASC 842 Leases, USD Reporting)",
  },
};

function onMarketChange() {
  const market = uploadMarketEl ? uploadMarketEl.value : "US";
  const std = FILING_STANDARDS[market] || FILING_STANDARDS.TH;

  if (uploadFilingTypeEl) {
    uploadFilingTypeEl.innerHTML = std.types
      .map((t) => `<option value="${t.value}">${t.label}</option>`)
      .join("");
  }

  // Sync defaults
  if (uploadCurrencyEl) uploadCurrencyEl.value = std.currency;
  if (uploadAccountingStandardEl) uploadAccountingStandardEl.value = std.accounting;
  if (filingTypeHintEl) filingTypeHintEl.textContent = std.hint;

  if (typeof batchQueue !== "undefined" && batchQueue.length > 0) {
    batchQueue.forEach((item) => {
      const detected = detectMetadataFromFilename(item.file.name, market);
      item.filing_type = detected.filing_type;
    });
    renderBatchQueue();
  }
}

// ============================================================================
// 2. Chat Session Management (+ New Chat, Memory, Switching)
// ============================================================================
async function loadSessions() {
  try {
    const res = await fetch("/api/chat/sessions");
    if (!res.ok) return;
    sessionsList = await res.json();
    renderSessionsList();

    if (sessionsList.length > 0) {
      if (!currentSessionId || !sessionsList.find((s) => s.id === currentSessionId)) {
        selectSession(sessionsList[0].id);
      }
    } else {
      currentSessionId = null;
      if (currentChatTitleEl) currentChatTitleEl.textContent = "New Conversation";
      if (chatMessagesEl) chatMessagesEl.innerHTML = "";
      if (welcomeBannerEl) welcomeBannerEl.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Failed to load chat sessions:", err);
  }
}

function renderSessionsList() {
  sessionCountBadgeEl.textContent = sessionsList.length;
  if (sessionsList.length === 0) {
    sessionsListEl.innerHTML = `
      <div class="text-[11px] text-slate-500 text-center py-4">ยังไม่มีบทสนทนา</div>
    `;
    return;
  }

  sessionsListEl.innerHTML = sessionsList
    .map((s) => {
      const isActive = s.id === currentSessionId;
      const isPinned = !!s.pinned;
      const activeCls = isActive
        ? "bg-slate-800/95 text-white border-l-2 border-emerald-400 font-semibold shadow"
        : "text-slate-400 hover:bg-slate-900/80 hover:text-slate-200";

      return `
        <div onclick="selectSession('${s.id}')"
             oncontextmenu="handleSessionContextMenu(event, '${s.id}')"
             class="group relative flex items-center justify-between px-2.5 py-2 rounded-lg cursor-pointer text-xs transition ${activeCls}">
          <div class="flex items-center gap-2 truncate flex-1 mr-1">
            <span class="text-xs">${isPinned ? "📌" : "💬"}</span>
            <span id="title_${s.id}" class="truncate max-w-[155px]">${escapeHtml(s.title || "New Chat")}</span>
          </div>
          <div class="flex items-center gap-1 flex-shrink-0">
            ${isPinned ? '<span class="text-[9px] text-amber-400/90 font-mono font-bold bg-amber-400/10 px-1 py-0.2 rounded border border-amber-400/20">PIN</span>' : ""}
            <button onclick="openSessionMenuButton(event, '${s.id}')"
                    class="opacity-0 group-hover:opacity-100 hover:text-white hover:bg-slate-700/80 rounded px-1.5 py-0.5 text-xs transition font-mono tracking-widest text-slate-400"
                    title="ตัวเลือก (Rename, Pin, Delete)">
              ···
            </button>
          </div>
        </div>
      `;
    })
    .join("");
}

// ----------------------------------------------------------------------------
// Context Menu & Actions (Rename, Pin, Delete)
// ----------------------------------------------------------------------------
function openSessionMenu(sessionId, x, y) {
  contextMenuSessionId = sessionId;
  const sess = sessionsList.find((s) => s.id === sessionId);
  const menu = document.getElementById("sessionContextMenu");
  const pinLabel = document.getElementById("pinActionLabel");
  if (sess && pinLabel) {
    pinLabel.textContent = sess.pinned ? "Unpin (ถอนหมุด)" : "Pin (ปักหมุด)";
  }

  const menuWidth = 192;
  const menuHeight = 140;
  const adjustedX = Math.min(x, window.innerWidth - menuWidth - 10);
  const adjustedY = Math.min(y, window.innerHeight - menuHeight - 10);

  menu.style.left = `${adjustedX}px`;
  menu.style.top = `${adjustedY}px`;
  menu.classList.remove("hidden");
}

function openSessionMenuButton(e, sessionId) {
  e.stopPropagation();
  const rect = e.currentTarget.getBoundingClientRect();
  openSessionMenu(sessionId, rect.right - 10, rect.bottom + 5);
}

function handleSessionContextMenu(e, sessionId) {
  e.preventDefault();
  openSessionMenu(sessionId, e.clientX, e.clientY);
}

function closeSessionContextMenu() {
  const menu = document.getElementById("sessionContextMenu");
  if (menu) menu.classList.add("hidden");
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#sessionContextMenu")) {
    closeSessionContextMenu();
  }
});

function triggerRenameSession() {
  closeSessionContextMenu();
  if (!contextMenuSessionId) return;
  const sess = sessionsList.find((s) => s.id === contextMenuSessionId);
  const modal = document.getElementById("renameModal");
  const input = document.getElementById("renameInput");
  input.value = sess ? sess.title : "";
  modal.classList.remove("hidden");
  input.focus();
  input.select();
}

function closeRenameModal() {
  document.getElementById("renameModal").classList.add("hidden");
}

function handleRenameInputKey(e) {
  if (e.key === "Enter") {
    submitRenameModal();
  } else if (e.key === "Escape") {
    closeRenameModal();
  }
}

async function submitRenameModal() {
  const newTitle = document.getElementById("renameInput").value.trim();
  if (!newTitle || !contextMenuSessionId) return;
  closeRenameModal();

  try {
    const res = await fetch(`/api/chat/sessions/${contextMenuSessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: newTitle }),
    });
    if (res.ok) {
      if (currentSessionId === contextMenuSessionId) {
        currentChatTitleEl.textContent = newTitle;
      }
      await loadSessions();
    }
  } catch (err) {
    console.error("Failed to rename session:", err);
  }
}

async function triggerPinSession() {
  closeSessionContextMenu();
  if (!contextMenuSessionId) return;
  const sess = sessionsList.find((s) => s.id === contextMenuSessionId);
  const newPinnedState = sess ? !sess.pinned : true;

  try {
    const res = await fetch(`/api/chat/sessions/${contextMenuSessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned: newPinnedState }),
    });
    if (res.ok) {
      await loadSessions();
    }
  } catch (err) {
    console.error("Failed to toggle pin session:", err);
  }
}

async function triggerDeleteSession() {
  closeSessionContextMenu();
  if (!contextMenuSessionId) return;
  const sess = sessionsList.find((s) => s.id === contextMenuSessionId);
  const title = sess ? sess.title : "this chat";
  if (!confirm(`คุณต้องการลบแชท "${title}" ใช่หรือไม่?`)) return;
  await deleteSession(contextMenuSessionId);
}


async function createNewChat() {
  if (currentSessionId && sessionsList) {
    const activeSess = sessionsList.find((s) => s.id === currentSessionId);
    if (activeSess && activeSess.message_count === 0 && (activeSess.title === "New Conversation" || !activeSess.title)) {
      if (chatInputEl) chatInputEl.focus();
      return;
    }
  }
  try {
    const res = await fetch("/api/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Conversation" }),
    });
    if (res.ok) {
      const sess = await res.json();
      currentSessionId = sess.id;
      if (currentChatTitleEl) currentChatTitleEl.textContent = sess.title;
      if (chatMessagesEl) chatMessagesEl.innerHTML = "";
      if (welcomeBannerEl) welcomeBannerEl.classList.remove("hidden");
      await loadSessions();
      if (chatInputEl) chatInputEl.focus();
    }
  } catch (err) {
    console.error("Failed to create new chat:", err);
  }
}

async function selectSession(sessionId) {
  currentSessionId = sessionId;
  renderSessionsList();

  try {
    const res = await fetch(`/api/chat/sessions/${sessionId}`);
    if (!res.ok) return;
    const sess = await res.json();
    currentChatTitleEl.textContent = sess.title || "Conversation";

    chatMessagesEl.innerHTML = "";
    if (!sess.messages || sess.messages.length === 0) {
      welcomeBannerEl.classList.remove("hidden");
    } else {
      welcomeBannerEl.classList.add("hidden");
      sess.messages.forEach((m) => renderMessageBubble(m));
      scrollToBottom();
    }
  } catch (err) {
    console.error("Failed to fetch session detail:", err);
  }
}

async function deleteSession(sessionId, e) {
  if (e) e.stopPropagation();
  try {
    await fetch(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
    if (currentSessionId === sessionId) {
      currentSessionId = null;
    }
    await loadSessions();
  } catch (err) {
    console.error("Failed to delete session:", err);
  }
}

async function confirmDeleteAllChats() {
  if (!sessionsList || sessionsList.length === 0) {
    alert("ไม่มีประวัติการสนทนาให้ลบ");
    return;
  }
  const count = sessionsList.length;
  if (!confirm(`คุณต้องการลบประวัติการสนทนาทั้งหมด (${count} แชท) ใช่หรือไม่?\n(ประวัติข้อความทั้งหมดจะถูกลบออกทั้งหมด)`)) {
    return;
  }
  try {
    const res = await fetch("/api/chat/sessions", { method: "DELETE" });
    if (res.ok) {
      currentSessionId = null;
      if (chatMessagesEl) chatMessagesEl.innerHTML = "";
      if (welcomeBannerEl) welcomeBannerEl.classList.remove("hidden");
      if (currentChatTitleEl) currentChatTitleEl.textContent = "New Conversation";
      await loadSessions();
    } else {
      throw new Error(`Failed to delete all chats: ${res.status}`);
    }
  } catch (err) {
    console.error("Failed to delete all chats:", err);
    alert("เกิดข้อผิดพลาดในการลบแชททั้งหมด: " + err.message);
  }
}

// ============================================================================
// 3. Message Rendering & Bubble Construction
// ============================================================================
function renderMessageBubble(msg) {
  welcomeBannerEl.classList.add("hidden");
  const isUser = msg.role === "user";
  const bubbleId = msg.id || `msg_${Date.now()}`;

  if (msg.retrieved_chunks && msg.retrieved_chunks.length > 0) {
    activeChunksMap[bubbleId] = msg.retrieved_chunks;
  }

  const container = document.createElement("div");
  container.id = bubbleId;
  container.className = `flex gap-3 text-sm ${isUser ? "justify-end" : "justify-start"}`;

  if (isUser) {
    container.innerHTML = `
      <div class="max-w-[85%] sm:max-w-2xl bg-[#1e293b] border border-slate-700/80 rounded-2xl rounded-tr-sm px-4 py-3 text-slate-100 shadow-md">
        <div class="text-[11px] font-semibold text-slate-400 mb-1 flex items-center gap-1.5">
          <span>👤 คุณ (User)</span>
        </div>
        <div class="whitespace-pre-wrap leading-relaxed">${escapeHtml(msg.content)}</div>
      </div>
    `;
  } else {
    // Assistant Message with Markdown, Citations & Action Tools
    const parsedMarkdown = renderMarkdownWithCitations(msg.content);
    const traceHtml = renderTraceAccordion(msg.agent_trace);
    const hasSources = (msg.retrieved_chunks && msg.retrieved_chunks.length > 0) || (msg.citations && msg.citations.length > 0);

    container.innerHTML = `
      <div class="h-8 w-8 rounded-lg bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center text-slate-950 font-bold text-xs flex-shrink-0 shadow-md shadow-emerald-950/40 mt-1">
        AI
      </div>
      <div class="max-w-[92%] sm:max-w-3xl flex-1 bg-[#0b1220] border border-slate-800 rounded-2xl rounded-tl-sm px-5 py-4 text-slate-200 shadow-lg">
        <!-- Header -->
        <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80 text-[11px] mono">
          <div class="flex items-center gap-2">
            <span class="font-bold text-emerald-400">Quant Agent</span>
            ${msg.abstained ? '<span class="text-rose-400 font-semibold px-1.5 py-0.2 rounded bg-rose-500/10 border border-rose-500/20">ABSTAIN</span>' : '<span class="text-cyan-400 font-semibold px-1.5 py-0.2 rounded bg-cyan-500/10 border border-cyan-500/20">GROUNDED</span>'}
          </div>
          <span class="text-slate-500">SET 56-1 &amp; US 10-K</span>
        </div>

        <!-- Tool Traces (Accordion) -->
        ${traceHtml}

        <!-- Main Body -->
        <div class="prose-dark leading-relaxed text-sm answer-body" id="body_${bubbleId}">
          ${parsedMarkdown}
        </div>

        <!-- Footer Action Toolbar -->
        <div class="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div class="flex items-center gap-2">
            <!-- 1-Click Translation Button -->
            <button onclick="translateMessage('${bubbleId}')"
                    class="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 transition text-[11px]">
              <span>🌐</span>
              <span id="transText_${bubbleId}">แปลเป็นภาษาไทย / English</span>
            </button>

            <!-- Copy Button -->
            <button onclick="copyMessageText('${bubbleId}')"
                    class="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/80 transition text-[11px]">
              <span>📋</span>
              <span id="copyText_${bubbleId}">คัดลอก</span>
            </button>
          </div>

          <!-- View Sources Button -->
          ${hasSources ? `
            <button onclick="openSourcesModal('${bubbleId}')"
                    class="flex items-center gap-1 text-[11px] text-emerald-400 hover:text-emerald-300 font-mono transition">
              <span>📚 ข้อมูลอ้างอิง (${(msg.retrieved_chunks || []).length} chunks)</span>
            </button>
          ` : ""}
        </div>
      </div>
    `;
  }

  chatMessagesEl.appendChild(container);
  return container;
}

function renderTraceAccordion(trace) {
  if (!trace || trace.length === 0) return "";
  const stepsHtml = trace
    .map(
      (t) => `
      <div class="p-2 rounded bg-slate-950/80 border border-slate-800 text-[11px] font-mono mb-1.5">
        <div class="flex items-center justify-between text-cyan-400 font-bold">
          <span>Step ${t.step_number}: ${escapeHtml(t.tool_name)}</span>
          <span class="text-slate-500">${t.latency_ms ? t.latency_ms.toFixed(0) + "ms" : ""}</span>
        </div>
        <div class="text-slate-400 text-[10px] mt-0.5">${escapeHtml(t.thought)}</div>
        <div class="text-slate-300 text-[10px] mt-1 bg-slate-900 p-1.5 rounded truncate font-mono">
          ${escapeHtml(t.tool_output_summary || "")}
        </div>
      </div>
    `
    )
    .join("");

  return `
    <details class="mb-3 rounded bg-slate-950/40 border border-slate-800/80 p-2">
      <summary class="cursor-pointer text-[11px] font-mono text-slate-400 hover:text-slate-200 select-none flex items-center gap-1.5">
        <span>🧠 Thinking &amp; Tool Traces (${trace.length} ReAct steps)</span>
      </summary>
      <div class="mt-2 pt-2 border-t border-slate-800/60">
        ${stepsHtml}
      </div>
    </details>
  `;
}

function renderMarkdownWithCitations(rawText) {
  if (!rawText) return "";

  // Clean up LaTeX arrow notation to Unicode
  let cleaned = rawText
    .replace(/\$\uparrow\$/g, "↑")
    .replace(/\\uparrow/g, "↑")
    .replace(/\$\downarrow\$/g, "↓")
    .replace(/\\downarrow/g, "↓")
    .replace(/\$\rightarrow\$/g, "→")
    .replace(/\\rightarrow/g, "→")
    .replace(/\$\leftarrow\$/g, "←")
    .replace(/\\leftarrow/g, "←")
    .replace(/\$\pm\$/g, "±")
    .replace(/\\pm/g, "±");

  // Configure marked for safe HTML
  marked.setOptions({ gfm: true, breaks: true });
  let html = marked.parse(cleaned);

  // Replace [doc_id:chunk_id] with interactive citation pills
  html = html.replace(/\[([A-Za-z0-9_\-]+):(c\d+)\]/g, (match, docId, chunkId) => {
    return `<span class="citation-pill" onclick="showChunkSnippet('${docId}', '${chunkId}')" title="Click to view filing chunk">${docId}:${chunkId}</span>`;
  });

  return html;
}

// ============================================================================
// 4. Chat Submission & Server-Sent Events (SSE) Streaming
// ============================================================================
async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  const query = chatInputEl.value.trim();
  if (!query) return;

  chatInputEl.value = "";
  chatInputEl.style.height = "auto";
  chatInputEl.disabled = true;
  sendBtnEl.disabled = true;
  typingIndicatorEl.classList.remove("hidden");

  // If no active session, create one first
  if (!currentSessionId) {
    await createNewChat();
  }

  // 1. Render User Message Bubble
  renderMessageBubble({ role: "user", content: query });
  scrollToBottom();

  // 2. Prepare Placeholder Assistant Message Bubble
  const streamMsgId = `stream_${Date.now()}`;
  let accumulatedAnswer = "";
  let currentTrace = [];
  let currentChunks = [];
  let abstained = false;

  const assistantBubble = renderMessageBubble({
    id: streamMsgId,
    role: "assistant",
    content: "กำลังค้นหาเอกสารและวิเคราะห์ข้อมูล...",
    agent_trace: [],
  });
  scrollToBottom();

  const bodyEl = assistantBubble.querySelector(".answer-body");

  // 3. Connect to SSE Stream
  try {
    const payload = {
      query: query,
      session_id: currentSessionId,
      mode: document.getElementById("agentMode").value,
      target_currency: document.getElementById("targetCurrency").value,
      language: document.getElementById("responseLang").value,
    };

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP Error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/^event:\s*(\w+)/m);
        const dataMatch = block.match(/^data:\s*(.+)$/m);

        if (eventMatch && dataMatch) {
          const eventType = eventMatch[1];
          let eventData = {};
          try {
            eventData = JSON.parse(dataMatch[1]);
          } catch (_) {
            continue;
          }

          if (eventType === "token") {
            accumulatedAnswer += eventData.text || "";
            bodyEl.innerHTML = renderMarkdownWithCitations(accumulatedAnswer);
            scrollToBottom();
          } else if (eventType === "tool_result") {
            currentTrace.push(eventData);
          } else if (eventType === "complete") {
            accumulatedAnswer = eventData.answer || accumulatedAnswer;
            abstained = eventData.abstained || false;
            currentChunks = eventData.retrieved_chunks || [];
            currentTrace = eventData.agent_trace || currentTrace;
          }
        }
      }
    }

    // Final render of the completed bubble
    assistantBubble.remove();
    renderMessageBubble({
      id: streamMsgId,
      role: "assistant",
      content: accumulatedAnswer,
      agent_trace: currentTrace,
      retrieved_chunks: currentChunks,
      abstained: abstained,
    });
    scrollToBottom();

    // Refresh sessions list to update titles/counts
    await loadSessions();
  } catch (err) {
    console.error("Streaming error:", err);
    bodyEl.innerHTML = `<div class="text-rose-400 font-mono text-xs">⚠️ เกิดข้อผิดพลาดในการวิเคราะห์: ${escapeHtml(err.message)}</div>`;
  } finally {
    chatInputEl.disabled = false;
    sendBtnEl.disabled = false;
    typingIndicatorEl.classList.add("hidden");
    chatInputEl.focus();
  }
}

function handleTextareaKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleChatSubmit();
  }
}

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
}

function scrollToBottom() {
  chatScrollAreaEl.scrollTop = chatScrollAreaEl.scrollHeight;
}

function selectSuggestedPrompt(text) {
  chatInputEl.value = text;
  autoResizeTextarea(chatInputEl);
  handleChatSubmit();
}

// ============================================================================
// 5. 1-Click Bilingual Translation (TH <-> EN)
// ============================================================================
async function translateMessage(bubbleId) {
  const bodyEl = document.getElementById(`body_${bubbleId}`);
  const btnTextEl = document.getElementById(`transText_${bubbleId}`);
  if (!bodyEl) return;

  const currentText = bodyEl.innerText.trim();
  if (!currentText) return;

  // Detect current language roughly (check if Thai characters present)
  const isCurrentlyThai = /[\u0E00-\u0E7F]/.test(currentText);
  const targetLang = isCurrentlyThai ? "en" : "th";
  const targetLangLabel = targetLang === "th" ? "ภาษาไทย" : "English";

  btnTextEl.textContent = `กำลังแปลเป็น ${targetLangLabel}...`;

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: currentText, target_lang: targetLang }),
    });

    if (res.ok) {
      const data = await res.json();
      bodyEl.innerHTML = renderMarkdownWithCitations(data.translated_text);
      btnTextEl.textContent = targetLang === "th" ? "Translate to English" : "แปลเป็นภาษาไทย";
    } else {
      btnTextEl.textContent = "Translation Error";
    }
  } catch (err) {
    console.error("Translation error:", err);
    btnTextEl.textContent = "Error";
  }
}

function copyMessageText(bubbleId) {
  const bodyEl = document.getElementById(`body_${bubbleId}`);
  const copyTextEl = document.getElementById(`copyText_${bubbleId}`);
  if (!bodyEl) return;

  navigator.clipboard.writeText(bodyEl.innerText).then(() => {
    copyTextEl.textContent = "คัดลอกแล้ว! ✓";
    setTimeout(() => {
      copyTextEl.textContent = "คัดลอก";
    }, 2000);
  });
}

// ============================================================================
// 6. Source Chunks Modal
// ============================================================================
function openSourcesModal(bubbleId) {
  const chunks = activeChunksMap[bubbleId] || [];
  if (chunks.length === 0) {
    sourcesContentEl.innerHTML = `<div class="text-slate-400">ไม่มีข้อมูล Chunks ในข้อความนี้</div>`;
  } else {
    sourcesContentEl.innerHTML = chunks
      .map(
        (c) => `
        <div class="p-3 rounded-lg bg-slate-950 border border-slate-800">
          <div class="flex items-center justify-between text-emerald-400 font-bold mb-1">
            <span>[${c.doc_id}:${c.chunk_id}] - ${escapeHtml(c.section || "Overview")}</span>
            <span class="text-xs px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">Score: ${(c.score || 0).toFixed(4)}</span>
          </div>
          <div class="text-[11px] text-slate-400 mono mb-1">${c.market} • ${c.filing_type} • ${c.ticker}</div>
          <div class="text-slate-300 whitespace-pre-wrap leading-relaxed bg-slate-900/60 p-2.5 rounded border border-slate-800/80 text-[11px]">
            ${escapeHtml(c.text)}
          </div>
        </div>
      `
      )
      .join("");
  }
  sourcesModalEl.classList.remove("hidden");
}

function closeSourcesModal() {
  sourcesModalEl.classList.add("hidden");
}

function showChunkSnippet(docId, chunkId) {
  // Find in any active chunks
  for (const chunks of Object.values(activeChunksMap)) {
    const match = chunks.find((c) => c.doc_id === docId && c.chunk_id === chunkId);
    if (match) {
      openSourcesModal(null);
      sourcesContentEl.innerHTML = `
        <div class="p-3 rounded-lg bg-slate-950 border border-emerald-500/50">
          <div class="flex items-center justify-between text-emerald-400 font-bold mb-1">
            <span>[${match.doc_id}:${match.chunk_id}] - ${escapeHtml(match.section || "Overview")}</span>
            <span class="text-xs px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">Score: ${(match.score || 0).toFixed(4)}</span>
          </div>
          <div class="text-[11px] text-slate-400 mono mb-1">${match.market} • ${match.filing_type} • ${match.ticker}</div>
          <div class="text-slate-200 whitespace-pre-wrap leading-relaxed bg-slate-900/60 p-2.5 rounded border border-slate-800/80 text-[11px]">
            ${escapeHtml(match.text)}
          </div>
        </div>
      `;
      return;
    }
  }
}

// ============================================================================
// 7. Dynamic Stock Universe & Upload Filing
// ============================================================================
async function refreshStockUniverse() {
  try {
    const res = await fetch("/api/tickers");
    if (!res.ok) return;
    const data = await res.json();
    cachedTickers = data.tickers || [];
    const totalFilings = data.total_filings_count ?? cachedTickers.reduce((acc, t) => acc + (t.filings_count || 1), 0);
    stockCountBadgeEl.textContent = `${cachedTickers.length} หุ้น (${totalFilings} เอกสาร)`;

    if (cachedTickers.length === 0) {
      activeStocksBadgesEl.innerHTML = `<span class="text-[10px] text-slate-500">Corpus ว่างเปล่า (คลิก Upload Filing)</span>`;
      return;
    }

    activeStocksBadgesEl.innerHTML = cachedTickers
      .map((item) => {
        const isSET = item.market === "SET" || item.market === "TH";
        const badgeCls = isSET
          ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/30"
          : "bg-sky-500/15 text-sky-300 border-sky-500/30 hover:bg-sky-500/30";
        
        let filingsHtml = "";
        if (item.filings && item.filings.length > 0) {
          filingsHtml = `
            <div class="flex flex-col gap-1 mt-1 pl-1">
              ${item.filings.map(f => `
                <div class="text-[9.5px] px-1.5 py-0.5 rounded bg-slate-900/90 text-slate-300 border border-slate-800 font-mono flex items-center justify-between hover:border-slate-600 cursor-pointer"
                     onclick="event.stopPropagation(); selectSuggestedPrompt('วิเคราะห์และสรุปสาระสำคัญของ ${item.name || item.ticker} จากงบ ${f.label || f.doc_id}')">
                  <span>📄 ${f.label || f.doc_id}</span>
                  <span class="text-[8px] text-emerald-400">active</span>
                </div>
              `).join("")}
            </div>
          `;
        }

        return `
          <div class="bg-slate-950/70 border border-slate-800/80 rounded-lg p-1.5 mb-1.5">
            <button onclick="selectSuggestedPrompt('สรุปผลการดำเนินงานและงบการเงินของ ${item.ticker}')"
                    class="px-2 py-1 rounded border ${badgeCls} text-[10.5px] font-mono font-bold transition flex items-center justify-between w-full">
              <span>${item.ticker} <span class="font-normal opacity-80 text-[9px]">${item.name ? '(' + item.name + ')' : ''}</span></span>
              <span class="text-[9px] opacity-70">${isSET ? "SET" : "US"}</span>
            </button>
            ${filingsHtml}
          </div>
        `;
      })
      .join("");
  } catch (err) {
    console.error("Failed to refresh stock universe:", err);
  }
}

// ============================================================================
// Multi-File Batch Upload Queue State & Handlers
// ============================================================================
let batchQueue = [];

function openUploadModal() {
  onMarketChange();
  uploadModalEl.classList.remove("hidden");
}

function closeUploadModal() {
  uploadModalEl.classList.add("hidden");
  const feedback = document.getElementById("uploadFeedback");
  if (feedback) feedback.classList.add("hidden");
}

function clearBatchQueue() {
  batchQueue = [];
  const fileInput = document.getElementById("uploadFile");
  if (fileInput) fileInput.value = "";
  const progressWrapper = document.getElementById("batchProgressWrapper");
  if (progressWrapper) progressWrapper.classList.add("hidden");
  renderBatchQueue();
}

function handleDragOver(e) {
  e.preventDefault();
  const dropzone = document.getElementById("uploadDropzone");
  if (dropzone) dropzone.classList.add("border-emerald-400", "bg-emerald-950/20");
}

function handleDragLeave(e) {
  const dropzone = document.getElementById("uploadDropzone");
  if (dropzone) dropzone.classList.remove("border-emerald-400", "bg-emerald-950/20");
}

function handleFileDrop(e) {
  e.preventDefault();
  const dropzone = document.getElementById("uploadDropzone");
  if (dropzone) dropzone.classList.remove("border-emerald-400", "bg-emerald-950/20");
  if (e.dataTransfer && e.dataTransfer.files) {
    addFilesToQueue(e.dataTransfer.files);
  }
}

function handleFilesPicked(e) {
  if (e.target && e.target.files) {
    addFilesToQueue(e.target.files);
  }
}

function detectMetadataFromFilename(filename, market) {
  const lower = filename.toLowerCase();

  // 1. Detect Year: look for 2020-2035
  let year = "2026";
  const yearEl = document.getElementById("uploadYear");
  if (yearEl && yearEl.value) year = yearEl.value;
  const yearMatch = filename.match(/\b(20[2-3]\d)\b/);
  if (yearMatch) {
    year = yearMatch[1];
  }

  // 2. Detect Quarter
  let quarter = "FY";
  if (/second[\-_ ]?quarter|2nd[\-_ ]?quarter|\bq2\b|quarter[\-_ ]?2/i.test(filename)) {
    quarter = "Q2";
  } else if (/third[\-_ ]?quarter|3rd[\-_ ]?quarter|\bq3\b|quarter[\-_ ]?3/i.test(filename)) {
    quarter = "Q3";
  } else if (/first[\-_ ]?quarter|1st[\-_ ]?quarter|\bq1\b|quarter[\-_ ]?1/i.test(filename)) {
    quarter = "Q1";
  } else if (/fourth[\-_ ]?quarter|4th[\-_ ]?quarter|\bq4\b|quarter[\-_ ]?4/i.test(filename)) {
    quarter = "Q4";
  } else if (/annual|10\-k|56\-1|full[\-_ ]?year|\bfy\b/i.test(filename)) {
    quarter = "FY";
  } else {
    const qEl = document.getElementById("uploadQuarter");
    quarter = qEl ? qEl.value : "FY";
  }

  // 3. Detect Filing Type based on Market
  let filing_type = market === "US" ? (quarter === "FY" ? "10-K" : "10-Q") : (quarter === "FY" ? "56-1" : "QUARTERLY_FINANCIALS");
  if (market === "US") {
    if (quarter in { Q1: 1, Q2: 1, Q3: 1 } || /10[\-_]?q/i.test(filename)) {
      filing_type = "10-Q";
    } else if (/8[\-_]?k/i.test(filename)) {
      filing_type = "8-K";
    } else if (/20[\-_]?f/i.test(filename)) {
      filing_type = "20-F";
    } else if (quarter === "FY" || /10[\-_]?k/i.test(filename)) {
      filing_type = "10-K";
    }
  } else {
    if (quarter in { Q1: 1, Q2: 1, Q3: 1 }) {
      filing_type = "QUARTERLY_FINANCIALS";
    } else if (/mda/i.test(filename)) {
      filing_type = "MDA";
    } else {
      filing_type = "56-1";
    }
  }

  // 4. Detect Ticker and Name
  let detectedTicker = "";
  let detectedName = "";
  if (/spacex/i.test(filename)) {
    detectedTicker = "SPCX";
    detectedName = "SpaceX";
  } else if (/nvidia|nvda/i.test(filename)) {
    detectedTicker = "NVDA";
    detectedName = "NVIDIA Corporation";
  } else if (/tesla|tsla/i.test(filename)) {
    detectedTicker = "TSLA";
    detectedName = "Tesla, Inc.";
  } else if (/apple|aapl/i.test(filename)) {
    detectedTicker = "AAPL";
    detectedName = "Apple Inc.";
  } else if (/microsoft|msft/i.test(filename)) {
    detectedTicker = "MSFT";
    detectedName = "Microsoft Corporation";
  } else if (/delta/i.test(filename)) {
    detectedTicker = "DELTA";
    detectedName = "Delta Electronics (Thailand)";
  } else if (/ptt/i.test(filename)) {
    detectedTicker = "PTT";
    detectedName = "PTT Public Company Limited";
  } else if (/advanc|ais/i.test(filename)) {
    detectedTicker = "ADVANC";
    detectedName = "Advanced Info Service (AIS)";
  } else if (/bdms/i.test(filename)) {
    detectedTicker = "BDMS";
    detectedName = "Bangkok Dusit Medical Services";
  } else if (/cpall/i.test(filename)) {
    detectedTicker = "CPALL";
    detectedName = "CP ALL Public Company Limited";
  }

  return { year, quarter, filing_type, detectedTicker, detectedName };
}

function addFilesToQueue(files) {
  if (!files || files.length === 0) return;
  const market = document.getElementById("uploadMarket").value;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const exists = batchQueue.some((item) => item.file.name === file.name && item.file.size === file.size);
    if (exists) continue;

    const detected = detectMetadataFromFilename(file.name, market);

    // Auto-fill top form ticker/name if empty
    const tickerInput = document.getElementById("uploadTicker");
    const nameInput = document.getElementById("uploadName");
    if (!tickerInput.value.trim() && detected.detectedTicker) {
      tickerInput.value = detected.detectedTicker;
    }
    if (!nameInput.value.trim() && detected.detectedName) {
      nameInput.value = detected.detectedName;
    }

    batchQueue.push({
      id: "file_" + Math.random().toString(36).substring(2, 9),
      file: file,
      year: detected.year,
      quarter: detected.quarter,
      filing_type: detected.filing_type,
      status: "pending",
      statusMessage: "รออัปโหลด",
    });
  }

  renderBatchQueue();
}

function updateQueueItem(idx, field, value) {
  if (batchQueue[idx]) {
    batchQueue[idx][field] = value;
  }
}

function removeQueueItem(idx) {
  batchQueue.splice(idx, 1);
  renderBatchQueue();
}

function renderBatchQueue() {
  const container = document.getElementById("batchFilesQueueContainer");
  const listEl = document.getElementById("batchFilesList");
  const badgeEl = document.getElementById("batchFilesCountBadge");
  const submitBtn = document.getElementById("uploadSubmitBtn");
  const market = document.getElementById("uploadMarket").value;
  const std = FILING_STANDARDS[market] || FILING_STANDARDS.TH;

  if (batchQueue.length === 0) {
    if (container) container.classList.add("hidden");
    if (submitBtn) submitBtn.innerHTML = "<span>🚀 อัปโหลด &amp; สร้าง Vector Chunks</span>";
    return;
  }

  if (container) container.classList.remove("hidden");
  if (badgeEl) badgeEl.textContent = `${batchQueue.length} ไฟล์`;
  if (submitBtn) submitBtn.innerHTML = `<span>🚀 อัปโหลดทั้งหมด (${batchQueue.length} ไฟล์)</span>`;

  if (listEl) {
    listEl.innerHTML = batchQueue
      .map((item, idx) => {
        const sizeKB = (item.file.size / 1024).toFixed(1);
        const sizeStr = item.file.size > 1024 * 1024 ? `${(item.file.size / (1024 * 1024)).toFixed(1)} MB` : `${sizeKB} KB`;

        let statusClass = "text-slate-400 border-slate-700 bg-slate-900/60";
        if (item.status === "uploading") {
          statusClass = "text-sky-300 border-sky-600 bg-sky-950/40 animate-pulse";
        } else if (item.status === "success") {
          statusClass = "text-emerald-300 border-emerald-600 bg-emerald-950/40";
        } else if (item.status === "error") {
          statusClass = "text-rose-300 border-rose-600 bg-rose-950/40";
        }

        const quarterOptions = ["FY", "Q1", "Q2", "Q3", "Q4"]
          .map((q) => `<option value="${q}" ${item.quarter === q ? "selected" : ""}>${q}</option>`)
          .join("");

        const typeOptions = std.types
          .map((t) => `<option value="${t.value}" ${item.filing_type === t.value ? "selected" : ""}>${t.label.split(" [")[0]}</option>`)
          .join("");

        return `
          <div class="bg-slate-950 border border-slate-800 rounded-lg p-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <span class="text-base flex-shrink-0">📄</span>
              <div class="truncate">
                <div class="font-semibold text-slate-200 truncate" title="${escapeHtml(item.file.name)}">
                  ${escapeHtml(item.file.name)}
                </div>
                <div class="text-[10px] text-slate-500 font-mono mt-0.5">
                  ${sizeStr} &bull; <span class="px-1.5 py-0.2 rounded border text-[9px] ${statusClass}">${item.statusMessage}</span>
                </div>
              </div>
            </div>

            <div class="flex items-center gap-1.5 flex-shrink-0">
              <div class="flex items-center gap-1">
                <span class="text-[10px] text-slate-500">ปี:</span>
                <input type="number" min="1990" max="2035" value="${item.year}"
                       onchange="updateQueueItem(${idx}, 'year', this.value)"
                       class="w-16 bg-slate-900 border border-slate-700 rounded px-1.5 py-1 text-white font-mono text-[11px] focus:outline-none focus:border-emerald-500" />
              </div>

              <div class="flex items-center gap-1">
                <span class="text-[10px] text-slate-500">งวด:</span>
                <select onchange="updateQueueItem(${idx}, 'quarter', this.value)"
                        class="bg-slate-900 border border-slate-700 rounded px-1.5 py-1 text-white font-mono text-[11px] focus:outline-none focus:border-emerald-500">
                  ${quarterOptions}
                </select>
              </div>

              <div class="flex items-center gap-1">
                <span class="text-[10px] text-slate-500">ประเภท:</span>
                <select onchange="updateQueueItem(${idx}, 'filing_type', this.value)"
                        class="bg-slate-900 border border-slate-700 rounded px-1.5 py-1 text-white font-mono text-[11px] focus:outline-none focus:border-emerald-500 max-w-[130px] truncate">
                  ${typeOptions}
                </select>
              </div>

              <button type="button" onclick="removeQueueItem(${idx})"
                      class="text-slate-500 hover:text-rose-400 p-1 rounded hover:bg-rose-950/40 text-xs font-mono ml-1"
                      title="นำไฟล์นี้ออกจากคิว">✕</button>
            </div>
          </div>
        `;
      })
      .join("");
  }
}

async function handleUploadFiling(e) {
  e.preventDefault();
  if (batchQueue.length === 0) {
    alert("กรุณาเลือกไฟล์เอกสารอย่างน้อย 1 ไฟล์เพื่ออัปโหลด (สามารถเลือกได้หลายไฟล์พร้อมกัน)");
    return;
  }

  const ticker = document.getElementById("uploadTicker").value.trim().toUpperCase();
  const name = document.getElementById("uploadName").value.trim();
  const market = document.getElementById("uploadMarket").value;
  const currency = document.getElementById("uploadCurrency").value;
  const accounting = document.getElementById("uploadAccountingStandard").value;

  if (!ticker || !name) {
    alert("กรุณากรอก Stock Ticker Symbol และ Company Name ให้ครบถ้วน");
    return;
  }

  const submitBtn = document.getElementById("uploadSubmitBtn");
  const progressWrapper = document.getElementById("batchProgressWrapper");
  const progressBar = document.getElementById("batchProgressBar");
  const progressText = document.getElementById("batchProgressText");
  const progressPct = document.getElementById("batchProgressPct");
  const feedback = document.getElementById("uploadFeedback");

  submitBtn.disabled = true;
  if (feedback) feedback.classList.add("hidden");
  if (progressWrapper) progressWrapper.classList.remove("hidden");

  let successCount = 0;
  let totalChunksAdded = 0;
  let errorCount = 0;

  for (let i = 0; i < batchQueue.length; i++) {
    const item = batchQueue[i];
    item.status = "uploading";
    item.statusMessage = "กำลังประมวลผล & สร้าง Chunks...";
    renderBatchQueue();

    const currentPct = Math.round((i / batchQueue.length) * 100);
    if (progressBar) progressBar.style.width = `${currentPct}%`;
    if (progressPct) progressPct.textContent = `${currentPct}%`;
    if (progressText) progressText.textContent = `กำลังอัปโหลดไฟล์ที่ ${i + 1}/${batchQueue.length}: ${item.file.name}...`;

    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("ticker", ticker);
    formData.append("name", name);
    formData.append("market", market);
    formData.append("year", item.year);
    formData.append("quarter", item.quarter);
    formData.append("filing_type", item.filing_type);
    formData.append("currency", currency);
    formData.append("accounting_standard", accounting);

    // Optional metrics for single file upload
    if (batchQueue.length === 1) {
      const rev = document.getElementById("metricRevenue")?.value;
      if (rev) formData.append("revenue", rev);
      const np = document.getElementById("metricNetProfit")?.value;
      if (np) formData.append("net_profit", np);
      const assets = document.getElementById("metricTotalAssets")?.value;
      if (assets) formData.append("total_assets", assets);
      const equity = document.getElementById("metricTotalEquity")?.value;
      if (equity) formData.append("total_equity", equity);
      const gp = document.getElementById("metricGrossProfit")?.value;
      if (gp) formData.append("gross_profit", gp);
    }

    try {
      const res = await fetch("/api/filings/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || `Error status ${res.status}`);
      }
      item.status = "success";
      item.statusMessage = `✓ สำเร็จ (+${data.chunks_added} chunks)`;
      successCount++;
      totalChunksAdded += data.chunks_added;
    } catch (err) {
      console.error(`Upload error for ${item.file.name}:`, err);
      item.status = "error";
      item.statusMessage = `✕ ${err.message}`;
      errorCount++;
    }

    renderBatchQueue();
  }

  if (progressBar) progressBar.style.width = "100%";
  if (progressPct) progressPct.textContent = "100%";
  if (progressText) progressText.textContent = `เสร็จสิ้นการประมวลผล (${successCount}/${batchQueue.length} สำเร็จ)`;

  if (feedback) {
    if (errorCount === 0) {
      feedback.textContent = `✓ อัปโหลดสำเร็จครบทั้ง ${successCount} ไฟล์! รวมเพิ่ม ${totalChunksAdded} Chunks สำหรับ ${ticker} เรียบร้อย`;
      feedback.className = "text-xs rounded p-2.5 font-mono bg-emerald-950/80 text-emerald-300 border border-emerald-800";
      feedback.classList.remove("hidden");

      setTimeout(() => {
        closeUploadModal();
        refreshStockUniverse();
        clearBatchQueue();
        selectSuggestedPrompt(`ช่วยวิเคราะห์และเปรียบเทียบงบการเงินของหุ้น ${ticker} ในแต่ละไตรมาสย้อนหลังล่าสุดให้หน่อย`);
      }, 1500);
    } else {
      feedback.textContent = `⚠️ อัปโหลดสำเร็จ ${successCount} ไฟล์, ล้มเหลว ${errorCount} ไฟล์ (ดูรายละเอียดในรายการด้านบน)`;
      feedback.className = "text-xs rounded p-2.5 font-mono bg-amber-950/80 text-amber-300 border border-amber-800";
      feedback.classList.remove("hidden");
      refreshStockUniverse();
    }
  }

  submitBtn.disabled = false;
  submitBtn.innerHTML = `<span>🚀 อัปโหลดทั้งหมด (${batchQueue.length} ไฟล์)</span>`;
}

async function confirmClearCorpus() {
  if (!confirm("คุณต้องการลบข้อมูลเอกสารและฐานข้อมูลหุ้นทั้งหมดใน Corpus ใช่หรือไม่?\n(ข้อมูลทั้งหมดจะถูกรีเซ็ตเป็นสถานะว่างเปล่า เพื่อให้คุณอัปโหลดเองใหม่)")) {
    return;
  }
  try {
    const res = await fetch("/api/corpus/clear", { method: "POST" });
    if (res.ok) {
      alert("ล้างข้อมูลใน Corpus สำเร็จเรียบร้อย!");
      await refreshStockUniverse();
      await createNewChat();
    }
  } catch (err) {
    alert("ไม่สามารถล้างข้อมูลได้: " + err.message);
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("hidden");
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Initialization on load
document.addEventListener("DOMContentLoaded", () => {
  onMarketChange();
  loadSessions();
  refreshStockUniverse();
});

// Backward-compatible alias for e2e tests
function executeStreamingQuery() { return sendChatMessage(); }
window.executeStreamingQuery = executeStreamingQuery;
