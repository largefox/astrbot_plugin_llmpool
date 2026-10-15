/**
 * Frontend logic for LLM Pool Dashboard using window.AstrBotPluginPage bridge
 */

const bridge = window.AstrBotPluginPage;

let currentConfig = {};
let availableProviders = [];
let allDiscoveredNodes = new Set();
let nodeGroupFilter = new Set();
let currentLogPage = 1;
const LOG_PAGE_SIZE = 50;

// Cost analysis time range: "today" | "yesterday" | "24h" | "week"
let currentCostRange = "today";

function round1(n) {
  return Math.round(n * 10) / 10;
}

/**
 * Compute window_seconds for the cost analysis time range selector.
 * - "today":     from local 00:00 to now
 * - "yesterday": full 24h window (yesterday 00:00 ~ 23:59:59)
 * - "24h":       rolling last 24 hours from now
 * - "week":      from 7 days ago at local 00:00 to now
 */
function getCostWindowSeconds(range) {
  const now = new Date();
  switch (range) {
    case "today": {
      const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0);
      return Math.max(1, Math.round((now.getTime() - startOfDay.getTime()) / 1000));
    }
    case "yesterday":
      return 86400;
    case "week": {
      const startOfWeek = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 6, 0, 0, 0);
      return Math.max(1, Math.round((now.getTime() - startOfWeek.getTime()) / 1000));
    }
    case "24h":
    default:
      return 86400;
  }
}

function getCostRangeLabel(range) {
  switch (range) {
    case "today":
      return "今日";
    case "yesterday":
      return "昨日";
    case "week":
      return "本周";
    case "24h":
    default:
      return "近 24 小时";
  }
}

function formatLocalTime(timestampSeconds) {
  if (!timestampSeconds) return "-";
  const date = new Date(timestampSeconds * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const h = pad(date.getHours());
  const min = pad(date.getMinutes());
  const s = pad(date.getSeconds());
  return `${y}-${m}-${d} ${h}:${min}:${s}`;
}

function getPluginName() {
  const ctx = bridge?.getContext?.();
  return ctx?.pluginName || "astrbot_plugin_llmpool";
}

// Toast Notification Utility (Iframe Sandbox Safe)
function showToast(message, type = "info", duration = 3000) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${escapeHtml(message)}</span>
    <span style="cursor:pointer;opacity:0.8;font-size:1.1rem;line-height:1;" title="关闭">&times;</span>
  `;
  const closeBtn = toast.querySelector("span:last-child");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => toast.remove());
  }
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.transition = "opacity 0.25s ease, transform 0.25s ease";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(-8px)";
    setTimeout(() => toast.remove(), 260);
  }, duration);
}

// In-page Confirm Modal Utility (Bypasses window.confirm sandbox block)
function showConfirm({ title = "提示", message, okText = "确认", okClass = "btn-warning" }) {
  return new Promise((resolve) => {
    const modal = document.getElementById("confirm-modal");
    const titleEl = document.getElementById("confirm-modal-title");
    const msgEl = document.getElementById("confirm-modal-message");
    const cancelBtn = document.getElementById("confirm-modal-cancel");
    const okBtn = document.getElementById("confirm-modal-ok");

    if (!modal || !titleEl || !msgEl || !cancelBtn || !okBtn) {
      resolve(true);
      return;
    }

    titleEl.textContent = title;
    msgEl.textContent = message;
    okBtn.textContent = okText;
    okBtn.className = `btn ${okClass}`;

    const cleanup = () => {
      modal.style.display = "none";
      cancelBtn.removeEventListener("click", onCancel);
      okBtn.removeEventListener("click", onOk);
      modal.removeEventListener("click", onOverlay);
    };

    const onCancel = () => {
      cleanup();
      resolve(false);
    };

    const onOk = () => {
      cleanup();
      resolve(true);
    };

    const onOverlay = (e) => {
      if (e.target === modal) {
        cleanup();
        resolve(false);
      }
    };

    cancelBtn.addEventListener("click", onCancel);
    okBtn.addEventListener("click", onOk);
    modal.addEventListener("click", onOverlay);

    modal.style.display = "flex";
  });
}

// Tab switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.tab;
    document.getElementById(target)?.classList.add("active");

    if (target === "tab-logs") loadLogs();
    if (target === "tab-nodes") loadConfig();
    if (target === "tab-overview") loadStats();
    if (target === "tab-cost") loadCostAnalysis();
    if (target === "tab-inspector") {
      startInspector();
    } else {
      stopInspector();
    }
  });
});

// ─── Intro Wizard (Step-by-step) ───

let introCurrentStep = 1;
const INTRO_TOTAL_STEPS = 3;

function showIntroModal() {
  const modal = document.getElementById("intro-modal");
  if (!modal) return;
  introCurrentStep = 1;
  renderIntroStep();
  modal.style.display = "flex";
}

function hideIntroModal() {
  const modal = document.getElementById("intro-modal");
  if (!modal) return;
  modal.style.display = "none";
}

function renderIntroStep() {
  // Show/hide step panels
  document.querySelectorAll(".intro-wizard-step").forEach((panel) => {
    panel.style.display = panel.dataset.step == String(introCurrentStep) ? "" : "none";
  });

  // Update progress dots
  document.querySelectorAll(".intro-dot").forEach((dot, idx) => {
    dot.classList.toggle("active", idx + 1 === introCurrentStep);
  });

  // Update buttons
  const btnPrev = document.getElementById("intro-btn-prev");
  const btnNext = document.getElementById("intro-btn-next");
  const btnFinish = document.getElementById("intro-btn-finish");
  const btnSkip = document.getElementById("intro-btn-skip");

  if (btnPrev) btnPrev.style.display = introCurrentStep > 1 ? "" : "none";
  if (btnNext) btnNext.style.display = introCurrentStep < INTRO_TOTAL_STEPS ? "" : "none";
  if (btnFinish) btnFinish.style.display = introCurrentStep === INTRO_TOTAL_STEPS ? "" : "none";
  if (btnSkip) btnSkip.style.display = introCurrentStep < INTRO_TOTAL_STEPS ? "" : "none";
}

function introNext() {
  if (introCurrentStep < INTRO_TOTAL_STEPS) {
    introCurrentStep++;
    renderIntroStep();
  }
}

function introPrev() {
  if (introCurrentStep > 1) {
    introCurrentStep--;
    renderIntroStep();
  }
}

function markIntroSeen() {
  try { localStorage.setItem("llm_pool_intro_seen", "1"); } catch {}
  try { sessionStorage.setItem("llm_pool_intro_seen", "1"); } catch {}
}

function introFinish() {
  markIntroSeen();
  hideIntroModal();
}

function introSkip() {
  markIntroSeen();
  hideIntroModal();
}

// Show intro on first visit
function maybeShowIntro() {
  let seen = false;
  try {
    seen = localStorage.getItem("llm_pool_intro_seen") === "1";
  } catch {
    // localStorage blocked (sandbox/cross-origin) — use sessionStorage fallback
    try {
      seen = sessionStorage.getItem("llm_pool_intro_seen") === "1";
    } catch {}
  }
  if (!seen) {
    showIntroModal();
  }
}

// Intro wizard button bindings
document.getElementById("intro-btn-next")?.addEventListener("click", introNext);
document.getElementById("intro-btn-prev")?.addEventListener("click", introPrev);
document.getElementById("intro-btn-finish")?.addEventListener("click", introFinish);
document.getElementById("intro-btn-skip")?.addEventListener("click", introSkip);
document.getElementById("intro-modal")?.addEventListener("click", (e) => {
  if (e.target.id === "intro-modal") introSkip();
});

// Progress dot click → jump to step
document.querySelectorAll(".intro-dot").forEach((dot, idx) => {
  dot.addEventListener("click", () => {
    introCurrentStep = idx + 1;
    renderIntroStep();
  });
});

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  const modal = document.getElementById("intro-modal");
  if (!modal || modal.style.display === "none") return;
  if (e.key === "ArrowRight") introNext();
  else if (e.key === "ArrowLeft") introPrev();
  else if (e.key === "Escape") introSkip();
});

// Step 1: import button inside intro
document.getElementById("intro-import-btn")?.addEventListener("click", async () => {
  const btn = document.getElementById("intro-import-btn");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "⏳ 导入中...";
  try {
    const res = await apiPost("config/import_providers", {});
    if (res.ok) {
      btn.innerHTML = `✅ 已导入 ${res.imported_count} 个 Provider`;
      btn.classList.remove("btn-primary");
      btn.classList.add("btn-secondary");
      // Reload config in background
      await loadConfig();
    } else {
      btn.innerHTML = `⚠️ ${res.message}`;
      btn.classList.remove("btn-primary");
      btn.classList.add("btn-warning");
    }
  } catch (err) {
    btn.innerHTML = `❌ 导入失败：${err.message}`;
    btn.classList.remove("btn-primary");
    btn.classList.add("btn-danger");
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btn.innerHTML = origText;
      btn.classList.remove("btn-secondary", "btn-warning", "btn-danger");
      btn.classList.add("btn-primary");
    }, 3000);
  }
});

// Management tab: show intro button
document.getElementById("btn-show-intro")?.addEventListener("click", showIntroModal);

window.switchTabToNodeLogs = function(nodeId) {
  // Set filter in logs tab and switch to it
  document.getElementById("log-node-filter").value = nodeId;
  document.getElementById("log-start-time").value = "";
  document.getElementById("log-end-time").value = "";
  document.getElementById("log-keyword-filter").value = "";
  currentLogPage = 1;
  const logsTabBtn = document.querySelector('.tab-btn[data-tab="tab-logs"]');
  logsTabBtn?.click();
};

async function apiGet(endpoint, params = {}) {
  if (bridge && typeof bridge.apiGet === "function") {
    return await bridge.apiGet(endpoint, params);
  }
  // Standalone dev fallback
  const pName = getPluginName();
  const query = new URLSearchParams(params).toString();
  const url = `/api/v1/plugins/extensions/${encodeURIComponent(pName)}/${endpoint}${query ? "?" + query : ""}`;
  const res = await fetch(url);
  const json = await res.json();
  if (json?.status === "error") {
    throw new Error(json.message || "Request failed");
  }
  return json?.data ?? json;
}

async function apiPost(endpoint, body = {}) {
  if (bridge && typeof bridge.apiPost === "function") {
    return await bridge.apiPost(endpoint, body);
  }
  // Standalone dev fallback
  const pName = getPluginName();
  const url = `/api/v1/plugins/extensions/${encodeURIComponent(pName)}/${endpoint}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (json?.status === "error") {
    throw new Error(json.message || "Request failed");
  }
  return json?.data ?? json;
}

// Load statistics and circuit breaker status
async function loadStats() {
  try {
    const data = await apiGet("stats", { window_seconds: 86400 });
    const summary = data.summary || {};
    document.getElementById("kpi-total").textContent = summary.total_requests || 0;
    document.getElementById("kpi-success-sub").textContent = `成功: ${summary.success_requests || 0} | 失败: ${summary.failed_requests || 0}`;
    document.getElementById("kpi-avail").textContent = `${summary.availability_pct || 100}%`;
    document.getElementById("kpi-latency").textContent = `${summary.avg_latency_ms || 0} ms`;
    document.getElementById("kpi-ttft").textContent = `${summary.avg_ttft_ms || 0} ms`;

    // Render Active Group & Promotion Status
    renderRoutingStatus(data.routing_status || {});

    // Render pool selection warning
    renderPoolSelectionWarning(data.pool_selection || {});

    // Render node status matrix
    const tbody = document.getElementById("nodes-status-tbody");
    tbody.innerHTML = "";
    const nodes = data.nodes || [];
    const circuitMap = data.circuit_status || {};

    // Collect unique model groups from nodes
    const nodeGroups = new Set();
    nodes.forEach((n) => {
      if (n.group_name) nodeGroups.add(n.group_name);
    });
    renderNodeGroupFilter(nodeGroups);

    // Filter nodes by selected groups
    const filteredNodes = nodeGroupFilter.size > 0
      ? nodes.filter((n) => nodeGroupFilter.has(n.group_name || "default"))
      : nodes;

    if (filteredNodes.length === 0) {
      const msg = nodes.length > 0 ? "当前筛选条件下无匹配节点" : "近 24 小时暂无调用记录";
      tbody.innerHTML = `<tr><td colspan="10" class="text-center" style="color:var(--text-muted);padding:1.5rem;">${msg}</td></tr>`;
    } else {
      filteredNodes.forEach((node) => {
        allDiscoveredNodes.add(node.node_id);
        const cst = circuitMap[node.node_id] || { status: "CLOSED", is_open: false, cooldown_remaining_seconds: 0 };
        let statusBadge;
        if (cst.status === "DEEP-SLEEP" || (cst.is_deep_sleep && cst.cooldown_remaining_seconds > 0)) {
          statusBadge = `<span class="badge" style="background:#7c3aed;color:#fff;">DEEP-SLEEP (深休眠)</span>`;
        } else if (cst.status === "OPEN" || (cst.is_open && cst.cooldown_remaining_seconds > 0)) {
          statusBadge = `<span class="badge badge-danger">OPEN (熔断)</span>`;
        } else if (cst.status === "HALF-OPEN") {
          statusBadge = `<span class="badge badge-warning">HALF-OPEN (探活)</span>`;
        } else {
          statusBadge = `<span class="badge badge-success">HEALTHY</span>`;
        }

        const cooldownText = cst.cooldown_remaining_seconds > 0 ? `${cst.cooldown_remaining_seconds}s` : "-";

        let reasonHtml = "";
        const tripReason = cst.trip_reason || cst.last_error_category || "";
        const tripErr = cst.trip_error_msg || cst.last_error_msg || "";
        if (tripReason || tripErr) {
          let catTag = "";
          if (tripReason.includes("欠费") || tripReason.includes("Quota") || cst.last_error_category === "InsufficientQuotaError") {
            catTag = `<span class="badge" style="background:#dc2626;color:#fff;font-size:0.7rem;margin-top:2px;">🔴 欠费/额度耗尽</span>`;
          } else if (tripReason.includes("429") || tripReason.includes("Rate") || cst.last_error_category === "RateLimitError") {
            catTag = `<span class="badge" style="background:#d97706;color:#fff;font-size:0.7rem;margin-top:2px;">🟡 429 限频</span>`;
          } else if (tripReason.includes("渠道") || tripReason.includes("Dead") || cst.last_error_category === "UpstreamDeadError") {
            catTag = `<span class="badge" style="background:#7c3aed;color:#fff;font-size:0.7rem;margin-top:2px;">💀 渠道故障</span>`;
          } else if (tripReason.includes("超时") || tripReason.includes("Timeout") || cst.last_error_category === "TimeoutError") {
            catTag = `<span class="badge" style="background:#e11d48;color:#fff;font-size:0.7rem;margin-top:2px;">⏱️ 超时超限</span>`;
          } else if (tripReason) {
            catTag = `<span class="badge" style="background:#4b5563;color:#fff;font-size:0.7rem;margin-top:2px;">${escapeHtml(tripReason)}</span>`;
          }
          const escapedErr = escapeHtml(tripErr);
          reasonHtml = `
            <div style="margin-top:4px;" title="${escapedErr}">
              ${catTag}
              ${tripErr ? `<div style="font-size:0.7rem;color:var(--text-muted);max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:help;" title="${escapedErr}">${escapedErr}</div>` : ""}
            </div>
          `;
        }

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>
            <strong>${escapeHtml(node.node_id)}</strong>
            <div style="font-size:0.75rem;color:var(--text-muted);">${escapeHtml(node.model_name || "")}</div>
          </td>
          <td><span class="badge badge-info">${escapeHtml(node.group_name || "default")}</span></td>
          <td>${statusBadge}${reasonHtml}</td>
          <td>${cooldownText}</td>
          <td>${cst.active_requests || 0}</td>
          <td>
            <a href="javascript:void(0)" onclick="switchTabToNodeLogs('${escapeHtml(node.node_id)}')" style="color:var(--primary);text-decoration:none;font-weight:700;" title="点击查看该模型近24小时请求流水">
              ${node.total || 0} 次 🔍
            </a>
          </td>
          <td><strong>${node.availability_pct}%</strong></td>
          <td>${node.avg_latency_ms} ms</td>
          <td>${node.avg_ttft_ms ? node.avg_ttft_ms + " ms" : "-"}</td>
          <td>
            <div style="display:flex;gap:4px;">
              <button class="btn btn-secondary btn-sm" onclick="switchTabToNodeLogs('${escapeHtml(node.node_id)}')">流水</button>
              <button class="btn btn-secondary btn-sm" onclick="resetSingleCircuit('${escapeHtml(node.node_id)}')">解封</button>
            </div>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    updateLogNodeFilterOptions();
    loadHourlyProfile();
    loadWeeklyHeatmap();
    loadKeyFleet();
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

// Load 24-hour profile bars
async function loadHourlyProfile() {
  try {
    const data = await apiGet("hourly", { days: 7 });
    const profile = data.profile || [];
    const container = document.getElementById("hourly-bars");
    container.innerHTML = "";

    profile.forEach((item, index) => {
      const wrap = document.createElement("div");
      wrap.className = "hourly-bar-wrap";

      const bar = document.createElement("div");
      bar.className = "hourly-bar";

      // Prevent tooltip from being clipped at horizontal edges
      if (index <= 3) {
        bar.classList.add("tooltip-left");
      } else if (index >= 20) {
        bar.classList.add("tooltip-right");
      }

      const heightPct = Math.max(12, item.availability_pct);
      bar.style.height = `${heightPct}%`;

      // 85% and above is marked green as requested
      if (item.total === 0) {
        bar.style.background = "var(--border-color)";
      } else if (item.availability_pct >= 85) {
        bar.style.background = "var(--success)";
      } else if (item.availability_pct >= 65) {
        bar.style.background = "var(--warning)";
      } else {
        bar.style.background = "var(--danger)";
      }

      bar.dataset.tooltip = `${item.hour}:00 ~ ${item.hour}:59\n可用率: ${item.availability_pct}%\n平均耗时: ${item.avg_latency_ms} ms\n调用量: ${item.total} 次`;

      const label = document.createElement("div");
      label.className = "hourly-label";
      label.textContent = item.hour;

      wrap.appendChild(bar);
      wrap.appendChild(label);
      container.appendChild(wrap);
    });
  } catch (err) {
    console.error("Failed to load hourly profile:", err);
  }
}

function renderNodeGroupFilter(groups) {
  const bar = document.getElementById("node-group-filter-bar");
  const chipsContainer = document.getElementById("node-group-filter-chips");
  const clearBtn = document.getElementById("btn-clear-node-group-filter");
  if (!bar || !chipsContainer || !clearBtn) return;

  if (groups.size <= 1) {
    bar.style.display = "none";
    return;
  }
  bar.style.display = "flex";

  // Remove groups no longer present in data
  for (const g of nodeGroupFilter) {
    if (!groups.has(g)) nodeGroupFilter.delete(g);
  }

  chipsContainer.innerHTML = "";
  Array.from(groups).sort().forEach((grp) => {
    const chip = document.createElement("span");
    chip.className = "group-chip";
    if (nodeGroupFilter.has(grp)) chip.classList.add("active");
    chip.textContent = grp;
    chip.addEventListener("click", () => {
      if (nodeGroupFilter.has(grp)) {
        nodeGroupFilter.delete(grp);
      } else {
        nodeGroupFilter.add(grp);
      }
      clearBtn.style.display = nodeGroupFilter.size > 0 ? "" : "none";
      loadStats();
    });
    chipsContainer.appendChild(chip);
  });

  clearBtn.style.display = nodeGroupFilter.size > 0 ? "" : "none";
}

function updateLogNodeFilterOptions() {
  const select = document.getElementById("log-node-filter");
  if (!select) return;
  const currentVal = select.value;
  select.innerHTML = '<option value="">全部模型/节点</option>';
  Array.from(allDiscoveredNodes).sort().forEach((nodeId) => {
    const opt = document.createElement("option");
    opt.value = nodeId;
    opt.textContent = nodeId;
    if (nodeId === currentVal) opt.selected = true;
    select.appendChild(opt);
  });
}

function renderRoutingStatus(routing) {
  const activeNameEl = document.getElementById("current-active-group-name");
  const badgeEl = document.getElementById("badge-promotion-status");
  const defaultTextEl = document.getElementById("active-group-default-text");
  const chainTextEl = document.getElementById("active-group-chain-text");
  const selectEl = document.getElementById("select-active-group");
  const cardEl = document.querySelector(".active-group-card");

  if (!activeNameEl || !badgeEl || !defaultTextEl || !chainTextEl) return;

  // Hide the entire active group card if there's only 0 or 1 group
  const availableGroups = Array.isArray(routing.available_groups) ? routing.available_groups : [];
  if (availableGroups.length <= 1) {
    if (cardEl) cardEl.style.display = "none";
    return;
  }
  if (cardEl) cardEl.style.display = "";

  const activeGroup = routing.active_group || routing.default_group || "-";
  activeNameEl.textContent = activeGroup;

  if (routing.is_manual_override) {
    badgeEl.className = "badge badge-warning";
    badgeEl.textContent = "📌 手动指定 (Manual Override)";
  } else if (routing.is_promoted) {
    badgeEl.className = "badge badge-danger";
    badgeEl.textContent = "⚡ 忙时已自动提权 (Congestion Promoted)";
  } else if (routing.is_congested) {
    badgeEl.className = "badge badge-warning";
    badgeEl.textContent = "⚠️ 主组拥堵中";
  } else {
    badgeEl.className = "badge badge-success";
    badgeEl.textContent = "🟢 常态运行 (Auto Normal)";
  }

  defaultTextEl.textContent = `默认主组: ${routing.default_group || "-"}`;
  const chainList = Array.isArray(routing.active_chain) && routing.active_chain.length > 0
    ? routing.active_chain.join(" ➔ ")
    : "-";
  chainTextEl.textContent = `当前降级链路: ${chainList}`;

  // Update select options
  if (selectEl) {
    const available = Array.isArray(routing.available_groups) ? routing.available_groups : [];
    selectEl.innerHTML = '<option value="">自动调度 (跟随策略/提权)</option>';
    available.forEach((grp) => {
      const opt = document.createElement("option");
      opt.value = grp;
      opt.textContent = `模型组: ${grp}`;
      if (routing.manual_active_group === grp) {
        opt.selected = true;
      }
      selectEl.appendChild(opt);
    });
    if (!routing.manual_active_group) {
      selectEl.value = "";
    }
  }
}

function renderPoolSelectionWarning(selection) {
  // Remove existing warning
  const existing = document.getElementById("pool-selection-warning");
  if (existing) existing.remove();

  if (selection.is_chat_model) return; // All good, no warning needed

  // Build warning message
  let msg = "";
  let type = "warning";
  if (selection.is_fallback_model) {
    msg = "⚠️ 模型池已被选为回退对话模型。模型池仅在「对话模型」以及前序所有回退模型都失败后才会被调用。如需让模型池作为主力，请在 AstrBot 配置文件中将其设为「对话模型」。";
    type = "warning";
  } else {
    msg = "⚠️ 模型池（llm-pool）尚未被选为对话模型或回退对话模型。请在 AstrBot 管理面板的「配置文件」中将 llm-pool 设为对话模型，否则模型池不会生效。";
    type = "danger";
  }

  const kpiGrid = document.querySelector("#tab-overview .kpi-grid");
  if (!kpiGrid) return;

  const warning = document.createElement("div");
  warning.id = "pool-selection-warning";
  warning.className = `pool-selection-warning pool-selection-${type}`;
  warning.innerHTML = `<span>${escapeHtml(msg)}</span>`;
  kpiGrid.parentNode.insertBefore(warning, kpiGrid);
}

// Set / Reset Active Group
document.getElementById("btn-set-active-group")?.addEventListener("click", async () => {
  const selectEl = document.getElementById("select-active-group");
  const selectedGroup = selectEl ? selectEl.value : "";
  if (!selectedGroup) {
    showToast("请先在下拉框选择目标模型组，或直接点击重置为自动", "warning");
    return;
  }
  try {
    const res = await apiPost("routing/set_active_group", {
      action: "set",
      group_name: selectedGroup,
    });
    showToast(res.message || `已将活跃组切换为：${selectedGroup}`, "success");
    await loadStats();
  } catch (err) {
    showToast(`设置活跃组失败: ${err.message}`, "error");
  }
});

document.getElementById("btn-reset-active-group")?.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "重置模型组为自动模式",
    message: "确认清除手动指定的活跃组，并将当前模型组的拥堵提权状态重置为常态吗？（所有已熔断/冷却中的节点将严格保持封锁状态）",
    okText: "确认重置",
    okClass: "btn-primary",
  });
  if (!confirmed) return;
  try {
    const res = await apiPost("routing/set_active_group", { action: "reset" });
    showToast(res.message || "已恢复为自动调度模式并重置模型组提权状态", "success");
    await loadStats();
  } catch (err) {
    showToast(`恢复自动调度失败: ${err.message}`, "error");
  }
});

// Load Cost & Tier Spillover Analytics
async function loadCostAnalysis() {
  try {
    const windowSeconds = getCostWindowSeconds(currentCostRange);
    const rangeLabel = getCostRangeLabel(currentCostRange);
    const data = await apiGet("stats/cost_analysis", { window_seconds: windowSeconds });

    const primaryRate = data.primary_rate ?? data.free_rate ?? 0;
    const primaryName = data.primary_label || "主力层";
    const primaryReqs = data.primary_requests ?? data.free_requests ?? 0;

    const leakageRate = data.leakage_rate ?? 0;
    const terminalName = data.terminal_label || "保底层";
    const terminalReqs = data.terminal_requests ?? data.official_requests ?? 0;

    const bufferReqs = data.buffer_requests ?? ((data.fallback_requests || 0) + (data.relay_requests || 0));
    const bufferName = data.buffer_label || "中转后备";

    const dedicatedReqs = data.dedicated_requests ?? data.relay_requests ?? 0;
    const dedicatedRate = data.dedicated_rate ?? data.relay_rate ?? 0;
    const dedicatedNames = (data.funnel_tiers || []).filter((t) => t.role === "dedicated").map((t) => t.group_name);
    const dedicatedLabel = dedicatedNames.length > 0 ? dedicatedNames.join(", ") : "专用轻量层";

    // 1. KPIs — update titles to reflect selected range
    const freeTitleEl = document.getElementById("cost-kpi-free-title");
    if (freeTitleEl) freeTitleEl.textContent = `${rangeLabel}主力承接率`;
    document.getElementById("cost-kpi-free-rate").textContent = `${primaryRate}%`;
    document.getElementById("cost-kpi-free-sub").textContent = `${primaryName}: ${primaryReqs} 次`;
    document.getElementById("cost-kpi-leakage-rate").textContent = `${leakageRate}%`;
    document.getElementById("cost-kpi-official-sub").textContent = `${terminalName}: ${terminalReqs} 次`;
    document.getElementById("cost-kpi-buffer-reqs").textContent = `${bufferReqs}`;
    document.getElementById("cost-kpi-buffer-sub").textContent = `后备承接: ${bufferReqs} 次`;
    document.getElementById("cost-kpi-dedicated-reqs").textContent = `${dedicatedReqs}`;
    document.getElementById("cost-kpi-dedicated-sub").textContent = `${dedicatedLabel}: ${dedicatedReqs} 次 (${dedicatedRate}%)`;

    // 2. Main Chain Funnel Bar (excludes dedicated tiers)
    const funnelBar = document.getElementById("cost-funnel-bar");
    const funnelLegend = document.getElementById("cost-funnel-legend");
    const funnelBadge = document.getElementById("cost-funnel-badge");
    if (funnelBadge) funnelBadge.textContent = `基于${rangeLabel}调用与动态梯队`;
    if (funnelBar && funnelLegend) {
      funnelBar.innerHTML = "";
      funnelLegend.innerHTML = "";

      const allTiers = data.funnel_tiers && data.funnel_tiers.length > 0
        ? data.funnel_tiers
        : [
            { name: `${primaryName} (主力)`, count: primaryReqs, rate: primaryRate, cls: "seg-free" },
            { name: `${bufferName} (后备)`, count: bufferReqs, rate: data.buffer_rate ?? data.fallback_rate ?? 0, cls: "seg-fallback" },
            { name: `${terminalName} (保底)`, count: terminalReqs, rate: leakageRate, cls: "seg-official" },
          ];

      // Filter out dedicated tiers from the main funnel
      const mainTiers = allTiers.filter((t) => t.role !== "dedicated");

      // Recalculate rates relative to main-chain total (excluding dedicated)
      const mainTotal = mainTiers.reduce((sum, t) => sum + (t.count || 0), 0);
      mainTiers.forEach((t) => {
        t.rate = mainTotal > 0 ? round1((t.count / mainTotal) * 100.0) : 0.0;
      });

      mainTiers.forEach((t) => {
        if (t.rate > 0) {
          const seg = document.createElement("div");
          seg.className = `funnel-seg ${t.cls || "seg-fallback"}`;
          seg.style.width = `${t.rate}%`;
          seg.title = `${t.name}: ${t.count} 次 (${t.rate}%)`;
          funnelBar.appendChild(seg);
        }

        const leg = document.createElement("div");
        leg.className = "funnel-legend-item";
        leg.innerHTML = `<span class="legend-dot ${t.cls || "seg-fallback"}"></span> <span>${escapeHtml(t.name)}: <strong>${t.count}</strong> 次 (<strong>${t.rate}%</strong>)</span>`;
        funnelLegend.appendChild(leg);
      });
    }

    // 2b. Dedicated Pool Funnel Bar (separate section)
    const dedicatedFunnelBar = document.getElementById("cost-dedicated-funnel-bar");
    const dedicatedFunnelLegend = document.getElementById("cost-dedicated-funnel-legend");
    const dedicatedBadge = document.getElementById("dedicated-funnel-badge");
    if (dedicatedFunnelBar && dedicatedFunnelLegend) {
      dedicatedFunnelBar.innerHTML = "";
      dedicatedFunnelLegend.innerHTML = "";

      const dedicatedTiers = (data.funnel_tiers || []).filter((t) => t.role === "dedicated");

      if (dedicatedTiers.length === 0) {
        dedicatedFunnelBar.style.display = "none";
        dedicatedFunnelLegend.innerHTML = '<div class="dedicated-empty-hint">暂无专用节点流量（专用节点独立于主链梯队，不参与向下击穿）</div>';
        if (dedicatedBadge) dedicatedBadge.textContent = "无专用节点流量";
      } else {
        dedicatedFunnelBar.style.display = "";
        if (dedicatedBadge) dedicatedBadge.textContent = `${dedicatedTiers.length} 个专用节点组 · 独立于主链梯队`;

        const dedicatedTotal = dedicatedTiers.reduce((sum, t) => sum + (t.count || 0), 0);
        dedicatedTiers.forEach((t) => {
          const tRate = dedicatedTotal > 0 ? round1((t.count / dedicatedTotal) * 100.0) : 0.0;
          if (tRate > 0) {
            const seg = document.createElement("div");
            seg.className = `funnel-seg ${t.cls || "seg-relay"}`;
            seg.style.width = `${tRate}%`;
            seg.title = `${t.name}: ${t.count} 次 (${tRate}%)`;
            dedicatedFunnelBar.appendChild(seg);
          }

          const leg = document.createElement("div");
          leg.className = "funnel-legend-item";
          leg.innerHTML = `<span class="legend-dot ${t.cls || "seg-relay"}"></span> <span>${escapeHtml(t.name)}: <strong>${t.count}</strong> 次 (<strong>${tRate}%</strong>)</span>`;
          dedicatedFunnelLegend.appendChild(leg);
        });
      }
    }

    // 3. Hourly Stacked Bars
    const hourlyTitleEl = document.getElementById("cost-hourly-title");
    if (hourlyTitleEl) hourlyTitleEl.textContent = `🕒 ${rangeLabel}各时段梯队成本溢出画像 (Hourly Tier Spillover)`;
    const stackedContainer = document.getElementById("cost-stacked-bars");
    if (stackedContainer) {
      stackedContainer.innerHTML = "";
      const hourly = data.hourly_tiers || [];
      const maxHourlyTotal = Math.max(1, ...hourly.map((h) => h.total));

      hourly.forEach((item, index) => {
        const wrap = document.createElement("div");
        wrap.className = "stacked-bar-wrap";

        const bar = document.createElement("div");
        bar.className = "stacked-bar";

        // Prevent tooltip clipping at horizontal edges
        if (index <= 3) {
          bar.classList.add("tooltip-left");
        } else if (index >= 20) {
          bar.classList.add("tooltip-right");
        }

        const barHeightPct = item.total === 0 ? 5 : Math.max(12, Math.round((item.total / maxHourlyTotal) * 100));
        bar.style.height = `${barHeightPct}%`;

        if (item.total > 0) {
          const segs = [
            { count: item.free, cls: "seg-free" },
            { count: item.fallback, cls: "seg-fallback" },
            { count: item.official, cls: "seg-official" },
            { count: item.relay, cls: "seg-relay" },
          ];
          segs.forEach((s) => {
            if (s.count > 0) {
              const segEl = document.createElement("div");
              segEl.className = `stacked-segment ${s.cls}`;
              segEl.style.height = `${(s.count / item.total) * 100}%`;
              bar.appendChild(segEl);
            }
          });
        }

        bar.dataset.tooltip = `${item.hour}:00 ~ ${item.hour}:59\n总调用: ${item.total} 次\n🟢 首发主力 (${primaryName}): ${item.free} (${item.free_pct}%)\n🔵 中转后备 (${bufferName}): ${item.fallback}\n🔴 末级保底 (${terminalName}): ${item.official}\n🟡 专用轻量: ${item.relay}`;

        const label = document.createElement("div");
        label.className = "hourly-label";
        label.textContent = item.hour;

        wrap.appendChild(bar);
        wrap.appendChild(label);
        stackedContainer.appendChild(wrap);
      });
    }

    // 4. Smart Insights
    const insightsList = document.getElementById("cost-insights-list");
    if (insightsList) {
      insightsList.innerHTML = "";
      const insights = data.insights || [];
      if (insights.length === 0) {
        insightsList.innerHTML = '<div class="insight-item insight-info">近 24 小时运行平稳，无异常溢出。</div>';
      } else {
        insights.forEach((ins) => {
          const div = document.createElement("div");
          div.className = `insight-item insight-${ins.type || "info"}`;
          div.innerHTML = `<strong>${escapeHtml(ins.title)}</strong><span>${escapeHtml(ins.desc)}</span>`;
          insightsList.appendChild(div);
        });
      }
    }

    // 5. Contributors & Culprits
    const contribList = document.getElementById("cost-top-contributors");
    if (contribList) {
      contribList.innerHTML = "";
      const contribs = data.top_contributors || [];
      if (contribs.length === 0) {
        contribList.innerHTML = '<li class="text-muted" style="padding:0.5rem;">暂无调用记录</li>';
      } else {
        contribs.forEach((c) => {
          const li = document.createElement("li");
          li.className = "ranking-list-item";
          li.innerHTML = `
            <div class="ranking-item-main">
              <strong>${escapeHtml(c.node_id)}</strong>
              <div style="font-size:0.7rem;color:var(--text-muted);">${escapeHtml(c.group_name)}</div>
            </div>
            <div class="ranking-item-stat text-success">${c.success} 次 (${c.avail_pct}%)</div>
          `;
          contribList.appendChild(li);
        });
      }
    }

    const culpritList = document.getElementById("cost-leakage-culprits");
    if (culpritList) {
      culpritList.innerHTML = "";
      const culprits = data.leakage_culprits || [];
      if (culprits.length === 0) {
        culpritList.innerHTML = '<li class="text-muted" style="padding:0.5rem;">暂无损耗节点</li>';
      } else {
        culprits.forEach((c) => {
          const li = document.createElement("li");
          li.className = "ranking-list-item";
          li.innerHTML = `
            <div class="ranking-item-main">
              <strong>${escapeHtml(c.node_id)}</strong>
              <div style="font-size:0.7rem;color:var(--text-muted);">${escapeHtml(c.group_name)} · 均耗: ${c.avg_latency_ms}ms</div>
            </div>
            <div class="ranking-item-stat text-danger">${c.failed} 次失败 (${c.fail_rate_pct}%)</div>
          `;
          culpritList.appendChild(li);
        });
      }
    }

    loadErrorAttribution();
    loadHedgingStats();
  } catch (err) {
    console.error("Failed to load cost analysis:", err);
  }
}

// Load logs
async function loadLogs(page = 1) {
  try {
    currentLogPage = page;
    const statusFilter = document.getElementById("log-status-filter").value;
    const nodeFilter = document.getElementById("log-node-filter").value;
    const keyword = document.getElementById("log-keyword-filter").value.trim();
    const startTimeVal = document.getElementById("log-start-time").value;
    const endTimeVal = document.getElementById("log-end-time").value;
    const badge = document.getElementById("log-current-filter-badge");

    // Convert datetime-local (local timezone) to epoch seconds
    let startTimeEpoch = null;
    let endTimeEpoch = null;
    if (startTimeVal) {
      startTimeEpoch = new Date(startTimeVal).getTime() / 1000;
    }
    if (endTimeVal) {
      endTimeEpoch = new Date(endTimeVal).getTime() / 1000;
    }

    const filterLabels = [];
    if (nodeFilter) filterLabels.push(`模型：${nodeFilter}`);
    if (statusFilter) filterLabels.push(`状态：${statusFilter}`);
    if (startTimeVal || endTimeVal) {
      filterLabels.push(`时间范围：${startTimeVal || "不限"} ~ ${endTimeVal || "现在"}`);
    }
    if (keyword) filterLabels.push(`关键词："${keyword}"`);

    if (filterLabels.length > 0) {
      badge.style.display = "inline-block";
      badge.textContent = `当前筛选：${filterLabels.join(" | ")}`;
    } else {
      badge.style.display = "none";
    }

    const params = {
      page: currentLogPage,
      page_size: LOG_PAGE_SIZE,
      status: statusFilter,
      node_id: nodeFilter,
      keyword: keyword,
    };
    if (startTimeEpoch) params.start_time = startTimeEpoch;
    if (endTimeEpoch) params.end_time = endTimeEpoch;
    if (!startTimeEpoch && !endTimeEpoch && !keyword && !nodeFilter) {
      // Default to 24h if no specific query
      params.window_seconds = 86400;
    }

    const data = await apiGet("logs", params);
    const tbody = document.getElementById("logs-tbody");
    tbody.innerHTML = "";
    const logs = data.logs || [];
    const total = data.total || 0;
    const totalPages = data.total_pages || 1;

    // Update pagination UI
    document.getElementById("pagination-info").textContent = `共 ${total} 条记录 · 第 ${currentLogPage} / ${totalPages} 页`;
    document.getElementById("pagination-current-page").textContent = currentLogPage;
    document.getElementById("btn-prev-page").disabled = currentLogPage <= 1;
    document.getElementById("btn-next-page").disabled = currentLogPage >= totalPages;

    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" class="text-center" style="color:var(--text-muted);padding:1.5rem;">未查询到匹配的流水记录</td></tr>`;
      return;
    }

    logs.forEach((log) => {
      const isOk = log.status === "SUCCESS";
      const statusBadge = isOk
        ? `<span class="badge badge-success">SUCCESS</span>`
        : `<span class="badge badge-danger">FAILED (${escapeHtml(log.error_type || "Error")})</span>`;

      const localTimeStr = formatLocalTime(log.timestamp);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${log.id}</td>
        <td style="font-size:0.8rem;white-space:nowrap;"><strong>${escapeHtml(localTimeStr)}</strong></td>
        <td><span class="badge badge-info">${escapeHtml(log.group_name || "-")}</span></td>
        <td><code>${escapeHtml(log.node_id)}</code></td>
        <td>${log.is_stream ? "Stream" : "Sync"}</td>
        <td>${log.ttft_ms ? log.ttft_ms + " ms" : "-"}</td>
        <td>${log.total_latency_ms} ms</td>
        <td>${statusBadge}</td>
        <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(log.error_msg || "")}">
          ${escapeHtml(log.error_msg || "-")}
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load logs:", err);
  }
}

// Load config
async function loadConfig() {
  try {
    const data = await apiGet("config/get");
    currentConfig = data.config || {};
    availableProviders = data.available_providers || [];

    document.getElementById("cfg-strategy").value = currentConfig.strategy || "usage-based-routing-v2";
    document.getElementById("cfg-max-attempts").value = currentConfig.max_attempts ?? 5;
    document.getElementById("cfg-failover").checked = currentConfig.failover !== false;
    document.getElementById("cfg-suppress-error").checked = currentConfig.suppress_error_raise !== false;
    document.getElementById("cfg-retry-after").value = currentConfig.retry_after_seconds ?? 0;
    document.getElementById("cfg-weighted-failover").checked = Boolean(currentConfig.enable_weighted_failover);
    document.getElementById("cfg-request-timeout").value = currentConfig.request_timeout_seconds ?? 50;
    document.getElementById("cfg-stream-timeout").value = currentConfig.stream_timeout ?? 26;
    document.getElementById("cfg-stall-timeout").value = currentConfig.chunk_stall_timeout ?? 10;
    document.getElementById("cfg-global-timeout").value = currentConfig.global_timeout_seconds ?? 120.0;

    document.getElementById("cfg-cooldown-seconds").value = currentConfig.cooldown_seconds ?? 1800;
    document.getElementById("cfg-ratelimit-cooldown").value = currentConfig.ratelimit_cooldown_seconds ?? 1800;
    document.getElementById("cfg-model-request-interval").value = currentConfig.model_request_interval_seconds ?? 3.0;
    document.getElementById("cfg-daily-reset-hour").value = currentConfig.daily_reset_hour ?? 2;
    document.getElementById("cfg-ratelimit-fails").value = currentConfig.ratelimit_allowed_fails ?? 0;
    document.getElementById("cfg-timeout-fails").value = currentConfig.timeout_allowed_fails ?? 2;
    document.getElementById("cfg-badrequest-fails").value = currentConfig.badrequest_allowed_fails ?? 2;
    document.getElementById("cfg-aggressive-fails").value = currentConfig.aggressive_allowed_fails ?? 5;

    document.getElementById("cfg-enable-hedging").checked = currentConfig.enable_hedging === true;
    document.getElementById("cfg-hedging-delay").value = currentConfig.hedging_delay_seconds ?? 24.0;
    document.getElementById("cfg-enable-congestion").checked = currentConfig.enable_congestion_promotion === true;
    document.getElementById("cfg-congestion-threshold").value = currentConfig.congestion_latency_threshold ?? 40.0;

    document.getElementById("cfg-api-server-enabled").checked = currentConfig.api_server_enabled === true;
    document.getElementById("cfg-api-server-port").value = currentConfig.api_server_port ?? 5120;
    document.getElementById("cfg-api-server-host").value = currentConfig.api_server_host || "0.0.0.0";
    document.getElementById("cfg-api-server-key").value = currentConfig.api_server_key || "";

    document.getElementById("cfg-verbose-logging").checked = Boolean(currentConfig.verbose_logging);

    // Render dedicated pools table
    renderDedicatedPools(currentConfig);

    // Render fallbacks and alias visual editors
    renderFallbacks(currentConfig);
    renderAlias(currentConfig);

    renderMembers(currentConfig.members || []);
    updateMemberCountBadge(currentConfig.members || []);
  } catch (err) {
    console.error("Failed to load config:", err);
  }
}

// Collapsible config card toggle
window.toggleConfigCard = function(headerEl) {
  const card = headerEl.closest(".config-card");
  if (!card) return;
  card.classList.toggle("collapsed");
};

function updateMemberCountBadge(members) {
  const badge = document.getElementById("member-count-badge");
  if (badge) {
    const count = (members || []).length;
    badge.textContent = `${count} 个成员`;
  }
}

function renderMembers(members) {
  const tbody = document.getElementById("members-editor-tbody");
  tbody.innerHTML = "";

  const knownPids = new Set(availableProviders.map((p) => p.provider_id));

  members.forEach((m, idx) => {
    const tr = document.createElement("tr");
    tr.className = "member-row";
    tr.draggable = true;
    tr.dataset.idx = idx;

    const rawPid = (m.provider_id || "").trim();
    const pid = escapeHtml(rawPid);
    const grp = escapeHtml(m.group_name || "default");

    // Build options ensuring the current member's PID is always available
    let customOptionHtml = "";
    if (rawPid && !knownPids.has(rawPid)) {
      customOptionHtml = `<option value="${pid}">${pid} (当前配置)</option>`;
    }

    const providerOpts = availableProviders.map(
      (p) => `<option value="${escapeHtml(p.provider_id)}">${escapeHtml(p.provider_id)}</option>`
    ).join("");

    tr.innerHTML = `
      <td class="td-drag" title="拖拽排序">⠿</td>
      <td style="text-align:center;"><input type="checkbox" class="m-enabled" ${m.enabled !== false ? "checked" : ""} /></td>
      <td>
        <select class="form-control m-pid">
          <option value="">— 选择 Provider —</option>
          ${customOptionHtml}
          ${providerOpts}
          <option value="__custom__">➕ 手动输入 Provider ID...</option>
        </select>
      </td>
      <td><input type="text" class="form-control m-group" value="${grp}" placeholder="tier1, light_llm_pool" /></td>
      <td style="white-space:nowrap;">
        <button class="btn btn-secondary btn-sm m-toggle-adv" title="展开/折叠高级参数">⚙️</button>
        <button class="btn btn-danger btn-sm" onclick="removeMember(${idx})">删除</button>
      </td>
    `;

    // Set selected provider after innerHTML
    const pidSelect = tr.querySelector(".m-pid");
    if (rawPid) {
      pidSelect.value = rawPid;
    }

    // Support typing custom provider ID
    pidSelect.addEventListener("change", () => {
      if (pidSelect.value === "__custom__") {
        const customVal = prompt("请输入 Provider ID（例如 amd/DeepSeek-V4.1-Flash）：");
        if (customVal && customVal.trim()) {
          const cleanVal = customVal.trim();
          const newOpt = document.createElement("option");
          newOpt.value = cleanVal;
          newOpt.textContent = cleanVal;
          pidSelect.insertBefore(newOpt, pidSelect.lastElementChild);
          pidSelect.value = cleanVal;
          knownPids.add(cleanVal);
        } else {
          pidSelect.value = rawPid || "";
        }
      }
    });

    // Build collapsible advanced row
    const advTr = document.createElement("tr");
    advTr.className = "member-adv-row";
    advTr.style.display = "none";
    advTr.innerHTML = `
      <td colspan="5" style="padding:0.5rem 1rem;background:rgba(0,0,0,0.02);">
        <div class="member-adv-grid">
          <label>⚡ 激进 <input type="checkbox" class="m-aggressive" ${m.aggressive_mode ? "checked" : ""} /></label>
          <label>超时(s) <input type="number" class="form-control m-timeout" value="${m.timeout_seconds || ""}" min="0" max="300" step="0.5" placeholder="默认" style="width:65px;" /></label>
          <label>冷却(s) <input type="number" class="form-control m-cooldown" value="${m.cooldown_seconds || ""}" min="0" max="86400" step="1" placeholder="默认" style="width:65px;" /></label>
          <label>权重 <input type="number" class="form-control m-weight" value="${m.weight || 1}" min="1" max="100" style="width:55px;" /></label>
          <label>并发 <input type="number" class="form-control m-parallel" value="${m.max_parallel_requests || 6}" min="1" max="50" style="width:55px;" /></label>
          <label>RPM <input type="number" class="form-control m-rpm" value="${m.rpm || 90}" min="1" max="9999" style="width:55px;" /></label>
          <label>视觉 <input type="checkbox" class="m-vision" ${m.supports_vision !== false ? "checked" : ""} /></label>
        </div>
      </td>
    `;

    // Toggle advanced row
    tr.querySelector(".m-toggle-adv").addEventListener("click", () => {
      advTr.style.display = advTr.style.display === "none" ? "" : "none";
    });

    // Drag-and-drop sorting
    tr.addEventListener("dragstart", (e) => {
      tr.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });
    tr.addEventListener("dragend", () => {
      tr.classList.remove("dragging");
      reorderMembers();
    });
    tr.addEventListener("dragover", (e) => {
      e.preventDefault();
      const dragging = tbody.querySelector(".dragging");
      if (!dragging || dragging === tr) return;
      const rect = tr.getBoundingClientRect();
      const after = e.clientY > rect.top + rect.height / 2;
      if (after) {
        tr.parentNode.insertBefore(dragging, tr.nextSibling);
      } else {
        tr.parentNode.insertBefore(dragging, tr);
      }
      // Also move the adv row
      const draggingAdv = dragging.nextElementSibling;
      if (draggingAdv && draggingAdv.classList.contains("member-adv-row")) {
        if (after) {
          tr.parentNode.insertBefore(draggingAdv, tr.nextSibling.nextSibling);
        } else {
          tr.parentNode.insertBefore(draggingAdv, tr.nextSibling);
        }
      }
    });

    tbody.appendChild(tr);
    tbody.appendChild(advTr);
  });
}

function reorderMembers() {
  // After drag, re-index removeMember calls
  const rows = document.querySelectorAll("#members-editor-tbody tr.member-row");
  rows.forEach((row, idx) => {
    const delBtn = row.querySelector("button[onclick^='removeMember']");
    if (delBtn) delBtn.setAttribute("onclick", `removeMember(${idx})`);
  });
}

// Add member
document.getElementById("btn-add-member").addEventListener("click", () => {
  const members = collectMembers();
  members.push({
    provider_id: "",
    group_name: "tier1",
    weight: 1,
    max_parallel_requests: 6,
    rpm: 90,
    supports_vision: true,
    enabled: true,
    aggressive_mode: false,
    timeout_seconds: 0,
    cooldown_seconds: 0,
  });
  renderMembers(members);
});

// One-click import AstrBot providers
document.getElementById("btn-import-providers")?.addEventListener("click", async () => {
  const btn = document.getElementById("btn-import-providers");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "⏳ 导入中...";
  try {
    const res = await apiPost("config/import_providers", {});
    if (res.ok) {
      showToast(`✅ ${res.message}`, "success", 4000);
      // Reload config to reflect new members
      await loadConfig();
    } else {
      showToast(`⚠️ ${res.message}`, "warning", 4000);
    }
  } catch (err) {
    showToast(`❌ 导入失败：${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = origText;
  }
});

window.removeMember = function(idx) {
  const members = collectMembers();
  members.splice(idx, 1);
  renderMembers(members);
};

// Add fallback rule button
document.getElementById("btn-add-fallback")?.addEventListener("click", () => {
  const tbody = document.getElementById("fallbacks-editor-tbody");
  if (!tbody) return;
  // Remove empty hint row if present
  if (tbody.querySelector('td[colspan]')) tbody.innerHTML = "";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><select class="form-control fb-source">${getGroupOptions("")}</select></td>
    <td style="text-align:center;color:var(--text-muted);">→</td>
    <td><select class="form-control fb-target">${getGroupOptions("")}</select></td>
    <td><button class="btn btn-danger btn-sm fb-remove">🗑️</button></td>
  `;
  tr.querySelector(".fb-remove").addEventListener("click", () => tr.remove());
  tbody.appendChild(tr);
});

// Add alias rule button
document.getElementById("btn-add-alias")?.addEventListener("click", () => {
  const tbody = document.getElementById("alias-editor-tbody");
  if (!tbody) return;
  if (tbody.querySelector('td[colspan]')) tbody.innerHTML = "";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input type="text" class="form-control al-alias" value="" placeholder="deepseek-chat" /></td>
    <td style="text-align:center;color:var(--text-muted);">→</td>
    <td><select class="form-control al-target">${getGroupOptions("")}</select></td>
    <td><button class="btn btn-danger btn-sm al-remove">🗑️</button></td>
  `;
  tr.querySelector(".al-remove").addEventListener("click", () => tr.remove());
  tbody.appendChild(tr);
});

// Add dedicated pool button
document.getElementById("btn-add-dedicated-pool")?.addEventListener("click", () => {
  const tbody = document.getElementById("dedicated-pools-tbody");
  const emptyHint = document.getElementById("dedicated-pools-empty");
  if (emptyHint) emptyHint.style.display = "none";
  if (tbody) tbody.style.display = "";

  const tr = document.createElement("tr");
  tr.className = "dedicated-pool-row";
  const strategyOptions = STRATEGY_OPTIONS.map(
    (opt) => `<option value="${opt.value}">${opt.label}</option>`
  ).join("");
  tr.innerHTML = `
    <td style="text-align:center;"><input type="checkbox" class="dp-enabled" checked /></td>
    <td><input type="text" class="form-control dp-id" value="" placeholder="my_dedicated_pool" /></td>
    <td><input type="text" class="form-control dp-group" value="" placeholder="my_dedicated_pool" /></td>
    <td><input type="text" class="form-control dp-model" value="" placeholder="my-pool-router" /></td>
    <td style="text-align:center;"><input type="checkbox" class="dp-strict" /></td>
    <td><select class="form-control dp-strategy">${strategyOptions}</select></td>
    <td><input type="number" class="form-control dp-timeout" value="0" min="0" max="300" step="1" style="width:60px;" /></td>
    <td><input type="number" class="form-control dp-stream-timeout" value="0" min="0" max="120" step="1" style="width:60px;" /></td>
    <td><input type="number" class="form-control dp-max-attempts" value="0" min="0" max="10" step="1" style="width:50px;" /></td>
    <td><button class="btn btn-danger btn-sm dp-remove">🗑️</button></td>
  `;
  tr.querySelector(".dp-remove").addEventListener("click", () => {
    tr.remove();
    updateDedicatedPoolsCount();
  });
  tbody.appendChild(tr);
  updateDedicatedPoolsCount();
});

function collectMembers() {
  const rows = document.querySelectorAll("#members-editor-tbody tr.member-row");
  const members = [];
  rows.forEach((row) => {
    const advRow = row.nextElementSibling;
    const adv = advRow && advRow.classList.contains("member-adv-row") ? advRow : null;
    members.push({
      __template_key: "provider",
      enabled: row.querySelector(".m-enabled").checked,
      provider_id: row.querySelector(".m-pid").value.trim(),
      group_name: row.querySelector(".m-group").value.trim(),
      aggressive_mode: adv ? adv.querySelector(".m-aggressive").checked : false,
      timeout_seconds: adv ? parseFloat(adv.querySelector(".m-timeout").value) || 0.0 : 0.0,
      cooldown_seconds: adv ? parseFloat(adv.querySelector(".m-cooldown").value) || 0.0 : 0.0,
      weight: adv ? parseInt(adv.querySelector(".m-weight").value, 10) || 1 : 1,
      max_parallel_requests: adv ? parseInt(adv.querySelector(".m-parallel").value, 10) || 6 : 6,
      rpm: adv ? parseInt(adv.querySelector(".m-rpm").value, 10) || 90 : 90,
      supports_vision: adv ? adv.querySelector(".m-vision").checked : true,
    });
  });
  return members;
}

// ─── Dedicated Pools Management ───

const STRATEGY_OPTIONS = [
  { value: "", label: "继承全局" },
  { value: "usage-based-routing-v2", label: "usage-based-routing-v2" },
  { value: "least_busy", label: "least_busy" },
  { value: "latency_based", label: "latency_based" },
  { value: "weighted_random", label: "weighted_random" },
  { value: "round_robin", label: "round_robin" },
  { value: "ordered_failover", label: "ordered_failover" },
];

function migrateLegacySecondaryPool(config) {
  return Array.isArray(config.dedicated_pools) ? config.dedicated_pools : [];
}

function renderDedicatedPools(config) {
  const tbody = document.getElementById("dedicated-pools-tbody");
  const emptyHint = document.getElementById("dedicated-pools-empty");
  const countBadge = document.getElementById("dedicated-pools-count");
  if (!tbody) return;

  const pools = migrateLegacySecondaryPool(config);
  tbody.innerHTML = "";

  if (pools.length === 0) {
    emptyHint.style.display = "";
    tbody.style.display = "none";
  } else {
    emptyHint.style.display = "none";
    tbody.style.display = "";
  }

  if (countBadge) countBadge.textContent = `${pools.length} 个专用池`;

  pools.forEach((pool, index) => {
    const tr = document.createElement("tr");
    tr.className = "dedicated-pool-row";

    const strategyOptions = STRATEGY_OPTIONS.map(
      (opt) => `<option value="${opt.value}" ${opt.value === (pool.strategy || "") ? "selected" : ""}>${opt.label}</option>`
    ).join("");

    tr.innerHTML = `
      <td style="text-align:center;"><input type="checkbox" class="dp-enabled" checked /></td>
      <td><input type="text" class="form-control dp-id" value="${escapeHtml(pool.id || "")}" placeholder="my_dedicated_pool" /></td>
      <td><input type="text" class="form-control dp-group" value="${escapeHtml(pool.group || "")}" placeholder="my_dedicated_pool" /></td>
      <td><input type="text" class="form-control dp-model" value="${escapeHtml(pool.model_name || "")}" placeholder="my-pool-router" /></td>
      <td style="text-align:center;"><input type="checkbox" class="dp-strict" ${pool.strict_group ? "checked" : ""} /></td>
      <td><select class="form-control dp-strategy">${strategyOptions}</select></td>
      <td><input type="number" class="form-control dp-timeout" value="${pool.request_timeout_seconds || 0}" min="0" max="300" step="1" style="width:60px;" /></td>
      <td><input type="number" class="form-control dp-stream-timeout" value="${pool.stream_timeout || 0}" min="0" max="120" step="1" style="width:60px;" /></td>
      <td><input type="number" class="form-control dp-max-attempts" value="${pool.max_attempts || 0}" min="0" max="10" step="1" style="width:50px;" /></td>
      <td><button class="btn btn-danger btn-sm dp-remove">🗑️</button></td>
    `;

    tr.querySelector(".dp-remove").addEventListener("click", () => {
      tr.remove();
      updateDedicatedPoolsCount();
    });

    tbody.appendChild(tr);
  });
}

function updateDedicatedPoolsCount() {
  const rows = document.querySelectorAll("#dedicated-pools-tbody tr");
  const countBadge = document.getElementById("dedicated-pools-count");
  const emptyHint = document.getElementById("dedicated-pools-empty");
  const tbody = document.getElementById("dedicated-pools-tbody");
  if (countBadge) countBadge.textContent = `${rows.length} 个专用池`;
  if (emptyHint) emptyHint.style.display = rows.length === 0 ? "" : "none";
  if (tbody) tbody.style.display = rows.length === 0 ? "none" : "";
}

function collectDedicatedPools() {
  const rows = document.querySelectorAll("#dedicated-pools-tbody tr");
  const pools = [];
  rows.forEach((row) => {
    const enabled = row.querySelector(".dp-enabled").checked;
    const id = row.querySelector(".dp-id").value.trim();
    if (!enabled || !id) return;
    pools.push({
      __template_key: "pool",
      id,
      group: row.querySelector(".dp-group").value.trim() || id,
      model_name: row.querySelector(".dp-model").value.trim() || `${id}-router`,
      strict_group: row.querySelector(".dp-strict").checked,
      strategy: row.querySelector(".dp-strategy").value,
      max_attempts: parseInt(row.querySelector(".dp-max-attempts").value, 10) || 0,
      request_timeout_seconds: parseFloat(row.querySelector(".dp-timeout").value) || 0,
      stream_timeout: parseFloat(row.querySelector(".dp-stream-timeout").value) || 0,
    });
  });
  return pools;
}

// ─── Fallbacks & Alias Visual Editors ───

function getGroupOptions(selected) {
  // Collect group names from current members + dedicated pools
  const groups = new Set();
  (currentConfig.members || []).forEach((m) => {
    if (m.group_name) {
      String(m.group_name).split(/[,;，；/|]+/).forEach((g) => {
        const trimmed = g.trim();
        if (trimmed) groups.add(trimmed);
      });
    }
  });
  (currentConfig.dedicated_pools || []).forEach((p) => {
    if (p.group) groups.add(p.group);
    if (p.id) groups.add(p.id);
  });
  // Also include groups from existing fallbacks/alias
  const fb = _normalizeFallbacksDict(currentConfig.fallbacks);
  Object.entries(fb).forEach(([src, tgts]) => {
    groups.add(src);
    tgts.forEach((v) => groups.add(v));
  });
  const alias = _normalizeAliasDict(currentConfig.model_group_alias);
  Object.keys(alias).forEach((k) => groups.add(k));
  Object.values(alias).forEach((v) => groups.add(v));

  const opts = Array.from(groups).sort();
  return `<option value="">— 选择/输入 —</option>` +
    opts.map((g) => `<option value="${escapeHtml(g)}" ${g === selected ? "selected" : ""}>${escapeHtml(g)}</option>`).join("");
}

function _normalizeFallbacksDict(raw) {
  if (Array.isArray(raw)) {
    const result = {};
    raw.forEach((item) => {
      if (!item || typeof item !== "object") return;
      const src = String(item.source_group || "").trim();
      const tgt = String(item.target_group || "").trim();
      if (src && tgt) {
        if (!result[src]) result[src] = [];
        result[src].push(tgt);
      }
    });
    return result;
  }
  if (raw && typeof raw === "object") return raw;
  return {};
}

function _normalizeAliasDict(raw) {
  if (Array.isArray(raw)) {
    const result = {};
    raw.forEach((item) => {
      if (!item || typeof item !== "object") return;
      const alias = String(item.alias_name || "").trim();
      const target = String(item.target_group || "").trim();
      if (alias && target) result[alias] = target;
    });
    return result;
  }
  if (raw && typeof raw === "object") return raw;
  return {};
}

function renderFallbacks(config) {
  const tbody = document.getElementById("fallbacks-editor-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const fbDict = _normalizeFallbacksDict(config.fallbacks);
  const rules = [];
  for (const [src, tgts] of Object.entries(fbDict)) {
    for (const tgt of tgts) {
      rules.push({ source_group: src, target_group: tgt });
    }
  }

  if (rules.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:0.75rem;">暂无回退规则，点击"添加回退规则"创建</td></tr>';
    return;
  }

  rules.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <select class="form-control fb-source">${getGroupOptions(r.source_group)}</select>
      </td>
      <td style="text-align:center;color:var(--text-muted);">→</td>
      <td>
        <select class="form-control fb-target">${getGroupOptions(r.target_group)}</select>
      </td>
      <td><button class="btn btn-danger btn-sm fb-remove">🗑️</button></td>
    `;
    tr.querySelector(".fb-remove").addEventListener("click", () => tr.remove());
    tbody.appendChild(tr);
  });
}

function collectFallbacks() {
  const rows = document.querySelectorAll("#fallbacks-editor-tbody tr");
  const dict = {};
  rows.forEach((row) => {
    const src = row.querySelector(".fb-source")?.value.trim();
    const tgt = row.querySelector(".fb-target")?.value.trim();
    if (src && tgt) {
      if (!dict[src]) dict[src] = [];
      dict[src].push(tgt);
    }
  });
  return dict;
}

function renderAlias(config) {
  const tbody = document.getElementById("alias-editor-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const aliasDict = _normalizeAliasDict(config.model_group_alias);
  const rules = Object.entries(aliasDict).map(([alias, target]) => ({ alias_name: alias, target_group: target }));

  if (rules.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:0.75rem;">暂无别名映射</td></tr>';
    return;
  }

  rules.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="text" class="form-control al-alias" value="${escapeHtml(r.alias_name)}" placeholder="deepseek-chat" /></td>
      <td style="text-align:center;color:var(--text-muted);">→</td>
      <td>
        <select class="form-control al-target">${getGroupOptions(r.target_group)}</select>
      </td>
      <td><button class="btn btn-danger btn-sm al-remove">🗑️</button></td>
    `;
    tr.querySelector(".al-remove").addEventListener("click", () => tr.remove());
    tbody.appendChild(tr);
  });
}

function collectAlias() {
  const rows = document.querySelectorAll("#alias-editor-tbody tr");
  const dict = {};
  rows.forEach((row) => {
    const alias = row.querySelector(".al-alias")?.value.trim();
    const target = row.querySelector(".al-target")?.value.trim();
    if (alias && target) dict[alias] = target;
  });
  return dict;
}

// Export config
document.getElementById("btn-export-config")?.addEventListener("click", async () => {
  try {
    const pName = getPluginName();
    const url = `/api/v1/plugins/extensions/${encodeURIComponent(pName)}/config/export`;
    // Use bridge if available, otherwise direct fetch for download
    if (bridge && typeof bridge.apiGet === "function") {
      const data = await bridge.apiGet("config/export", {});
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "llm_pool_config.json";
      a.click();
      URL.revokeObjectURL(a.href);
    } else {
      window.open(url, "_blank");
    }
    showToast("✅ 配置已导出", "success", 2000);
  } catch (err) {
    showToast(`❌ 导出失败：${err.message}`, "error");
  }
});

// Import config from file
document.getElementById("btn-import-config")?.addEventListener("click", () => {
  document.getElementById("import-config-file").click();
});

document.getElementById("import-config-file")?.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const config = JSON.parse(text);
    const confirmed = await showConfirm({
      title: "📥 导入配置",
      message: `确认从文件 "${file.name}" 导入配置吗？这将覆盖当前的成员列表、回退链、专用池等所有设置。`,
      okText: "确认导入",
      okClass: "btn-primary",
    });
    if (!confirmed) {
      e.target.value = "";
      return;
    }
    await apiPost("config/import_json", config);
    showToast("✅ 配置已导入，正在刷新...", "success");
    await loadConfig();
  } catch (err) {
    showToast(`❌ 导入失败：${err.message}`, "error");
  } finally {
    e.target.value = "";
  }
});

// Clear all members
document.getElementById("btn-clear-members")?.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "🗑️ 清空成员",
    message: "确认清空所有模型池成员吗？此操作不可撤销（可通过导出的配置恢复）。",
    okText: "确认清空",
    okClass: "btn-warning",
  });
  if (!confirmed) return;
  try {
    const res = await apiPost("config/clear_members", {});
    showToast(`✅ ${res.message}`, "success");
    await loadConfig();
  } catch (err) {
    showToast(`❌ 清空失败：${err.message}`, "error");
  }
});

// ─── Management Tab Buttons ───

// Management tab: export (reuse same logic)
document.getElementById("btn-manage-export")?.addEventListener("click", async () => {
  try {
    if (bridge && typeof bridge.apiGet === "function") {
      const data = await bridge.apiGet("config/export", {});
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "llm_pool_config.json";
      a.click();
      URL.revokeObjectURL(a.href);
    }
    showToast("✅ 配置已导出", "success", 2000);
  } catch (err) {
    showToast(`❌ 导出失败：${err.message}`, "error");
  }
});

// Management tab: import
document.getElementById("btn-manage-import")?.addEventListener("click", () => {
  document.getElementById("manage-import-file").click();
});

document.getElementById("manage-import-file")?.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const config = JSON.parse(text);
    const confirmed = await showConfirm({
      title: "📥 导入配置",
      message: `确认从文件 "${file.name}" 导入配置吗？这将覆盖当前的所有设置。`,
      okText: "确认导入",
      okClass: "btn-primary",
    });
    if (!confirmed) { e.target.value = ""; return; }
    await apiPost("config/import_json", config);
    showToast("✅ 配置已导入，正在刷新...", "success");
    await loadConfig();
  } catch (err) {
    showToast(`❌ 导入失败：${err.message}`, "error");
  } finally {
    e.target.value = "";
  }
});

// Management tab: clear members
document.getElementById("btn-manage-clear")?.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "🗑️ 清空成员",
    message: "确认清空所有模型池成员吗？此操作不可撤销（可通过导出的配置恢复）。",
    okText: "确认清空",
    okClass: "btn-warning",
  });
  if (!confirmed) return;
  try {
    const res = await apiPost("config/clear_members", {});
    showToast(`✅ ${res.message}`, "success");
    await loadConfig();
  } catch (err) {
    showToast(`❌ 清空失败：${err.message}`, "error");
  }
});

// Management tab: reset all config
document.getElementById("btn-manage-reset")?.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "⚠️ 重置所有配置",
    message: "确认将所有配置恢复为默认值吗？所有成员、回退链、专用池、调度参数都将被清空或重置。此操作不可撤销，请务必先导出配置。",
    okText: "确认重置",
    okClass: "btn-danger",
  });
  if (!confirmed) return;
  try {
    const res = await apiPost("config/reset_all", {});
    showToast(`✅ ${res.message}`, "success");
    await loadConfig();
  } catch (err) {
    showToast(`❌ 重置失败：${err.message}`, "error");
  }
});

// Save config
document.getElementById("btn-save-config").addEventListener("click", async () => {
  const fallbacks = collectFallbacks();
  const modelGroupAlias = collectAlias();

  const payload = {
    strategy: document.getElementById("cfg-strategy").value,
    max_attempts: parseInt(document.getElementById("cfg-max-attempts").value, 10) || 0,
    failover: document.getElementById("cfg-failover").checked,
    suppress_error_raise: document.getElementById("cfg-suppress-error").checked,
    retry_after_seconds: parseFloat(document.getElementById("cfg-retry-after").value) || 0.0,
    enable_weighted_failover: document.getElementById("cfg-weighted-failover").checked,
    request_timeout_seconds: parseInt(document.getElementById("cfg-request-timeout").value, 10) || 50,
    stream_timeout: parseInt(document.getElementById("cfg-stream-timeout").value, 10) || 26,
    chunk_stall_timeout: parseInt(document.getElementById("cfg-stall-timeout").value, 10) || 10,
    global_timeout_seconds: parseFloat(document.getElementById("cfg-global-timeout").value) || 120.0,
    cooldown_seconds: parseInt(document.getElementById("cfg-cooldown-seconds").value, 10) || 1800,
    ratelimit_cooldown_seconds: parseFloat(document.getElementById("cfg-ratelimit-cooldown").value) || 1800.0,
    model_request_interval_seconds: parseFloat(document.getElementById("cfg-model-request-interval").value) || 0.0,
    daily_reset_hour: parseInt(document.getElementById("cfg-daily-reset-hour").value, 10) || 2,
    ratelimit_allowed_fails: parseInt(document.getElementById("cfg-ratelimit-fails").value, 10) || 0,
    timeout_allowed_fails: parseInt(document.getElementById("cfg-timeout-fails").value, 10) || 2,
    badrequest_allowed_fails: parseInt(document.getElementById("cfg-badrequest-fails").value, 10) || 2,
    aggressive_allowed_fails: parseInt(document.getElementById("cfg-aggressive-fails").value, 10) || 5,
    enable_hedging: document.getElementById("cfg-enable-hedging").checked,
    hedging_delay_seconds: parseFloat(document.getElementById("cfg-hedging-delay").value) || 24.0,
    enable_congestion_promotion: document.getElementById("cfg-enable-congestion").checked,
    congestion_latency_threshold: parseFloat(document.getElementById("cfg-congestion-threshold").value) || 40.0,
    api_server_enabled: document.getElementById("cfg-api-server-enabled").checked,
    api_server_port: parseInt(document.getElementById("cfg-api-server-port").value, 10) || 5120,
    api_server_host: (document.getElementById("cfg-api-server-host").value || "0.0.0.0").trim(),
    api_server_key: (document.getElementById("cfg-api-server-key").value || "").trim(),
    dedicated_pools: collectDedicatedPools(),
    verbose_logging: document.getElementById("cfg-verbose-logging").checked,
    fallbacks,
    model_group_alias: modelGroupAlias,
    members: collectMembers(),
  };

  const saveBtn = document.getElementById("btn-save-config");
  const origText = saveBtn.innerHTML;
  saveBtn.disabled = true;
  saveBtn.innerHTML = "💾 保存中...";

  try {
    await apiPost("config/save", payload);
    showToast("✅ 配置已成功保存！", "success");
    loadStats();
  } catch (err) {
    showToast("❌ 保存失败：" + err.message, "error");
  } finally {
    saveBtn.disabled = false;
    saveBtn.innerHTML = origText;
  }
});

// Circuit reset
window.resetSingleCircuit = async function(nodeId) {
  try {
    await apiPost("circuit/reset", { node_id: nodeId });
    showToast(`✅ 节点 ${nodeId} 已成功解除熔断！`, "success");
    loadStats();
  } catch (err) {
    showToast("❌ 重置失败：" + err.message, "error");
  }
};

const btnResetAll = document.getElementById("btn-reset-all-circuit");
btnResetAll.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "⚡ 重置全部节点状态与计量",
    message: "确定要重置所有节点的熔断冷却状态与实时 QoS 延迟计量吗？所有节点的 EWMA 延迟与拥堵标记将重置为健康基线并立即解封。",
    okText: "立即重置",
    okClass: "btn-warning",
  });
  if (!confirmed) return;

  const originalText = btnResetAll.innerHTML;
  btnResetAll.disabled = true;
  btnResetAll.innerHTML = "⏳ 正在重置...";

  try {
    await apiPost("circuit/reset", {});
    showToast("✅ 已成功重置所有节点的熔断状态与实时计量数据！", "success");
    await loadStats();
  } catch (err) {
    showToast("❌ 重置失败：" + err.message, "error");
  } finally {
    btnResetAll.disabled = false;
    btnResetAll.innerHTML = originalText;
  }
});

document.getElementById("btn-refresh").addEventListener("click", async () => {
  const btn = document.getElementById("btn-refresh");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "⏳ 刷新中...";
  try {
    const activeTab = document.querySelector(".tab-btn.active")?.dataset.tab;
    if (activeTab === "tab-cost") {
      await loadCostAnalysis();
    } else {
      await loadStats();
    }
    showToast("数据已刷新", "info", 1500);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origText;
  }
});

// Cost analysis time range selector
document.querySelectorAll(".cost-range-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".cost-range-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentCostRange = btn.dataset.range;
    loadCostAnalysis();
  });
});

document.getElementById("log-status-filter").addEventListener("change", () => {
  loadLogs(1);
});

document.getElementById("log-node-filter").addEventListener("change", () => {
  loadLogs(1);
});

document.getElementById("btn-search-logs").addEventListener("click", () => {
  loadLogs(1);
});

document.getElementById("log-keyword-filter").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    loadLogs(1);
  }
});

document.getElementById("btn-clear-log-filters").addEventListener("click", () => {
  document.getElementById("log-status-filter").value = "";
  document.getElementById("log-node-filter").value = "";
  document.getElementById("log-start-time").value = "";
  document.getElementById("log-end-time").value = "";
  document.getElementById("log-keyword-filter").value = "";
  loadLogs(1);
});

document.getElementById("btn-clear-node-group-filter")?.addEventListener("click", () => {
  nodeGroupFilter.clear();
  document.getElementById("btn-clear-node-group-filter").style.display = "none";
  loadStats();
});

document.getElementById("btn-prev-page").addEventListener("click", () => {
  if (currentLogPage > 1) {
    loadLogs(currentLogPage - 1);
  }
});

document.getElementById("btn-next-page").addEventListener("click", () => {
  loadLogs(currentLogPage + 1);
});

// ─── Prompt & Payload Inspector (Real-time On-demand Capture) ───

let inspectorRecords = [];
let inspectorSelectedId = null;
let inspectorPaused = false;
let inspectorTimer = null;

function startInspector() {
  stopInspector();
  updateInspectorStatusBadge(true, inspectorPaused);
  fetchInspectorData();
  inspectorTimer = setInterval(fetchInspectorData, 4000);
}

function stopInspector() {
  if (inspectorTimer) {
    clearInterval(inspectorTimer);
    inspectorTimer = null;
  }
  updateInspectorStatusBadge(false, inspectorPaused);
}

document.addEventListener("visibilitychange", () => {
  const activeTab = document.querySelector(".tab-btn.active")?.dataset.tab;
  if (activeTab === "tab-inspector") {
    if (document.visibilityState === "visible") {
      startInspector();
    } else {
      stopInspector();
    }
  }
});

function updateInspectorStatusBadge(isActive, isPaused) {
  const badge = document.getElementById("inspector-status-badge");
  if (!badge) return;
  if (isPaused) {
    badge.className = "badge badge-warning";
    badge.textContent = "⏸️ 抓包已手动暂停";
  } else if (isActive) {
    badge.className = "badge badge-success";
    badge.textContent = "🟢 实时抓包中 (页面在线 · 内存保留100条)";
  } else {
    badge.className = "badge badge-secondary";
    badge.textContent = "⚪ 待机中 (页面切出自动休眠 · 0资源开销)";
  }
}

async function fetchInspectorData() {
  try {
    const res = await apiGet("inspector/records?limit=100");
    if (!res) return;
    const records = res.records || [];
    inspectorRecords = records;

    const status = res.status || {};
    updateInspectorStatusBadge(status.active && !status.paused, status.paused);

    renderInspectorList();

    // Auto-scroll / auto-select newest record if enabled or nothing is selected
    const autoScroll = document.getElementById("chk-inspector-autoscroll")?.checked;
    if (records.length > 0) {
      if (autoScroll || inspectorSelectedId === null) {
        const latest = records[records.length - 1];
        if (inspectorSelectedId !== latest.id) {
          inspectorSelectedId = latest.id;
          renderInspectorDetail(latest);
        }
      } else {
        const current = records.find((r) => r.id === inspectorSelectedId);
        if (current) {
          renderInspectorDetail(current);
        }
      }
    }
  } catch (err) {
    console.error("fetchInspectorData error:", err);
  }
}

function renderInspectorList() {
  const container = document.getElementById("inspector-records-list");
  if (!container) return;

  const keyword = (document.getElementById("inspector-filter-input")?.value || "").trim().toLowerCase();
  const filtered = inspectorRecords.filter((r) => {
    if (!keyword) return true;
    const promptStr = String(r.prompt || "").toLowerCase();
    const modelStr = String(r.model || "").toLowerCase();
    const nodeStr = String(r.hit_node || "").toLowerCase();
    const respStr = String(r.response_text || "").toLowerCase();
    return promptStr.includes(keyword) || modelStr.includes(keyword) || nodeStr.includes(keyword) || respStr.includes(keyword);
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">暂无符合条件的抓包记录。保持此页面打开并向 Bot 发送消息。</div>`;
    return;
  }

  // Render in reverse chronological order (newest first)
  const html = filtered
    .slice()
    .reverse()
    .map((r) => {
      const isSelected = r.id === inspectorSelectedId;
      const isSuccess = r.status === "SUCCESS";
      const statusClass = isSuccess ? "badge-success" : "badge-danger";
      const statusText = isSuccess ? "SUCCESS" : "FAILED";
      const streamBadge = r.is_stream ? `<span class="badge badge-info" style="font-size:0.65rem;">流式</span>` : `<span class="badge badge-secondary" style="font-size:0.65rem;">非流式</span>`;
      const promptSnippet = escapeHtml(r.prompt || (r.contexts && r.contexts.length > 0 ? "[多轮对话上下文]" : "[无提示词]"));

      return `
        <div class="inspector-item-card ${isSelected ? "active" : ""}" data-id="${r.id}">
          <div class="inspector-item-header">
            <span style="font-family:monospace;font-weight:600;">#${r.id} · ${escapeHtml(r.time_str.split(" ")[1] || r.time_str)}</span>
            <div class="inspector-item-meta">
              ${streamBadge}
              <span class="badge ${statusClass}" style="font-size:0.65rem;">${statusText}</span>
            </div>
          </div>
          <div class="inspector-item-snippet" title="${promptSnippet}">${promptSnippet}</div>
          <div class="inspector-item-node">
            <span style="color:var(--text-muted);">${escapeHtml(r.model || "llm-pool")}</span>
            <span style="font-family:monospace;font-weight:600;color:var(--primary);">${r.total_latency_ms} ms</span>
          </div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = html;

  container.querySelectorAll(".inspector-item-card").forEach((card) => {
    card.addEventListener("click", () => {
      const id = parseInt(card.dataset.id, 10);
      inspectorSelectedId = id;
      container.querySelectorAll(".inspector-item-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      const record = inspectorRecords.find((r) => r.id === id);
      if (record) {
        renderInspectorDetail(record);
      }
    });
  });
}

function renderInspectorDetail(record) {
  const panel = document.getElementById("inspector-detail-panel");
  if (!panel || !record) return;

  const isSuccess = record.status === "SUCCESS";
  const statusBadge = isSuccess
    ? `<span class="badge badge-success">调用成功 (SUCCESS)</span>`
    : `<span class="badge badge-danger">调用失败 (FAILED)</span>`;

  // 1. Meta Grid
  const metaHtml = `
    <div class="inspector-meta-grid">
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">记录 ID / 时间</span>
        <span class="inspector-meta-val">#${record.id} (${escapeHtml(record.time_str)})</span>
      </div>
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">请求模型 / 组</span>
        <span class="inspector-meta-val">${escapeHtml(record.model)} (${escapeHtml(record.group_name)})</span>
      </div>
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">命中执行节点</span>
        <span class="inspector-meta-val" style="color:var(--primary);">${escapeHtml(record.hit_node)}</span>
      </div>
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">传输模式</span>
        <span class="inspector-meta-val">${record.is_stream ? "⚡ 流式传输 (Stream)" : "📦 常规阻塞 (Non-stream)"}</span>
      </div>
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">总响应耗时</span>
        <span class="inspector-meta-val" style="color:var(--accent);">${record.total_latency_ms} ms</span>
      </div>
      ${record.ttft_ms !== null && record.ttft_ms !== undefined ? `
      <div class="inspector-meta-item">
        <span class="inspector-meta-label">首字延迟 (TTFT)</span>
        <span class="inspector-meta-val" style="color:var(--success);">${record.ttft_ms} ms</span>
      </div>
      ` : ""}
    </div>
  `;

  // 2. System Prompt Box
  let systemPromptHtml = "";
  if (record.system_prompt) {
    systemPromptHtml = `
      <div class="inspector-box-card">
        <div class="inspector-box-header">
          <span>⚙️ 系统人设 (System Prompt)</span>
          <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'sys')">📋 复制</button>
        </div>
        <div class="inspector-box-body" id="insp-content-sys">${escapeHtml(record.system_prompt)}</div>
      </div>
    `;
  }

  // 3. Multi-turn Contexts
  let contextsHtml = "";
  if (record.contexts && record.contexts.length > 0) {
    const bubbles = record.contexts.map((msg) => {
      const role = String(msg.role || "user").toLowerCase();
      let contentStr = "";
      let images = [];
      if (typeof msg.content === "string") {
        contentStr = msg.content;
      } else if (Array.isArray(msg.content)) {
        msg.content.forEach((part) => {
          if (typeof part === "string") contentStr += part;
          else if (part && part.text) contentStr += part.text;
          else if (part && part.image_url) images.push(part.image_url);
        });
      } else {
        contentStr = JSON.stringify(msg.content, null, 2);
      }

      const imgThumbs = images.map((u) => `<img class="inspector-img-thumb" src="${escapeHtml(u)}" alt="attachment" />`).join("");

      return `
        <div class="chat-bubble role-${escapeHtml(role)}">
          <div class="chat-bubble-role-badge">${escapeHtml(role)}</div>
          <div>${escapeHtml(contentStr)}</div>
          ${imgThumbs}
        </div>
      `;
    }).join("");

    contextsHtml = `
      <div class="inspector-box-card">
        <div class="inspector-box-header">
          <span>💬 对话历史上下文 (${record.contexts.length} 条消息)</span>
          <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'ctx')">📋 复制上下文 JSON</button>
        </div>
        <div class="inspector-box-body chat-bubble-stream">
          ${bubbles}
        </div>
      </div>
    `;
  }

  // 4. Current Prompt & Multimodal Images
  let imagesHtml = "";
  if (record.image_urls && record.image_urls.length > 0) {
    imagesHtml = `
      <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem;">
        ${record.image_urls.map((u) => `<img class="inspector-img-thumb" src="${escapeHtml(u)}" alt="image" />`).join("")}
      </div>
    `;
  }

  const promptHtml = `
    <div class="inspector-box-card">
      <div class="inspector-box-header">
        <span>📥 提交给 LLM 的本轮提示词 (Prompt)</span>
        <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'prompt')">📋 复制</button>
      </div>
      <div class="inspector-box-body" id="insp-content-prompt" style="font-weight:500;">
        ${escapeHtml(record.prompt || "[无单轮 prompt，通过 contexts 传输]")}
        ${imagesHtml}
      </div>
    </div>
  `;

  // 5. Response or Error
  let respHtml = "";
  if (isSuccess) {
    respHtml = `
      <div class="inspector-box-card" style="border-color:rgba(16,185,129,0.35);">
        <div class="inspector-box-header" style="background:rgba(16,185,129,0.06);">
          <span style="color:var(--success);">📤 LLM 返回的完整内容 (Completion Response)</span>
          <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'resp')">📋 复制回复</button>
        </div>
        <div class="inspector-box-body" id="insp-content-resp" style="background:rgba(16,185,129,0.02);">${escapeHtml(record.response_text || "[返回内容为空]")}</div>
      </div>
    `;
  } else {
    respHtml = `
      <div class="inspector-box-card" style="border-color:rgba(239,68,68,0.35);">
        <div class="inspector-box-header" style="background:rgba(239,68,68,0.06);">
          <span style="color:var(--danger);">⚠️ 错误详情 (Error Details)</span>
          <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'err')">📋 复制错误</button>
        </div>
        <div class="inspector-box-body" id="insp-content-err" style="color:var(--danger);font-family:monospace;background:rgba(239,68,68,0.02);">${escapeHtml(record.error_msg || "未知错误")}</div>
      </div>
    `;
  }

  // 6. Raw JSON Payload
  const rawJson = JSON.stringify(record, null, 2);
  const rawJsonHtml = `
    <details class="inspector-box-card" style="cursor:pointer;">
      <summary class="inspector-box-header" style="user-select:none;">
        <span>📦 完整结构化数据快照 (Raw JSON)</span>
        <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation();copyInspectorField(this, 'raw')">📋 复制全部 JSON</button>
      </summary>
      <div class="inspector-box-body" id="insp-content-raw" style="font-family:monospace;font-size:0.75rem;background:rgba(0,0,0,0.02);">${escapeHtml(rawJson)}</div>
    </details>
  `;

  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border-color);padding-bottom:0.75rem;">
      <h3 style="margin:0;font-size:1.05rem;">抓包详情 #${record.id}</h3>
      <div style="display:flex;align-items:center;gap:0.5rem;">
        ${statusBadge}
        <button class="btn btn-secondary btn-sm" onclick="copyInspectorField(this, 'raw')">📋 复制全部 JSON</button>
      </div>
    </div>
    ${metaHtml}
    ${systemPromptHtml}
    ${contextsHtml}
    ${promptHtml}
    ${respHtml}
    ${rawJsonHtml}
  `;
}

// Inspector Copy Handler
window.copyInspectorField = function(btn, field) {
  const record = inspectorRecords.find((r) => r.id === inspectorSelectedId);
  if (!record) return;

  let textToCopy = "";
  if (field === "sys") textToCopy = record.system_prompt || "";
  else if (field === "ctx") textToCopy = JSON.stringify(record.contexts || [], null, 2);
  else if (field === "prompt") textToCopy = record.prompt || "";
  else if (field === "resp") textToCopy = record.response_text || "";
  else if (field === "err") textToCopy = record.error_msg || "";
  else if (field === "raw") textToCopy = JSON.stringify(record, null, 2);

  if (!navigator.clipboard) {
    showToast("当前浏览器环境不支持一键复制", "warning");
    return;
  }

  navigator.clipboard.writeText(textToCopy).then(() => {
    const orig = btn.innerHTML;
    btn.innerHTML = "✅ 已复制";
    setTimeout(() => {
      btn.innerHTML = orig;
    }, 1500);
  }).catch(() => {
    showToast("复制失败，请手动选择复制", "error");
  });
};

// Inspector Event Listeners
document.getElementById("btn-inspector-pause")?.addEventListener("click", async () => {
  inspectorPaused = !inspectorPaused;
  const btn = document.getElementById("btn-inspector-pause");
  if (inspectorPaused) {
    btn.innerHTML = "▶️ 继续抓包";
    showToast("实时抓包已暂停", "info", 1500);
  } else {
    btn.innerHTML = "⏸️ 暂停抓包";
    showToast("实时抓包已恢复", "success", 1500);
  }
  updateInspectorStatusBadge(!inspectorPaused, inspectorPaused);
  try {
    await apiPost("inspector/heartbeat", { paused: inspectorPaused });
  } catch (err) {
    console.error("Failed to sync pause state:", err);
  }
});

document.getElementById("btn-inspector-clear")?.addEventListener("click", async () => {
  const confirmed = await showConfirm({
    title: "清空抓包记录",
    message: "确认清空内存中捕获的所有 Prompt 与回复记录？此操作不会影响已持久化的请求统计。",
    okText: "清空",
    okClass: "btn-danger",
  });
  if (!confirmed) return;

  try {
    await apiPost("inspector/clear", {});
    inspectorRecords = [];
    inspectorSelectedId = null;
    renderInspectorList();
    const panel = document.getElementById("inspector-detail-panel");
    if (panel) {
      panel.innerHTML = `<div class="empty-placeholder" style="margin-top:4rem;">👈 请从左侧列表中选择一条记录查看完整 Prompt、上下文与 LLM 回复</div>`;
    }
    showToast("抓包记录已清空", "success", 1500);
  } catch (err) {
    showToast("清空失败: " + err.message, "error");
  }
});

document.getElementById("btn-inspector-refresh")?.addEventListener("click", async () => {
  await fetchInspectorData();
  showToast("抓包列表已刷新", "info", 1200);
});

document.getElementById("inspector-filter-input")?.addEventListener("input", () => {
  renderInspectorList();
});

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// 7x24 Weekly Heatmap
async function loadWeeklyHeatmap() {
  try {
    const data = await apiGet("stats/heatmap", { days: 7 });
    const container = document.getElementById("weekly-heatmap-container");
    if (!container) return;
    container.innerHTML = "";

    const peakBadge = document.getElementById("heatmap-peak-badge");
    if (peakBadge && data.peak_weekday && data.peak_hour) {
      peakBadge.textContent = `⚡ 峰值时段: ${data.peak_weekday} ${data.peak_hour} (${data.max_cell_requests} 次)`;
    }

    // 1. Render hour header row (00h ~ 23h)
    const headerRow = document.createElement("div");
    headerRow.className = "heatmap-header-row";

    const emptyCorner = document.createElement("div");
    emptyCorner.className = "heatmap-row-label";
    emptyCorner.textContent = "时段";
    emptyCorner.style.fontSize = "0.7rem";
    emptyCorner.style.color = "var(--text-muted)";
    headerRow.appendChild(emptyCorner);

    for (let h = 0; h < 24; h++) {
      const hEl = document.createElement("div");
      hEl.className = "heatmap-header-cell";
      hEl.textContent = `${h.toString().padStart(2, "0")}`;
      hEl.title = `${h.toString().padStart(2, "0")}:00 ~ ${h.toString().padStart(2, "0")}:59`;
      headerRow.appendChild(hEl);
    }
    container.appendChild(headerRow);

    const matrix = data.matrix || [];
    matrix.forEach((row) => {
      const rowEl = document.createElement("div");
      rowEl.className = "heatmap-row";

      const labelEl = document.createElement("div");
      labelEl.className = "heatmap-row-label";
      labelEl.textContent = row.weekday_label;
      rowEl.appendChild(labelEl);

      row.hours.forEach((cell) => {
        const cellEl = document.createElement("div");
        cellEl.className = "heatmap-cell";

        const it = cell.intensity;
        if (cell.requests === 0) {
          cellEl.classList.add("hm-l0");
        } else if (it < 0.25) {
          cellEl.classList.add("hm-l1");
        } else if (it < 0.60) {
          cellEl.classList.add("hm-l2");
        } else if (it < 0.85) {
          cellEl.classList.add("hm-l3");
        } else {
          cellEl.classList.add("hm-l4");
        }

        const hourStr = cell.hour.toString().padStart(2, "0");
        const failText = cell.failures > 0 ? ` (失败 ${cell.failures} 次, 失败率 ${cell.fail_rate}%)` : "";
        cellEl.title = `${row.weekday_label} ${hourStr}:00 ~ ${hourStr}:59\n总请求: ${cell.requests} 次${failText}\n平均延迟: ${cell.avg_latency_ms} ms`;
        rowEl.appendChild(cellEl);
      });

      container.appendChild(rowEl);
    });
  } catch (err) {
    console.error("Failed to load heatmap:", err);
  }
}

// Multi-Key Fleet Health
async function loadKeyFleet() {
  try {
    const data = await apiGet("stats/key_fleet");
    document.getElementById("keyfleet-total").textContent = data.total_keys || 0;
    document.getElementById("keyfleet-healthy").textContent = data.healthy_keys || 0;
    document.getElementById("keyfleet-quarantined").textContent = data.quarantined_keys || 0;

    const badge = document.getElementById("keyfleet-health-badge");
    if (badge) {
      badge.textContent = `健康度 ${data.health_pct || 100}%`;
      badge.className = `badge ${data.quarantined_keys > 0 ? "badge-warning" : "badge-success"}`;
    }

    const container = document.getElementById("keyfleet-nodes-container");
    if (!container) return;
    container.innerHTML = "";

    const nodes = data.nodes || [];
    if (nodes.length === 0) {
      container.innerHTML = '<div class="text-muted" style="padding:0.5rem;font-size:0.85rem;">当前池成员暂未配置 API Key</div>';
      return;
    }

    nodes.forEach((n) => {
      const card = document.createElement("div");
      card.className = "keyfleet-node-card";

      const chipsHtml = (n.keys || []).map((k) => {
        const isQ = k.status === "QUARANTINED";
        const cls = isQ ? "key-chip key-quarantined" : "key-chip key-healthy";
        const icon = isQ ? "⏳" : "✓";
        const tip = isQ ? `临时隔离封禁中 (剩余 ${k.cooldown_seconds}s)` : "正常活跃就绪";
        return `<span class="${cls}" title="${tip}">${icon} ${escapeHtml(k.key_hash)}</span>`;
      }).join("");

      card.innerHTML = `
        <div class="keyfleet-node-header">
          <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%;">
            <span class="keyfleet-node-id" title="${escapeHtml(n.provider_id)}">${escapeHtml(n.provider_id)}</span>
            <div style="font-size:0.7rem;color:var(--text-muted);">${escapeHtml(n.group_name || "default")}</div>
          </div>
          <span class="badge ${n.quarantined_keys > 0 ? "badge-danger" : "badge-info"}" style="font-size:0.7rem;">${n.healthy_keys}/${n.total_keys} 可用</span>
        </div>
        <div class="keyfleet-chips">${chipsHtml}</div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load key fleet:", err);
  }
}

// Error Attribution
async function loadErrorAttribution() {
  try {
    const windowSeconds = getCostWindowSeconds(currentCostRange);
    const rangeLabel = getCostRangeLabel(currentCostRange);
    const data = await apiGet("stats/error_attribution", { window_seconds: windowSeconds });
    const badge = document.getElementById("error-attribution-total-badge");
    if (badge) {
      badge.textContent = `${rangeLabel}报错 ${data.total_errors || 0} 次`;
      badge.className = `badge ${(data.total_errors || 0) > 0 ? "badge-danger" : "badge-success"}`;
    }

    const catList = document.getElementById("error-categories-list");
    if (catList) {
      catList.innerHTML = "";
      const cats = data.categories || [];
      if (cats.length === 0) {
        catList.innerHTML = '<div class="text-muted" style="padding:0.75rem;font-size:0.85rem;color:var(--success);">✨ 近 24 小时零错误，运行极度健康！</div>';
      } else {
        cats.forEach((c) => {
          const item = document.createElement("div");
          item.className = "error-cat-item";
          item.innerHTML = `
            <div class="error-cat-header">
              <span>${c.icon} ${escapeHtml(c.name)}</span>
              <strong>${c.count} 次 (${c.percentage}%)</strong>
            </div>
            <div class="error-cat-bar-bg">
              <div class="error-cat-bar-fill" style="width:${c.percentage}%;background-color:${c.color};"></div>
            </div>
          `;
          catList.appendChild(item);
        });
      }
    }

    const rankList = document.getElementById("error-node-rankings");
    if (rankList) {
      rankList.innerHTML = "";
      const rankings = data.node_rankings || [];
      if (rankings.length === 0) {
        rankList.innerHTML = '<div class="text-muted" style="padding:0.5rem;font-size:0.85rem;">暂无节点报错记录</div>';
      } else {
        rankings.forEach((r) => {
          const row = document.createElement("div");
          row.className = "error-node-item";
          row.innerHTML = `
            <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:65%;">
              <strong>${escapeHtml(r.node_id)}</strong>
              <div style="font-size:0.7rem;color:var(--text-muted);">${r.primary_error_icon} 主因: ${escapeHtml(r.primary_error_name)}</div>
            </div>
            <span class="badge badge-danger">${r.total_errors} 次错误</span>
          `;
          rankList.appendChild(row);
        });
      }
    }
  } catch (err) {
    console.error("Failed to load error attribution:", err);
  }
}

// Speculative Hedging Stats
async function loadHedgingStats() {
  try {
    const windowSeconds = getCostWindowSeconds(currentCostRange);
    const data = await apiGet("stats/hedging", { window_seconds: windowSeconds });
    document.getElementById("hedge-kpi-races").textContent = `${data.total_races || 0} 次`;
    document.getElementById("hedge-kpi-overtake").textContent = `${data.overtake_rate || 0}%`;
    document.getElementById("hedge-kpi-saved").textContent = `${data.saved_seconds || 0}s`;

    const list = document.getElementById("hedge-events-list");
    if (!list) return;
    list.innerHTML = "";

    const events = data.recent_events || [];
    if (events.length === 0) {
      list.innerHTML = '<li class="text-muted" style="padding:0.5rem;font-size:0.85rem;">暂无对冲竞速触发记录</li>';
      return;
    }

    events.forEach((ev) => {
      const li = document.createElement("li");
      li.className = "ranking-list-item";
      const icon = ev.sibling_won ? "🏆" : "⚡";
      const statusClass = ev.sibling_won ? "text-success" : "text-primary";
      const resText = ev.sibling_won ? `备选反超抢答 (抢回 ${ev.saved_seconds}s)` : "主节点抢先完成";
      li.innerHTML = `
        <div class="ranking-item-main" style="max-width:75%;">
          <div>${icon} <strong>${escapeHtml(ev.primary_node)}</strong> vs ${escapeHtml(ev.sibling_node)}</div>
          <div style="font-size:0.7rem;color:var(--text-muted);">胜出者: ${escapeHtml(ev.winner_node)}</div>
        </div>
        <div class="ranking-item-stat ${statusClass}" style="font-size:0.75rem;">
          ${resText}
        </div>
      `;
      list.appendChild(li);
    });
  } catch (err) {
    console.error("Failed to load hedging stats:", err);
  }
}

// Init
async function init() {
  if (bridge && typeof bridge.ready === "function") {
    try {
      await Promise.race([
        bridge.ready(),
        new Promise((_, reject) => setTimeout(() => reject(new Error("bridge.ready timeout")), 3000)),
      ]);
    } catch (err) {
      console.warn("Bridge ready fallback:", err);
    }
  }
  loadStats();
  maybeShowIntro();
}

init();
