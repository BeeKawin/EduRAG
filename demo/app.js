const apiBaseKey = "ragaib-demo-api-base";

const $ = (id) => document.getElementById(id);

const form = $("evalForm");
const runButton = $("runButton");
const statusText = $("status");
const apiBaseUrl = $("apiBaseUrl");

function setStatus(text, busy = false) {
  statusText.textContent = text;
  runButton.disabled = busy;
  runButton.textContent = busy ? "Running..." : "Run Evaluation";
}

function valueOrNull(id) {
  const value = $(id).value.trim();
  return value ? value : null;
}

function cleanApiBaseUrl(value) {
  return (value || "").trim().replace(/\/+$/, "");
}

function liveQueryUrl() {
  const apiBase = cleanApiBaseUrl(apiBaseUrl.value);
  return `${apiBase}/eval/live-query`;
}

function metricCard(label, value) {
  const node = document.createElement("div");
  node.className = "metric";
  node.innerHTML = `<span>${label}</span><strong>${value ?? "-"}</strong>`;
  return node;
}

function renderMetrics(containerId, scores) {
  const container = $(containerId);
  container.innerHTML = "";
  [
    ["Correct", "correctness"],
    ["Ground", "groundedness"],
    ["Complete", "completeness"],
    ["Clear", "clarity"],
    ["Type", "type_alignment"],
  ].forEach(([label, key]) => container.appendChild(metricCard(label, scores?.[key])));
}

function bandText(scores) {
  return scores?.overall_band ? `${scores.overall_band}/10` : "-";
}

function winnerLabel(winner) {
  if (winner === "rag") return "RAG wins";
  if (winner === "baseline") return "Baseline wins";
  return "Tie";
}

function renderSources(sources) {
  const container = $("sources");
  const count = Array.isArray(sources) ? sources.length : 0;
  $("chunkCount").textContent = `${count} chunk${count === 1 ? "" : "s"}`;
  container.innerHTML = "";
  container.classList.toggle("empty", count === 0);

  if (!count) {
    container.textContent = "No retrieved chunks returned.";
    return;
  }

  const template = $("sourceTemplate");
  sources.forEach((source, index) => {
    const node = template.content.firstElementChild.cloneNode(true);
    const meta = [
      `#${index + 1}`,
      source.source || "unknown source",
      source.subject || "",
      source.grade || "",
      source.language || "",
    ]
      .filter(Boolean)
      .join(" · ");
    node.querySelector(".source-meta").textContent = meta;
    node.querySelector("h3").textContent =
      [source.topic, source.section || source.source_title].filter(Boolean).join(" > ") ||
      "Untitled chunk";
    node.querySelector("p").textContent = source.content || "No chunk text returned.";
    container.appendChild(node);
  });
}

function renderResult(result) {
  $("winnerBadge").textContent = winnerLabel(result.winner);
  $("winnerBadge").className = `badge winner-${result.winner || "tie"}`;
  $("comparisonText").textContent = result.comparison_rationale || "No comparison rationale returned.";
  $("comparisonText").classList.remove("empty");
  $("savedPath").textContent = result.saved_result_path ? `Saved to ${result.saved_result_path}` : "";

  $("ragAnswer").textContent = result.rag_answer || "No RAG answer returned.";
  $("baselineAnswer").textContent = result.baseline_answer || "No baseline answer returned.";
  $("ragAnswer").classList.remove("empty");
  $("baselineAnswer").classList.remove("empty");
  $("ragBand").textContent = bandText(result.rag_scores);
  $("baselineBand").textContent = bandText(result.baseline_scores);

  renderMetrics("ragMetrics", result.rag_scores);
  renderMetrics("baselineMetrics", result.baseline_scores);
  renderSources(result.retrieved_sources);
}

function buildPayload() {
  return {
    message: $("message").value.trim(),
    subject: valueOrNull("subject"),
    grade: valueOrNull("grade"),
    preferred_answer_type: valueOrNull("preferredAnswerType"),
    language: valueOrNull("language"),
    source: valueOrNull("source"),
    top_k: Number($("topK").value || 6),
    baseline_model: valueOrNull("baselineModel"),
  };
}

apiBaseUrl.value =
  localStorage.getItem(apiBaseKey) ||
  cleanApiBaseUrl(window.RAGAIB_API_BASE_URL || "");

apiBaseUrl.addEventListener("input", () => {
  localStorage.setItem(apiBaseKey, cleanApiBaseUrl(apiBaseUrl.value));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = buildPayload();
  if (!payload.message) {
    setStatus("Question required");
    return;
  }

  setStatus("Running live eval", true);
  try {
    const response = await fetch(liveQueryUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || `Request failed with ${response.status}`);
    }

    renderResult(result);
    setStatus("Complete");
  } catch (error) {
    console.error(error);
    setStatus("Error");
    $("comparisonText").textContent = error.message || "Live evaluation failed.";
    $("comparisonText").classList.remove("empty");
    $("winnerBadge").textContent = "Failed";
    $("winnerBadge").className = "badge winner-baseline";
  }
});
