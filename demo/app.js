/* Calibre chat demo.
 * Retrieval results are precomputed with calibre_rag v1 + v2 (see scripts/export_from_pilot.py):
 * for every test question we know the ranked passages with P(relevant), how many passages
 * Calibre sends for each recall target, how many a fixed top-5 sends, token counts, and
 * which passages the dataset labels as relevant.
 */
const DATA = window.CALIBRE_DEMO;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = n => Math.round(n).toLocaleString("en-US");
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

const state = { kb: null, target: DATA.default_target, mode: "same", thread: [], matchIdx: -1, matches: [] };
const MODE_LABEL = { same: "This dataset", other: "Other datasets", fewshot: `+ ${DATA.few_shot} labels` };
const MODE_NOTE = {
  same: "Calibrated on labelled questions from this dataset.",
  other: "Calibrated only on other datasets, with no labels from this one (v3).",
  fewshot: `Calibrated on other datasets plus ${DATA.few_shot} labelled questions from this one (v3).`,
};
const kb = () => DATA.kbs.find(k => k.id === state.kb);
const tokens = (item, n) => item.tk.slice(0, n).reduce((a, b) => a + b, 0);
const pct = p => (p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
const SHOW_FIRST = 4;   // passages shown before "show more"

/* ---------- matching typed text to test questions ---------- */
const STOP = new Set("a an the of in on at to for and or is are was were be by with what who whom which when where why how did does do from as that this it its into than".split(" "));
const words = s => s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w && !STOP.has(w));
function score(query, text) {
  const q = new Set(words(query)), t = new Set(words(text));
  if (!q.size || !t.size) return 0;
  let hit = 0; for (const w of q) if (t.has(w)) hit++;
  return hit / Math.sqrt(q.size * t.size);
}
function search(query, n = 6) {
  if (!query.trim()) return [];
  return kb().bank.map((it, i) => ({ i, s: score(query, it.q) })).filter(x => x.s > 0)
    .sort((a, b) => b.s - a.s).slice(0, n);
}

/* ---------- answer lookup (SQuAD) ---------- */
function findAnswer(item) {
  if (!item.answers) return null;
  if (item.ans !== undefined) return { pos: item.ans, text: item.ans >= 0 ? item.ans_text : item.answers[0] || "" };
  const k = kb();
  for (let p = 0; p < item.docs.length; p++) {
    const text = (k.passages[item.docs[p]].x || "").toLowerCase();
    for (const a of item.answers) if (a && text.includes(a.toLowerCase())) return { pos: p, text: a };
  }
  return { pos: -1, text: item.answers[0] || "" };
}

function highlight(text, answer) {
  if (!answer) return esc(text);
  const i = text.toLowerCase().indexOf(answer.toLowerCase());
  if (i < 0) return esc(text);
  return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + answer.length)) + "</mark>" + esc(text.slice(i + answer.length));
}

/* ---------- rendering ---------- */
function renderKbs() {
  $("kbs").innerHTML = DATA.kbs.map(k =>
    `<button type="button" data-kb="${k.id}" aria-pressed="${k.id === state.kb}">${esc(k.name)}</button>`).join("");
  $("kbs").querySelectorAll("button").forEach(b => b.addEventListener("click", () => switchKb(b.dataset.kb)));
}

function renderTargets() {
  $("targets").innerHTML = DATA.targets.map(t =>
    `<button type="button" data-t="${t}" aria-pressed="${t === state.target}">${t}%</button>`).join("");
  $("targets").querySelectorAll("button").forEach(b => b.addEventListener("click", () => setTarget(b.dataset.t)));
}

function renderModes() {
  $("modes").innerHTML = DATA.modes.map(m =>
    `<button type="button" data-m="${m}" aria-pressed="${m === state.mode}" title="${esc(MODE_NOTE[m])}">${esc(MODE_LABEL[m])}</button>`).join("");
  $("modes").querySelectorAll("button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.m)));
}

function setMode(m) {
  if (m === state.mode) return;
  state.mode = m;
  const lastBot = [...state.thread].reverse().find(x => x.role === "bot");
  if (lastBot) lastBot.animated = false;
  renderModes(); render();
}

function renderSummary() {
  const sm = kb().summary, cur = sm[state.mode][state.target], top = sm.top5;
  const hit = cur.recall >= Number(state.target) / 100 - 0.005;
  $("summary").innerHTML = `${esc(MODE_NOTE[state.mode])} Across ${cur.n} test questions, Calibre reaches
    <strong class="${hit ? "" : "short"}">${Math.round(cur.recall * 100)}% recall</strong> (target ${state.target}%) with
    <strong>${cur.k.toFixed(1)} passages</strong> on average. Top-5 reaches ${Math.round(top.recall * 100)}% with 5.`;
}

function setTarget(t) {
  if (t === state.target) return;
  state.target = t;
  // replay the meter on the latest reply so the change in context is visible
  const lastBot = [...state.thread].reverse().find(m => m.role === "bot");
  if (lastBot) lastBot.animated = false;
  renderTargets(); render();
}

function chipsHtml(indices) {
  return `<div class="chips">${indices.map(i =>
    `<button type="button" class="chip" data-ask="${i}">${esc(kb().bank[i].q)}</button>`).join("")}</div>`;
}

function pickSuggestions(n, exclude = new Set()) {
  const pool = kb().suggested.filter(i => !exclude.has(i));
  const shuffled = pool.slice().sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

function sourceCard(item, docId, rank, answer, extra) {
  const p = kb().passages[docId];
  const prob = item.p[state.mode][rank - 1];
  const isRel = item.rel[docId] > 0;
  const hasAnswer = answer && p.x.toLowerCase().includes(answer.toLowerCase());
  const tag = extra ? `Top-5 only, rank ${rank}` : rank > item.k_top ? `Rank ${rank}, beyond top-5` : `Rank ${rank}`;
  return `<div class="src ${isRel ? "rel" : ""} ${extra ? "extra" : ""} ${hasAnswer ? "open" : ""}">
    <div class="src-head"><span class="src-title">${esc(p.t || "Passage")}</span><span class="src-rank">${tag}</span></div>
    <button type="button" class="src-expand" aria-expanded="${hasAnswer}" aria-label="Show or hide the full passage">
      <p class="src-text">${highlight(p.x, answer)}</p></button>
    <div class="src-foot">
      <span class="prob" title="Calibre's estimate that this passage is relevant"><span class="pbar"><i style="width:${Math.max(prob * 100, 2).toFixed(0)}%"></i></span>${pct(prob)} likely relevant</span>
      ${isRel ? `<span class="pill rel">${item.rel[docId] >= 2 ? "Labelled highly relevant" : "Labelled relevant"}</span>` : ""}
    </div>
  </div>`;
}

function replyHtml(msg) {
  const k = kb(), item = k.bank[msg.item];
  const kCal = item.k[state.mode][state.target], kTop = item.k_top;
  const tokCal = tokens(item, kCal), tokTop = tokens(item, kTop);
  const relCal = item.docs.slice(0, kCal).filter(d => item.rel[d] > 0).length;
  const relTop = item.docs.slice(0, kTop).filter(d => item.rel[d] > 0).length;
  let head = "", answerText = null;

  if (item.answers) {
    const a = findAnswer(item);
    answerText = a.pos >= 0 ? a.text : null;
    if (a.pos >= 0 && a.pos < kCal) {
      head = `<p class="answer"><strong>${esc(a.text)}</strong></p><p class="answer-note">Found in ${kCal === 1 ? "the only passage" : `passage ${a.pos + 1} of the ${kCal}`} Calibre sent${a.pos >= kTop ? ". Top-5 would have missed it" : ""}.</p>`;
    } else if (a.pos >= 0 && a.pos < kTop) {
      head = `<p class="answer"><strong>${esc(a.text)}</strong></p><p class="answer-note bad">Calibre missed this one at a ${state.target}% target: the answer was in passage ${a.pos + 1}, which only top-5 sent.</p>`;
    } else if (a.pos >= 0) {
      head = `<p class="answer"><strong>${esc(a.text)}</strong></p><p class="answer-note bad">The answer was in passage ${a.pos + 1}, which neither top-5 nor Calibre at a ${state.target}% target sent.</p>`;
    } else {
      head = `<p class="answer">No answer in the retrieved passages.</p><p class="answer-note bad">Search didn't surface the right paragraph, so neither top-5 nor Calibre has it.</p>`;
    }
  } else if (item.rel_top20 === 0) {
    head = `<p class="answer">No relevant passage was found.</p><p class="answer-note bad">Search didn't surface any passage labelled relevant, so neither top-5 nor Calibre has one.</p>`;
  } else {
    head = `<p class="answer"><strong>${plural(relCal, "relevant passage")}</strong> in Calibre's context</p>
      <p class="answer-note ${relCal < relTop ? "bad" : ""}">Top-5 included ${relCal > relTop ? "only " : ""}${relTop}${relCal < relTop ? ", so Calibre dropped some here" : ""}.</p>`;
  }
  if (item.nothing[state.mode]) {
    head += `<p class="flag"><strong>Calibre flagged this question:</strong> no passage looks relevant (best ${pct(Math.max(...item.p[state.mode]))}). An agent could search again or say it doesn't know.</p>`;
  }

  const max = Math.max(tokCal, tokTop, 1);
  const diff = tokTop ? (tokCal - tokTop) / tokTop * 100 : 0;
  let saving;
  if (Math.abs(diff) < 0.5) saving = `<span class="saving">Same context as top-5</span>`;
  else if (diff < 0) saving = `<span class="saving">${Math.round(-diff)}% less context than top-5</span>`;
  else saving = `<span class="saving ${relCal > relTop ? "more" : "worse"}">${Math.round(diff)}% more context than top-5${relCal > relTop ? ", keeping more relevant passages" : ""}</span>`;

  const meter = `<div class="meter">
      <div class="meter-row cal"><span class="who">Calibre</span><span class="track"><span class="fill" data-w="${(tokCal / max * 100).toFixed(1)}"></span></span><span class="what">${plural(kCal, "passage")}, ${fmt(tokCal)} tokens</span></div>
      <div class="meter-row top"><span class="who">Top-5</span><span class="track"><span class="fill" data-w="${(tokTop / max * 100).toFixed(1)}"></span></span><span class="what">${plural(kTop, "passage")}, ${fmt(tokTop)} tokens</span></div>
    </div>${saving}`;

  const shown = msg.more ? kCal : Math.min(kCal, SHOW_FIRST);
  const sent = item.docs.slice(0, shown).map((d, i) => sourceCard(item, d, i + 1, answerText, false)).join("")
    + (kCal > SHOW_FIRST ? `<button type="button" class="linkish toggle-more" aria-expanded="${!!msg.more}">${msg.more ? "Show fewer" : `Show the other ${plural(kCal - SHOW_FIRST, "passage")} Calibre sent`}</button>` : "");
  const extras = item.docs.slice(kCal, kTop);
  const extraHtml = extras.length ? `
      <button type="button" class="linkish toggle-extra" aria-expanded="${msg.open}">${msg.open ? "Hide" : "Show"} the ${plural(extras.length, "passage")} only top-5 sends</button>
      ${msg.open ? `<div class="extra-wrap">${extras.map((d, j) => sourceCard(item, d, kCal + j + 1, answerText, true)).join("")}</div>` : ""}` : "";

  const matched = msg.typed && msg.typed.trim().toLowerCase() !== item.q.toLowerCase()
    ? `<p class="matched">Closest question in the test set: “${esc(item.q)}”</p>` : "";

  return `<div class="reply">${matched}${head}${meter}<div class="sources">${sent}</div>${extraHtml}</div>`;
}

function render() {
  const k = kb();
  $("ask").placeholder = k.placeholder;
  renderSummary();
  $("fine").textContent = `Questions from the ${k.source} test set, with precomputed BM25 + embedding search. The recall target holds on average across questions, not on every one. Answers come from the dataset's labels; no LLM is called.`;
  const html = state.thread.map((m, idx) => {
    if (m.role === "welcome") {
      return `<li class="welcome"><h1>Ask a question</h1><p>Calibre sends the LLM only the passages a question needs: just enough to reach the recall target you pick.</p>${chipsHtml(m.chips)}</li>`;
    }
    if (m.role === "user") return `<li class="msg-user">${esc(m.text)}</li>`;
    if (m.role === "thinking") return `<li class="msg-bot"><div class="reply thinking">Searching<span></span><span></span><span></span></div></li>`;
    if (m.role === "nomatch") {
      return `<li class="msg-bot"><div class="reply"><p class="answer">That question isn't in the ${esc(k.source)} test set.</p>
        <p class="answer-note">This demo only answers questions it has labels for. ${m.close ? "Closest questions it has:" : "Try one of these:"}</p>
        <div style="margin-top:10px">${chipsHtml(m.chips)}</div></div></li>`;
    }
    const last = idx === state.thread.length - 1;
    return `<li class="msg-bot" data-msg="${idx}">${replyHtml(m)}${last ? `<p class="chips-label">Try another</p>${chipsHtml(m.chips)}` : ""}</li>`;
  }).join("");
  $("thread").innerHTML = html;

  $("thread").querySelectorAll("[data-ask]").forEach(b => b.addEventListener("click", () => ask(Number(b.dataset.ask))));
  $("thread").querySelectorAll(".toggle-extra").forEach(b => b.addEventListener("click", () => {
    const m = state.thread[Number(b.closest("[data-msg]").dataset.msg)]; m.open = !m.open; m.animated = true; render();
  }));
  $("thread").querySelectorAll(".toggle-more").forEach(b => b.addEventListener("click", () => {
    const m = state.thread[Number(b.closest("[data-msg]").dataset.msg)]; m.more = !m.more; m.animated = true; render();
  }));
  $("thread").querySelectorAll(".src-expand").forEach(b => b.addEventListener("click", () => {
    const card = b.closest(".src"); const open = card.classList.toggle("open"); b.setAttribute("aria-expanded", open);
  }));
  // animate the context meters of new replies once
  $("thread").querySelectorAll("[data-msg]").forEach(li => {
    const m = state.thread[Number(li.dataset.msg)];
    const fills = li.querySelectorAll(".fill");
    if (m.animated || REDUCED) { fills.forEach(f => (f.style.width = f.dataset.w + "%")); return; }
    m.animated = true;
    requestAnimationFrame(() => requestAnimationFrame(() => fills.forEach(f => (f.style.width = f.dataset.w + "%"))));
  });
}

function scrollToEnd() {
  const last = $("thread").lastElementChild;
  if (last) last.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
}

function ask(itemIdx, typed) {
  const item = kb().bank[itemIdx];
  state.thread = state.thread.filter(m => m.role !== "welcome");
  state.thread.push({ role: "user", text: typed || item.q });
  state.thread.push({ role: "thinking" });
  render(); scrollToEnd();
  const asked = new Set(state.thread.filter(m => m.item !== undefined).map(m => m.item).concat([itemIdx]));
  setTimeout(() => {
    state.thread.pop();
    state.thread.push({ role: "bot", item: itemIdx, typed, open: false, chips: pickSuggestions(3, asked) });
    render(); scrollToEnd();
  }, REDUCED ? 0 : 450);
}

function askText(text) {
  const best = search(text, 1)[0];
  if (best && best.s >= 0.5) return ask(best.i, text);
  state.thread = state.thread.filter(m => m.role !== "welcome");
  state.thread.push({ role: "user", text });
  const close = search(text, 4).filter(x => x.s >= 0.25).map(x => x.i);
  state.thread.push({ role: "nomatch", close: close.length > 0, chips: close.length ? close : pickSuggestions(4) });
  render(); scrollToEnd();
}

function switchKb(id) {
  state.kb = id;
  state.thread = [{ role: "welcome", chips: pickSuggestions(6) }];
  renderKbs(); render(); closeMatches();
  window.scrollTo({ top: 0 });
}

/* ---------- composer with suggestions as you type ---------- */
function closeMatches() { $("matches").hidden = true; $("ask").setAttribute("aria-expanded", "false"); state.matchIdx = -1; state.matches = []; }
function showMatches() {
  const list = $("matches");
  state.matches = search($("ask").value);
  if (!state.matches.length) return closeMatches();
  list.innerHTML = state.matches.map((m, j) =>
    `<li role="option" id="opt-${j}" data-i="${m.i}" aria-selected="${j === state.matchIdx}">${esc(kb().bank[m.i].q)}</li>`).join("");
  list.hidden = false; $("ask").setAttribute("aria-expanded", "true");
  list.querySelectorAll("li").forEach(li => li.addEventListener("mousedown", e => {
    e.preventDefault(); const i = Number(li.dataset.i); $("ask").value = ""; closeMatches(); ask(i);
  }));
}
$("ask").addEventListener("input", () => { state.matchIdx = -1; showMatches(); });
$("ask").addEventListener("keydown", e => {
  if ($("matches").hidden) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const n = state.matches.length;
    state.matchIdx = e.key === "ArrowDown" ? (state.matchIdx + 1) % n : (state.matchIdx - 1 + n) % n;
    showMatches();
    $("ask").setAttribute("aria-activedescendant", `opt-${state.matchIdx}`);
  } else if (e.key === "Escape") closeMatches();
});
$("ask").addEventListener("blur", () => setTimeout(closeMatches, 100));
$("composer").addEventListener("submit", e => {
  e.preventDefault();
  const text = $("ask").value.trim();
  const chosen = state.matchIdx >= 0 ? state.matches[state.matchIdx] : null;
  $("ask").value = ""; closeMatches();
  if (chosen) return ask(chosen.i);
  if (text) askText(text);
});

$("gh").href = DATA.github;
renderTargets();
renderModes();
switchKb(DATA.kbs[0].id);
