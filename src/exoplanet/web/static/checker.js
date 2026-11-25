// The checker page: filter the KOI list, pick one, compare each model's verdict to the truth.
(function () {
  "use strict";

  const PAGE_SIZE = 50;
  const TEXT_FIELDS = ["q", "snr_min", "snr_max", "period_min", "period_max"];
  const SERIES = { first: "#2a6fdb", second: "#d9730d" };
  const LABELS = { 1: "Confirmed planet", 0: "False positive" };
  const SHORT_LABELS = { 1: "planet", 0: "false positive" };
  const CATALOGUE = [
    ["koi_period", "Period", "days", 3],
    ["koi_depth", "Depth", "ppm", 0],
    ["koi_prad", "Planet radius", "Earth radii", 2],
    ["koi_model_snr", "Signal-to-noise", "", 1],
    ["koi_duration", "Transit duration", "hours", 2],
  ];

  const $ = (id) => document.getElementById(id);
  const state = { models: [], thresholds: {}, defaults: {}, page: 1, selected: null, list: null, detail: null };

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function radio(name) {
    return document.querySelector(`input[name="${name}"]:checked`).value;
  }

  function setRadio(name, value) {
    const input = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (input) input.checked = true;
  }

  function model() {
    return $("model").value;
  }

  function filterParams() {
    const params = new URLSearchParams();
    for (const id of TEXT_FIELDS) if ($(id).value.trim()) params.set(id, $(id).value.trim());
    if (radio("split")) params.set("split", radio("split"));
    if (radio("label")) params.set("label", radio("label"));
    if ($("mistakes").checked) params.set("mistakes", "1");
    params.set("model", model());
    return params;
  }

  function syncUrl() {
    const params = filterParams();
    const threshold = state.thresholds[model()];
    if (threshold !== state.defaults[model()]) params.set("threshold", threshold);
    if (state.page > 1) params.set("page", state.page);
    if (state.selected) params.set("koi", state.selected);
    history.replaceState(null, "", `?${params}`);
  }

  function readUrl() {
    const params = new URLSearchParams(location.search);
    for (const id of TEXT_FIELDS) $(id).value = params.get(id) || "";
    setRadio("split", params.get("split") || "");
    setRadio("label", params.get("label") || "");
    $("mistakes").checked = params.get("mistakes") === "1";
    if (state.models.some((m) => m.name === params.get("model"))) $("model").value = params.get("model");
    const threshold = parseFloat(params.get("threshold"));
    if (threshold >= 0 && threshold <= 1) state.thresholds[model()] = threshold;
    state.page = Math.max(1, parseInt(params.get("page"), 10) || 1);
    state.selected = params.get("koi");
  }

  async function getJson(url) {
    const response = await fetch(url);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `The server answered ${response.status}.`);
    return body;
  }

  // ---- list ---------------------------------------------------------------------------

  function mark(correct) {
    if (correct === null) return make("span", "mark none", "no score");
    return make("span", `mark ${correct ? "right" : "wrong"}`, correct ? "✓ Right" : "✕ Wrong");
  }

  function renderTally(list) {
    const tally = $("tally");
    tally.textContent = "";
    const count = `${list.total.toLocaleString()} ${list.total === 1 ? "KOI" : "KOIs"}`;
    if ($("mistakes").checked) {
      // Every row here is a mistake, so a right/wrong split would only say "0% right".
      tally.appendChild(make("strong", "", count));
      tally.appendChild(make("span", "wrong", `all wrong for ${model()}`));
      return;
    }
    const judged = list.right + list.wrong;
    const accuracy = judged ? `${((100 * list.right) / judged).toFixed(1)}% right` : "nothing scored";
    tally.appendChild(make("strong", "", count));
    tally.appendChild(make("span", "right", `${list.right.toLocaleString()} right`));
    tally.appendChild(make("span", "wrong", `${list.wrong.toLocaleString()} wrong`));
    if (list.unscored) tally.appendChild(make("span", "none", `${list.unscored.toLocaleString()} no score`));
    tally.appendChild(make("span", "accuracy", accuracy));
  }

  function renderList(list) {
    const ul = $("kois");
    ul.textContent = "";
    if (!list.items.length) {
      const why = list.total
        ? "No KOIs on this page. Go back a page."
        : "No KOIs match these filters. Loosen a filter or clear them.";
      ul.appendChild(make("li", "empty", why));
    }
    for (const item of list.items) {
      const li = make("li", "koi");
      li.setAttribute("role", "option");
      li.dataset.name = item.kepoi_name;
      li.setAttribute("aria-selected", String(item.kepoi_name === state.selected));
      li.appendChild(make("span", "name", item.kepoi_name));
      li.appendChild(make("span", "truth", SHORT_LABELS[item.label]));
      li.appendChild(mark(item.correct));
      li.addEventListener("click", () => select(item.kepoi_name));
      ul.appendChild(li);
    }
    $("page-info").textContent = `Page ${list.page} of ${list.pages}`;
    $("prev").disabled = list.page <= 1;
    $("next").disabled = list.page >= list.pages;
  }

  async function loadList() {
    const params = filterParams();
    params.set("threshold", state.thresholds[model()]);
    params.set("page", state.page);
    params.set("page_size", PAGE_SIZE);
    syncUrl();
    try {
      const list = await getJson(`api/kois?${params}`);
      if (params.toString() !== latestQuery) return; // a newer request is on its way
      state.list = list;
      renderTally(list);
      renderList(list);
    } catch (error) {
      $("tally").textContent = "";
      $("kois").textContent = "";
      $("kois").appendChild(make("li", "empty error", `Could not load the list: ${error.message}`));
    }
  }
  let latestQuery = "";
  const refreshList = () => {
    const params = filterParams();
    params.set("threshold", state.thresholds[model()]);
    params.set("page", state.page);
    params.set("page_size", PAGE_SIZE);
    latestQuery = params.toString();
    return loadList();
  };
  const refreshListSoon = debounce(refreshList, 250);

  // ---- one KOI ------------------------------------------------------------------------

  function verdictCells(row, label) {
    if (row.score === null) return { verdict: row.status, result: make("span", "mark none", "—") };
    const verdict = row.score >= state.thresholds[row.model] ? 1 : 0;
    return { verdict: SHORT_LABELS[verdict], result: mark(verdict === label) };
  }

  function track(row) {
    const wrap = make("div", "track");
    const threshold = state.thresholds[row.model];
    const tick = make("i", "threshold");
    tick.style.left = `${threshold * 100}%`;
    wrap.appendChild(tick);
    if (row.score !== null) {
      const dot = make("i", "score");
      dot.style.left = `${row.score * 100}%`;
      wrap.appendChild(dot);
    }
    wrap.title = `Planet at or above ${threshold.toFixed(3)}`;
    return wrap;
  }

  function renderScores(detail) {
    const body = $("scores-body");
    if (!body) return;
    body.textContent = "";
    for (const row of detail.scores) {
      const tr = make("tr", row.model === model() ? "picked" : "");
      const cells = verdictCells(row, detail.label);
      tr.appendChild(make("th", "", row.model)).setAttribute("scope", "row");
      tr.appendChild(make("td", "track-cell")).appendChild(track(row));
      tr.appendChild(make("td", "num", row.score === null ? "—" : row.score.toFixed(3)));
      tr.appendChild(make("td", row.score === null ? "muted" : "", cells.verdict));
      tr.appendChild(make("td", "")).appendChild(cells.result);
      tr.appendChild(make("td", "muted", row.source || ""));
      body.appendChild(tr);
    }

    const picked = detail.scores.find((row) => row.model === model());
    const headline = $("headline");
    headline.textContent = "";
    if (!picked || picked.score === null) {
      headline.className = "headline none";
      headline.textContent = `${model()}: ${picked ? picked.status : "no score"} for this KOI.`;
    } else {
      const verdict = picked.score >= state.thresholds[model()] ? 1 : 0;
      const right = verdict === detail.label;
      headline.className = `headline ${right ? "right" : "wrong"}`;
      headline.appendChild(make("span", "word", right ? "✓ Right" : "✕ Wrong"));
      headline.appendChild(
        make("span", "", `${model()} said ${SHORT_LABELS[verdict]} (score ${picked.score.toFixed(3)}, ${picked.source})`)
      );
    }
  }

  function renderThreshold() {
    const name = model();
    const meta = state.models.find((m) => m.name === name);
    $("threshold").value = state.thresholds[name];
    $("threshold-value").textContent = state.thresholds[name].toFixed(3);
    $("threshold-name").textContent = name;
    const note = meta.threshold_is_default
      ? "Default 0.500: no dev-chosen threshold is saved for this model."
      : `Default ${meta.threshold.toFixed(3)}: chosen on dev folds for 95% precision.`;
    $("threshold-note").textContent = note;
    $("threshold-reset").disabled = state.thresholds[name] === state.defaults[name];
  }

  function chartBlock(parent, title, legend) {
    const figure = make("figure", "chart");
    const caption = make("figcaption", "");
    caption.appendChild(make("span", "chart-title", title));
    if (legend) {
      for (const [name, color] of legend) {
        const key = make("span", "key", name);
        const swatch = make("i", "");
        swatch.style.background = color;
        key.prepend(swatch);
        caption.appendChild(key);
      }
    }
    figure.appendChild(caption);
    const host = figure.appendChild(make("div", "plot"));
    parent.appendChild(figure);
    return host;
  }

  function renderCharts() {
    const detail = state.detail;
    const charts = $("charts");
    if (!detail || !charts) return;
    charts.textContent = "";
    const v = detail.views;
    if (!v) {
      charts.appendChild(make("p", "empty", "No light curve built for this star."));
      return;
    }
    const one = (name, values) => [{ name, color: SERIES.first, values }];
    Charts.draw(chartBlock(charts, "Global view: the whole orbit, folded"), {
      series: one("Flux", v.global_view), marks: "line", label: "Global view of the folded light curve",
    });
    Charts.draw(chartBlock(charts, "Local view: around the transit"), {
      series: one("Flux", v.local_view), marks: "dots", label: "Local view around the transit",
    });
    Charts.draw(
      chartBlock(charts, "Odd and even transits", [["Odd", SERIES.first], ["Even", SERIES.second]]),
      {
        series: [
          { name: "Odd", color: SERIES.first, values: v.odd_view },
          { name: "Even", color: SERIES.second, values: v.even_view },
        ],
        marks: "dots",
        label: "Odd and even transits overlaid",
      }
    );
    Charts.draw(chartBlock(charts, "Secondary view: half an orbit later"), {
      series: one("Flux", v.secondary_view), marks: "dots", label: "Secondary eclipse view",
    });
  }

  function renderDetail(detail) {
    const sheet = $("sheet");
    sheet.textContent = "";

    const head = sheet.appendChild(make("header", "koi-head"));
    head.appendChild(make("h1", "", detail.kepoi_name));
    const where = detail.split === "dev" ? `development, fold ${detail.fold}` : "held-out test set";
    head.appendChild(make("p", "muted", `KIC ${detail.kepid}, ${where}`));
    const truth = head.appendChild(make("p", "truth-line"));
    truth.appendChild(make("span", "muted", "True label "));
    truth.appendChild(make("strong", detail.label ? "planet" : "fp", LABELS[detail.label]));

    sheet.appendChild(make("p", "headline", "")).id = "headline";

    const control = sheet.appendChild(make("div", "threshold-control"));
    const label = control.appendChild(make("label", ""));
    label.htmlFor = "threshold";
    label.appendChild(document.createTextNode("Threshold for "));
    label.appendChild(make("strong", "", "")).id = "threshold-name";
    const slider = control.appendChild(make("input", ""));
    Object.assign(slider, { type: "range", id: "threshold", min: 0, max: 1, step: 0.001 });
    control.appendChild(make("output", "num", "")).id = "threshold-value";
    const reset = control.appendChild(make("button", "link", "Reset to default"));
    reset.type = "button";
    reset.id = "threshold-reset";
    control.appendChild(make("p", "muted note", "")).id = "threshold-note";
    slider.addEventListener("input", () => setThreshold(parseFloat(slider.value)));
    reset.addEventListener("click", () => setThreshold(state.defaults[model()]));

    const table = sheet.appendChild(make("table", "scores"));
    const headRow = table.appendChild(make("thead", "")).appendChild(make("tr", ""));
    for (const title of ["Model", "Score against threshold", "Score", "Verdict", "Result", "Score from"]) {
      headRow.appendChild(make("th", "", title)).setAttribute("scope", "col");
    }
    table.appendChild(make("tbody", "")).id = "scores-body";

    const facts = sheet.appendChild(make("dl", "facts"));
    for (const [key, name, unit, digits] of CATALOGUE) {
      if (!(key in detail.catalogue)) continue;
      const value = detail.catalogue[key];
      facts.appendChild(make("dt", "", name));
      const text = value === null ? "—" : `${value.toLocaleString(undefined, { maximumFractionDigits: digits })} ${unit}`;
      facts.appendChild(make("dd", "num", text.trim()));
    }
    facts.appendChild(make("dt", "", "False-positive flags"));
    facts.appendChild(make("dd", "", detail.fp_flags.length ? detail.fp_flags.join(", ") : "none"));

    sheet.appendChild(make("div", "charts", "")).id = "charts";

    renderThreshold();
    renderScores(detail);
    renderCharts();
  }

  function setThreshold(value) {
    state.thresholds[model()] = value;
    state.page = 1;
    if (state.detail) {
      renderThreshold();
      renderScores(state.detail);
    }
    refreshListSoon();
  }

  async function select(name) {
    state.selected = name;
    for (const li of $("kois").querySelectorAll(".koi")) {
      li.setAttribute("aria-selected", String(li.dataset.name === name));
    }
    syncUrl();
    try {
      const detail = await getJson(`api/koi/${encodeURIComponent(name)}`);
      if (state.selected !== name) return;
      state.detail = detail;
      renderDetail(detail);
    } catch (error) {
      state.detail = null;
      $("sheet").textContent = "";
      $("sheet").appendChild(make("p", "empty error", `Could not open ${name}: ${error.message}`));
    }
  }

  // ---- wiring -------------------------------------------------------------------------

  function filtersChanged() {
    state.page = 1;
    refreshListSoon();
  }

  function moveSelection(step) {
    const rows = [...$("kois").querySelectorAll(".koi")];
    if (!rows.length) return;
    const at = rows.findIndex((li) => li.dataset.name === state.selected);
    const next = rows[Math.max(0, Math.min(rows.length - 1, at + step))];
    next.scrollIntoView({ block: "nearest" });
    select(next.dataset.name);
  }

  async function start() {
    let meta;
    try {
      meta = await getJson("api/meta");
    } catch (error) {
      $("empty").textContent = `Could not load the model list: ${error.message}`;
      return;
    }
    if (!meta.models.length) {
      $("empty").textContent = "No saved model scores were found in the outputs folder. Train and evaluate first.";
      return;
    }
    state.models = meta.models;
    for (const m of meta.models) {
      state.thresholds[m.name] = m.threshold;
      state.defaults[m.name] = m.threshold;
      const option = make("option", "", m.scored_on_test ? m.name : `${m.name} (dev only)`);
      option.value = m.name;
      $("model").appendChild(option);
    }
    readUrl();

    for (const id of TEXT_FIELDS) $(id).addEventListener("input", filtersChanged);
    for (const input of document.querySelectorAll('input[name="split"], input[name="label"]')) {
      input.addEventListener("change", filtersChanged);
    }
    $("mistakes").addEventListener("change", filtersChanged);
    $("model").addEventListener("change", () => {
      filtersChanged();
      if (state.detail) {
        renderThreshold();
        renderScores(state.detail);
      }
    });
    $("clear").addEventListener("click", () => {
      for (const id of TEXT_FIELDS) $(id).value = "";
      setRadio("split", "");
      setRadio("label", "");
      $("mistakes").checked = false;
      filtersChanged();
    });
    $("prev").addEventListener("click", () => {
      state.page -= 1;
      refreshList();
    });
    $("next").addEventListener("click", () => {
      state.page += 1;
      refreshList();
    });
    $("kois").addEventListener("keydown", (event) => {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      moveSelection(event.key === "ArrowDown" ? 1 : -1);
    });
    window.addEventListener("resize", debounce(renderCharts, 150));

    await refreshList();
    if (state.selected) select(state.selected);
  }

  start();
})();
