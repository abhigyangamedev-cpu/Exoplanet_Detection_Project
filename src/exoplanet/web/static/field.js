// The map: every KOI as a dot, orbital period against signal-to-noise on log axes, coloured by
// whether the picked model was right. Hover for details, click to open, drag to zoom.
(function () {
  "use strict";

  const PAD = { top: 14, right: 18, bottom: 46, left: 60 };
  const COLORS = {
    planet: "#f4c35a",
    fp: "#6f8fd6",
    missed: "#ff7d6e",
    alarm: "#ff7d6e",
    none: "#58638f",
    grid: "#222c5a",
    text: "#9aa5cc",
    ink: "#eef0fb",
  };
  // Drawn in this order, so the mistakes sit on top of the crowd.
  const ORDER = ["none", "fp", "planet", "alarm", "missed"];
  const STATUS = {
    planet: "Right: called a planet",
    fp: "Right: called a false positive",
    missed: "Wrong: missed this planet",
    alarm: "Wrong: false alarm",
    none: "No score from this model",
  };
  const HOVER_REACH = 14;
  const DRAG_START = 5;

  function logTicks(lo, hi) {
    const all = [];
    for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) {
      for (const m of [1, 2, 5]) {
        const v = k + Math.log10(m);
        if (v >= lo && v <= hi) all.push({ v, major: m === 1 });
      }
    }
    const majors = all.filter((t) => t.major);
    return hi - lo > 2.5 && majors.length >= 2 ? majors : all;
  }

  function tickText(logValue) {
    const value = Math.pow(10, logValue);
    return value >= 1000 ? Math.round(value).toLocaleString() : String(Number(value.toPrecision(2)));
  }

  function create(options) {
    const host = options.host;
    const base = host.querySelector("canvas.base");
    const over = host.querySelector("canvas.over");
    const brushBox = host.querySelector(".brush");
    const emptyNote = host.querySelector(".field-empty");
    const tip = options.tooltip;
    const s = {
      data: null,
      model: "",
      index: new Map(),
      cat: [],
      lx: null,
      ly: null,
      px: null,
      py: null,
      dom: null,
      threshold: 0.5,
      hidden: new Set(),
      mistakesOnly: false,
      selected: -1,
      selectedName: null,
      hover: -1,
      w: 0,
      h: 0,
      drag: null,
    };

    const count = () => (s.data ? s.data.names.length : 0);
    const placed = (i) => !Number.isNaN(s.lx[i]);
    const isMistake = (c) => c === "missed" || c === "alarm";
    const shown = (i) => placed(i) && !s.hidden.has(s.cat[i]) && (!s.mistakesOnly || isMistake(s.cat[i]));

    function classify() {
      const d = s.data;
      s.cat = new Array(count());
      for (let i = 0; i < s.cat.length; i++) {
        if (d.score[i] === null) s.cat[i] = "none";
        else if (d.score[i] >= s.threshold) s.cat[i] = d.label[i] ? "planet" : "alarm";
        else s.cat[i] = d.label[i] ? "missed" : "fp";
      }
    }

    function counts() {
      const out = { planet: 0, fp: 0, missed: 0, alarm: 0, none: 0, unplaced: 0 };
      for (let i = 0; i < count(); i++) {
        out[s.cat[i]] += 1;
        if (!placed(i)) out.unplaced += 1;
      }
      return out;
    }

    function setDomain() {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < count(); i++) {
        if (!placed(i)) continue;
        x0 = Math.min(x0, s.lx[i]);
        x1 = Math.max(x1, s.lx[i]);
        y0 = Math.min(y0, s.ly[i]);
        y1 = Math.max(y1, s.ly[i]);
      }
      if (x0 > x1) {
        s.dom = null;
        return;
      }
      const widen = (lo, hi) => {
        const pad = Math.max((hi - lo) * 0.04, 0.05);
        return [lo - pad, hi + pad];
      };
      [x0, x1] = widen(x0, x1);
      [y0, y1] = widen(y0, y1);
      s.dom = { x0, x1, y0, y1 };
    }

    function layout() {
      s.w = Math.floor(host.clientWidth);
      s.h = Math.floor(host.clientHeight);
      const ratio = window.devicePixelRatio || 1;
      for (const canvas of [base, over]) {
        canvas.width = Math.max(1, Math.round(s.w * ratio));
        canvas.height = Math.max(1, Math.round(s.h * ratio));
        canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
      }
      const n = count();
      s.px = new Float32Array(n);
      s.py = new Float32Array(n);
      if (!s.dom) return;
      const plotW = s.w - PAD.left - PAD.right;
      const plotH = s.h - PAD.top - PAD.bottom;
      for (let i = 0; i < n; i++) {
        s.px[i] = PAD.left + ((s.lx[i] - s.dom.x0) / (s.dom.x1 - s.dom.x0)) * plotW;
        s.py[i] = PAD.top + (1 - (s.ly[i] - s.dom.y0) / (s.dom.y1 - s.dom.y0)) * plotH;
      }
    }

    function drawAxes(ctx) {
      const right = s.w - PAD.right;
      const bottom = s.h - PAD.bottom;
      const plotW = right - PAD.left;
      const plotH = bottom - PAD.top;
      ctx.lineWidth = 1;
      ctx.font = '12px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
      ctx.fillStyle = COLORS.text;
      ctx.strokeStyle = COLORS.grid;

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      let lastLabel = -Infinity;
      for (const t of logTicks(s.dom.x0, s.dom.x1)) {
        const x = Math.round(PAD.left + ((t.v - s.dom.x0) / (s.dom.x1 - s.dom.x0)) * plotW) + 0.5;
        ctx.globalAlpha = t.major ? 1 : 0.45;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (x - lastLabel > 44) {
          ctx.fillText(tickText(t.v), x, bottom + 7);
          lastLabel = x;
        }
      }
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      lastLabel = Infinity;
      for (const t of logTicks(s.dom.y0, s.dom.y1)) {
        const y = Math.round(PAD.top + (1 - (t.v - s.dom.y0) / (s.dom.y1 - s.dom.y0)) * plotH) + 0.5;
        ctx.globalAlpha = t.major ? 1 : 0.45;
        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (lastLabel - y > 18) {
          ctx.fillText(tickText(t.v), PAD.left - 8, y);
          lastLabel = y;
        }
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("Orbital period (days)", PAD.left + plotW / 2, s.h - 4);
      ctx.save();
      ctx.translate(14, PAD.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle";
      ctx.fillText("Signal-to-noise", 0, 0);
      ctx.restore();
    }

    function drawBase() {
      const ctx = base.getContext("2d");
      ctx.clearRect(0, 0, s.w, s.h);
      const nothing = !s.dom || s.w < 120 || s.h < 120;
      emptyNote.hidden = !s.data || Boolean(s.dom);
      if (nothing) return;
      drawAxes(ctx);

      for (const cat of ORDER) {
        if (s.hidden.has(cat) || (s.mistakesOnly && !isMistake(cat))) continue;
        const mistake = isMistake(cat);
        const radius = mistake ? 3 : 2;
        ctx.globalAlpha = mistake ? 0.9 : cat === "none" ? 0.5 : 0.6;
        ctx.fillStyle = COLORS[cat];
        ctx.strokeStyle = COLORS[cat];
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let i = 0; i < count(); i++) {
          if (s.cat[i] !== cat || !placed(i)) continue;
          ctx.moveTo(s.px[i] + radius, s.py[i]);
          ctx.arc(s.px[i], s.py[i], radius, 0, Math.PI * 2);
        }
        // A false alarm is a hollow ring, so the two kinds of mistake read apart without colour.
        if (cat === "alarm") ctx.stroke();
        else ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function ring(ctx, i, radius, label) {
      ctx.beginPath();
      ctx.arc(s.px[i], s.py[i], radius, 0, Math.PI * 2);
      ctx.stroke();
      if (!label) return;
      ctx.font = '600 13px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
      const width = ctx.measureText(label).width;
      const left = s.px[i] + radius + 6 + width > s.w - 4 ? s.px[i] - radius - 6 - width : s.px[i] + radius + 6;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, left, s.py[i]);
    }

    function drawOver() {
      const ctx = over.getContext("2d");
      ctx.clearRect(0, 0, s.w, s.h);
      if (!s.dom) return;
      ctx.strokeStyle = COLORS.ink;
      ctx.fillStyle = COLORS.ink;
      ctx.lineWidth = 2;
      if (s.selected >= 0 && placed(s.selected)) ring(ctx, s.selected, 9, s.data.names[s.selected]);
      if (s.hover >= 0 && s.hover !== s.selected && placed(s.hover)) ring(ctx, s.hover, 7, "");
    }

    function redraw() {
      drawBase();
      drawOver();
    }

    function nearest(x, y) {
      let best = -1;
      let bestD = HOVER_REACH * HOVER_REACH;
      for (let i = 0; i < count(); i++) {
        if (!shown(i)) continue;
        const dx = s.px[i] - x;
        const dy = s.py[i] - y;
        // Mistakes win ties with the crowd around them: they are what people aim for.
        const d = dx * dx + dy * dy - (isMistake(s.cat[i]) ? 12 : 0);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best;
    }

    function line(text, color) {
      const row = document.createElement("div");
      if (color) {
        const key = document.createElement("i");
        key.style.background = color;
        row.appendChild(key);
      }
      row.appendChild(document.createTextNode(text));
      return row;
    }

    function showTip(i, event) {
      const d = s.data;
      tip.textContent = "";
      const head = document.createElement("strong");
      head.textContent = d.names[i];
      tip.appendChild(head);
      tip.appendChild(line(STATUS[s.cat[i]], COLORS[s.cat[i]]));
      tip.appendChild(line(`Truly a ${d.label[i] ? "planet" : "false positive"}`));
      if (d.score[i] !== null) tip.appendChild(line(`${s.model} score ${d.score[i].toFixed(3)}`));
      tip.appendChild(
        line(`Period ${Number(d.period[i].toPrecision(4))} days, signal-to-noise ${Number(d.snr[i].toPrecision(3))}`)
      );
      tip.hidden = false;
      const left = Math.min(event.clientX + 16, window.innerWidth - tip.offsetWidth - 8);
      const top = Math.min(event.clientY + 16, window.innerHeight - tip.offsetHeight - 8);
      tip.style.left = `${Math.max(8, left)}px`;
      tip.style.top = `${Math.max(8, top)}px`;
    }

    function local(event) {
      const box = over.getBoundingClientRect();
      return { x: event.clientX - box.left, y: event.clientY - box.top };
    }

    function clampToPlot(p) {
      return {
        x: Math.max(PAD.left, Math.min(s.w - PAD.right, p.x)),
        y: Math.max(PAD.top, Math.min(s.h - PAD.bottom, p.y)),
      };
    }

    function setHover(i) {
      if (i === s.hover) return;
      s.hover = i;
      over.style.cursor = i >= 0 ? "pointer" : "crosshair";
      drawOver();
    }

    over.addEventListener("pointerdown", (event) => {
      if (!s.dom || event.button !== 0) return;
      s.drag = { from: clampToPlot(local(event)), active: false };
      over.setPointerCapture(event.pointerId);
    });

    over.addEventListener("pointermove", (event) => {
      if (!s.dom) return;
      const p = local(event);
      if (s.drag) {
        const to = clampToPlot(p);
        const from = s.drag.from;
        if (!s.drag.active && Math.hypot(to.x - from.x, to.y - from.y) < DRAG_START) return;
        s.drag.active = true;
        s.drag.to = to;
        tip.hidden = true;
        setHover(-1);
        brushBox.hidden = false;
        brushBox.style.left = `${Math.min(from.x, to.x)}px`;
        brushBox.style.top = `${Math.min(from.y, to.y)}px`;
        brushBox.style.width = `${Math.abs(to.x - from.x)}px`;
        brushBox.style.height = `${Math.abs(to.y - from.y)}px`;
        return;
      }
      const i = nearest(p.x, p.y);
      setHover(i);
      if (i >= 0) showTip(i, event);
      else tip.hidden = true;
    });

    over.addEventListener("pointerup", (event) => {
      const drag = s.drag;
      s.drag = null;
      brushBox.hidden = true;
      if (!drag) return;
      if (drag.active) {
        const plotW = s.w - PAD.left - PAD.right;
        const plotH = s.h - PAD.top - PAD.bottom;
        const period = (x) => Math.pow(10, s.dom.x0 + ((x - PAD.left) / plotW) * (s.dom.x1 - s.dom.x0));
        const snr = (y) => Math.pow(10, s.dom.y0 + (1 - (y - PAD.top) / plotH) * (s.dom.y1 - s.dom.y0));
        const a = drag.from;
        const b = drag.to;
        if (Math.abs(a.x - b.x) < 8 || Math.abs(a.y - b.y) < 8) return; // too thin to mean a range
        options.onBrush({
          period: [period(Math.min(a.x, b.x)), period(Math.max(a.x, b.x))],
          snr: [snr(Math.max(a.y, b.y)), snr(Math.min(a.y, b.y))],
        });
        return;
      }
      const p = local(event);
      const i = nearest(p.x, p.y);
      if (i >= 0) {
        tip.hidden = true;
        options.onPick(s.data.names[i]);
      }
    });

    over.addEventListener("pointercancel", () => {
      s.drag = null;
      brushBox.hidden = true;
    });

    over.addEventListener("pointerleave", () => {
      if (s.drag) return;
      tip.hidden = true;
      setHover(-1);
    });

    return {
      setData(data, model) {
        s.data = data;
        s.model = model;
        const n = data.names.length;
        s.index = new Map(data.names.map((name, i) => [name, i]));
        s.lx = new Float32Array(n);
        s.ly = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          const ok = data.period[i] > 0 && data.snr[i] > 0;
          s.lx[i] = ok ? Math.log10(data.period[i]) : NaN;
          s.ly[i] = ok ? Math.log10(data.snr[i]) : NaN;
        }
        s.hover = -1;
        s.selected = s.index.has(s.selectedName) ? s.index.get(s.selectedName) : -1;
        classify();
        setDomain();
        layout();
        redraw();
      },
      setThreshold(value) {
        s.threshold = value;
        if (!s.data) return;
        classify();
        redraw();
      },
      setHidden(hidden) {
        s.hidden = hidden;
        redraw();
      },
      setMistakesOnly(on) {
        s.mistakesOnly = on;
        redraw();
      },
      // The name is kept so the ring survives a change of data.
      setSelected(name) {
        s.selectedName = name;
        s.selected = s.index.has(name) ? s.index.get(name) : -1;
        drawOver();
      },
      setHoverName(name) {
        setHover(name !== null && s.index.has(name) ? s.index.get(name) : -1);
      },
      counts,
      resize() {
        if (!s.data || host.clientWidth < 1) return;
        layout();
        redraw();
      },
    };
  }

  window.Field = { create };
})();
