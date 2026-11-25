// Light-curve charts drawn as SVG. One y axis, thin marks, a crosshair and tooltip on hover.
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const PAD = { top: 10, right: 12, bottom: 30, left: 46 };

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return { ticks, step };
  }

  function format(value, step) {
    const digits = Math.max(0, Math.min(6, -Math.floor(Math.log10(step))));
    return value.toFixed(digits);
  }

  // series: [{name, color, values: (number|null)[]}]; marks: "line" or "dots".
  function draw(host, options) {
    const series = options.series;
    const height = options.height || 200;
    const width = Math.max(240, Math.floor(host.clientWidth));
    const n = Math.max(...series.map((s) => s.values.length));
    const all = series.flatMap((s) => s.values).filter((v) => v !== null);
    host.textContent = "";
    if (!all.length || n < 2) {
      const note = document.createElement("p");
      note.className = "chart-empty";
      note.textContent = "No usable points in this view.";
      host.appendChild(note);
      return;
    }

    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const margin = (hi - lo || 1) * 0.06;
    lo -= margin;
    hi += margin;
    const plotW = width - PAD.left - PAD.right;
    const plotH = height - PAD.top - PAD.bottom;
    const x = (i) => PAD.left + (i / (n - 1)) * plotW;
    const y = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;

    const svg = el("svg", {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": options.label,
    });

    const yAxis = niceTicks(lo, hi, 4);
    for (const t of yAxis.ticks) {
      svg.appendChild(el("line", { class: "grid", x1: PAD.left, x2: width - PAD.right, y1: y(t), y2: y(t) }));
      svg.appendChild(
        el("text", { class: "tick", x: PAD.left - 6, y: y(t) + 3.5, "text-anchor": "end" }, format(t, yAxis.step))
      );
    }
    const xAxis = niceTicks(0, n - 1, 5);
    for (const t of xAxis.ticks) {
      svg.appendChild(
        el("text", { class: "tick", x: x(t), y: height - PAD.bottom + 14, "text-anchor": "middle" }, String(t))
      );
    }
    svg.appendChild(
      el("text", { class: "axis-title", x: PAD.left + plotW / 2, y: height - 3, "text-anchor": "middle" }, "bin")
    );

    for (const s of series) {
      if (options.marks === "dots") {
        const group = el("g", { fill: s.color });
        s.values.forEach((v, i) => {
          if (v !== null) group.appendChild(el("circle", { cx: x(i), cy: y(v), r: 2.2 }));
        });
        svg.appendChild(group);
      } else {
        let d = "";
        let pen = false;
        s.values.forEach((v, i) => {
          if (v === null) {
            pen = false;
            return;
          }
          d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
          pen = true;
        });
        svg.appendChild(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.5, "stroke-linejoin": "round" }));
      }
    }

    const cross = el("line", { class: "crosshair", y1: PAD.top, y2: PAD.top + plotH, visibility: "hidden" });
    svg.appendChild(cross);
    const hit = el("rect", { x: PAD.left, y: PAD.top, width: plotW, height: plotH, fill: "transparent" });
    svg.appendChild(hit);

    const tip = document.getElementById("tooltip");
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - PAD.left) / plotW) * (n - 1))));
      cross.setAttribute("x1", x(i));
      cross.setAttribute("x2", x(i));
      cross.setAttribute("visibility", "visible");
      tip.textContent = "";
      const head = document.createElement("strong");
      head.textContent = `Bin ${i}`;
      tip.appendChild(head);
      for (const s of series) {
        const row = document.createElement("div");
        const key = document.createElement("i");
        key.style.background = s.color;
        row.appendChild(key);
        const v = s.values[i];
        row.appendChild(
          document.createTextNode(`${s.name}: ${v === null || v === undefined ? "no data" : v.toFixed(4)}`)
        );
        tip.appendChild(row);
      }
      tip.hidden = false;
      const left = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = `${Math.max(8, left)}px`;
      tip.style.top = `${event.clientY + 14}px`;
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden");
      tip.hidden = true;
    });

    host.appendChild(svg);
  }

  window.Charts = { draw };
})();
