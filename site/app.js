(() => {
  const ratioButtons = [...document.querySelectorAll("#ratio-tabs button")];
  const placementButtons = [...document.querySelectorAll("#placement-tabs button")];
  const stack = document.getElementById("stack-visual");
  const summary = document.getElementById("stack-summary");

  let ratio = 0.25;
  let placement = "block_end";

  function attentionIndices(n, count, where) {
    if (count <= 0) return [];
    if (count >= n) return Array.from({length:n}, (_,i) => i);

    if (where === "front") return Array.from({length:count}, (_,i) => i);
    if (where === "back") return Array.from({length:count}, (_,i) => n - count + i);
    if (where === "middle") {
      const start = Math.floor((n - count) / 2);
      return Array.from({length:count}, (_,i) => start + i);
    }

    const every = n / count;
    return Array.from({length:count}, (_,j) => Math.max(0, Math.min(n - 1, Math.round((j + 1) * every - 1))));
  }

  function ratioText(r) {
    if (r === 0) return "0";
    if (r === 0.125) return "1/8";
    if (r === 0.25) return "1/4";
    if (r === 0.5) return "1/2";
    return "1";
  }

  function render() {
    const n = 16;
    const count = Math.round(n * ratio);
    const indices = new Set(attentionIndices(n, count, placement));
    stack.innerHTML = "";

    for (let i = 0; i < n; i++) {
      const layer = document.createElement("div");
      layer.className = indices.has(i) ? "layer attention" : "layer";
      layer.dataset.i = String(i + 1);
      stack.appendChild(layer);
    }

    const rec = n - count;
    const aWord = count === 1 ? "layer" : "layers";
    const rWord = rec === 1 ? "layer" : "layers";
    summary.textContent = `${count} attention ${aWord} + ${rec} recurrent ${rWord} · r = ${ratioText(ratio)}`;
    stack.setAttribute("aria-label", `Sixteen-layer hybrid stack with ${count} attention layers and ${rec} recurrent layers.`);
  }

  function activate(buttons, active) {
    buttons.forEach(b => b.setAttribute("aria-pressed", b === active ? "true" : "false"));
  }

  ratioButtons.forEach(button => {
    button.addEventListener("click", () => {
      ratio = Number(button.dataset.ratio);
      activate(ratioButtons, button);
      render();
    });
  });

  placementButtons.forEach(button => {
    button.addEventListener("click", () => {
      placement = button.dataset.placement;
      activate(placementButtons, button);
      render();
    });
  });

  render();
})();