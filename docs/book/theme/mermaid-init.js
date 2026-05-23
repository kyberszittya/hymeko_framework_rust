// Render fenced ```mermaid blocks produced by mdBook as SVG (Mermaid 10).
document.addEventListener("DOMContentLoaded", () => {
  const blocks = document.querySelectorAll("pre code.language-mermaid");
  if (!blocks.length) {
    return;
  }

  const load = () =>
    new Promise((resolve, reject) => {
      if (window.mermaid) {
        resolve();
        return;
      }
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js";
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("mermaid CDN load failed"));
      document.head.appendChild(s);
    });

  load()
    .then(() => {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "neutral",
        flowchart: { useMaxWidth: true },
      });
      return Promise.all(
        Array.from(blocks).map(async (codeEl) => {
          const pre = codeEl.parentElement;
          if (!pre || pre.tagName !== "PRE") {
            return;
          }
          const graph = codeEl.textContent || "";
          const id = "mermaid-" + Math.random().toString(36).slice(2, 11);
          try {
            const { svg } = await window.mermaid.render(id, graph);
            const wrap = document.createElement("div");
            wrap.className = "mermaid";
            wrap.innerHTML = svg;
            pre.replaceWith(wrap);
          } catch (err) {
            console.warn("[mermaid]", err);
          }
        }),
      );
    })
    .catch((err) => console.warn("[mermaid-init]", err));
});
