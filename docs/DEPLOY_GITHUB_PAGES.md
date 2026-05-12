# Deploying the handbook (GitHub Pages & elsewhere)

## GitHub Pages (already wired)

This repository includes **`.github/workflows/docs.yml`**, which on every push to `main` / `master` (when relevant paths change) or on **manual dispatch**:

1. Builds **mdBook** from `docs/book/`
2. Builds **Rust API** docs (`cargo doc`) into `_site/api/`
3. Copies the **WASM editor** and **demo** into `_site/editor/` and `_site/demo/`
4. Uploads the combined **`_site/`** tree to **GitHub Pages**

### One-time GitHub setup

1. Repo **Settings → Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).
3. After the first successful workflow run, Pages shows the public URL (usually  
   `https://<owner>.github.io/<repo>/`  
   e.g. `https://kyberszittya.github.io/hymeko_framework_rust/` if that is your owner/repo name).

### Manual run

**Actions → “Deploy docs to GitHub Pages” → Run workflow.**

### What is published where

| URL path | Content |
|----------|---------|
| `/` | mdBook (handbook, including **Results & evidence**) |
| `/api/` | `cargo doc` workspace output |
| `/editor/` | WASM editor |
| `/demo/` | WASM demo |

### Triggers (paths)

The workflow runs when `docs/book/**`, **`docs/SOTA_RESULTS.md`**, **`docs/RESULTS_DISCIPLINE.md`**, **`COLD_START.md`**, editor/demo, Rust sources, or the workflow file itself change. If you edit only `signedkan_wip/` results JSON, add that path to `docs.yml` or use **workflow_dispatch**.

---

## Local “always on” service

Static site — no backend required.

```bash
cd docs/book
mdbook serve --hostname 0.0.0.0 --port 3000
```

Bind behind **nginx**, **Caddy**, or **Tailscale** if you need TLS or a stable hostname. Same `_site` output as `mdbook build` can be copied to any static host (**Netlify**, **Cloudflare Pages**, S3+CloudFront, etc.).

---

## Offline / air-gapped builds

Mermaid diagrams in the handbook load **jsDelivr** in the browser. For fully offline hosting, vendor Mermaid into `docs/book/theme/` and point `mermaid-init.js` at a relative path (follow-up change).
