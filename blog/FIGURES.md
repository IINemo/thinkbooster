# Article figures — how they were made

Medium only accepts raster images, so every figure in `ARTICLE.md` is exported to PNG
here. Recipes below so they are reproducible.

## Plots (rasterized from the paper's PDF figures)

```bash
IMG="../_ACL_2026__ThinkBooster_demo (1)/images"
pdftoppm -png -r 200 -singlefile "$IMG/endpoint.pdf"             endpoint_hero
pdftoppm -png -r 220 -singlefile "$IMG/qwen3_humaneval_ratio.pdf"  fig_qwen3_humaneval
pdftoppm -png -r 220 -singlefile "$IMG/qwen25_aggregate_ratio.pdf" fig_qwen25_aggregate
```

| PNG | Article slot |
|-----|--------------|
| `endpoint_hero.png` | hero |
| `fig_qwen3_humaneval.png` | §4, after the "confidence beats the reward model" paragraph |
| `fig_qwen25_aggregate.png` | §4, after the "spending more" paragraph |

## Tables (rebuilt as styled HTML, screenshotted)

The academic LaTeX tables don't read well in a blog, so they're rebuilt as clean HTML
(`fig_gpt_oss.html`, `fig_ttc_frameworks.html`) and screenshotted with headless Chrome at
2x for crisp text. Edit the HTML and re-run:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --default-background-color=FFFFFFFF --screenshot=fig_gpt_oss.png \
  --window-size=1000,348 "file://$PWD/fig_gpt_oss.html"
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --default-background-color=FFFFFFFF --screenshot=fig_ttc_frameworks.png \
  --window-size=1240,660 "file://$PWD/fig_ttc_frameworks.html"
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --default-background-color=FFFFFFFF --screenshot=fig_strategies.png \
  --window-size=1080,628 "file://$PWD/fig_strategies.html"
```

| PNG | Article slot |
|-----|--------------|
| `fig_gpt_oss.png` | §4, after the "drop-in change / CUDA kernels" paragraph |
| `fig_ttc_frameworks.png` | §6, framework comparison |
| `fig_strategies.png` | §2, the nine strategies |

## Debugger figures (from the paper's composited PDFs)

```bash
IMG="../_ACL_2026__ThinkBooster_demo (1)/images"
pdftoppm -png -r 200 -singlefile "$IMG/main-1-new.pdf" fig_debugger_a
pdftoppm -png -r 200 -singlefile "$IMG/main-2-new.pdf" fig_debugger_b
# then trim the card whitespace to content with Pillow:
#   diff vs white, threshold >28, crop to getbbox()+14px margin
```

| PNG | View |
|-----|------|
| `fig_debugger_a.png` | the debugger: run config, strategy comparison, reasoning timeline, step inspector |
| `fig_debugger_b.png` | trajectory tree (selected path in orange) |

The individual raw screenshots `images/demo-config.png` / `demo-result.png` /
`demo-treeview.png` are alternatives if you prefer one view per image.
