# README media

Screenshots and GIFs used by the main [README](../../README.md). Add each file with the exact name below, then uncomment its `<img>` block in the README (each one is wrapped in a `<!-- MEDIA: ... -->` comment).

## Shot list

| File | What to capture | Suggested size | Status |
|---|---|---|---|
| `demo-overlay.gif` | Hero image at the top of the README: 5–8 s of gameplay with the FPS counter, FPS graph and frame-time graph. A moment where the frame rate changes works best. | 900 px wide, 15–20 fps | to do |
| `gui-layout-editor.png` | The GUI with a video open: graphs arranged over a video frame, one graph selected in the inspector. | Full window, 1600–2000 px wide | done |
| `gui-drag-resize.gif` | Dragging a graph to a new position, then resizing it by its corner. | 900 px wide, 15 fps, under 10 s | to do |
| `hdr-scopes-grid.png` | A frame from a graphs-only export using the "HDR scopes grid" preset, from an HDR recording. | 1920×1080 | to do |
| `overlay-fps-hdr.png` | A frame from an exported video using the "FPS + HDR" preset. | 1920×1080 | done (header image until the GIF exists) |
| `overlay-fps-sdr.png` | A frame from an SDR export with the default FPS and frame-time overlay. | 1920×1080 | done |

Keep each file under about 10 MB. GitHub shows larger GIFs slowly or not at all.

## Capturing

**Screenshots of the GUI:** Win + Shift + S, or any screen recorder for the GIF.

**A frame from an exported video** (the second command tone-maps HDR exports so they don't look washed out):
```powershell
# SDR export
ffmpeg -ss 5 -i "capture_analysis.mp4" -frames:v 1 -vf "scale=1920:-1" overlay-fps-hdr.png

# HDR export
ffmpeg -ss 5 -i "capture_analysis.mp4" -frames:v 1 -vf "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:p=bt709:r=pc,format=rgb24,scale=1920:-1" overlay-fps-hdr.png
```

**A graphs-only frame** (ProRes with transparency; this flattens it onto black):
```powershell
ffmpeg -ss 2 -i "capture_graphs.mov" -frames:v 1 -vf "format=rgba" hdr-scopes-grid.png
```

**A GIF from an exported video** (two passes for a good colour palette; for an HDR export, put the tone-mapping filters from above before `fps=15`):
```powershell
ffmpeg -ss 10 -t 6 -i "capture_analysis.mp4" -vf "fps=15,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff" palette.png
ffmpeg -ss 10 -t 6 -i "capture_analysis.mp4" -i palette.png -lavfi "fps=15,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" demo-overlay.gif
```

If a GIF ends up too large, shorten it (`-t`), lower the frame rate (`fps=12`) or the width (`scale=720:-1`).
