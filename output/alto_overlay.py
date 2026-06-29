"""Render a per-page ALTO file as a standalone HTML overlay.

The ALTO layout is drawn on a canvas sized to the <Page> WIDTH/HEIGHT, with
every <TextLine> as a faint box and every <String> as a positioned span carrying
its CONTENT text. Strings that carry a TAGREFS are colour-coded by NER label
(the <OtherTag LABEL>), with a hover tooltip showing the tag's CT URI and KB id.

If the page scan image is available on disk, it is embedded as the canvas
background (base64 data URI, so the HTML is self-contained). The image is
resolved by the <Page IMAGE> attribute's stem — trying .png then .jpg — in the
ALTO file's own directory (overridable with --image). This shows the boxes
landed on the real handwriting.

Usage:
    python3 output/alto_overlay.py [alto_file] [out.html] [--image PATH]
    python3 output/alto_overlay.py            # renders all 10 third-letter pages
"""
import argparse
import base64
import glob
import html
import math
import mimetypes
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
A = "{" + ALTO_NS + "}"

# Label -> (border colour, fill colour). NER labels emitted by the pipeline.
LABEL_COLOURS = {
    "E53_Place":               ("#1b6ec2", "rgba(27,110,194,.10)"),
    "E22_Human-Made_Object":    ("#b35900", "rgba(179,89,0,.10)"),
    "Mode_of_Transportation":   ("#b35900", "rgba(179,89,0,.10)"),
    "E21_Person":               ("#8e44ad", "rgba(142,68,173,.10)"),
    "E39_Actor":                ("#8e44ad", "rgba(142,68,173,.10)"),
    "E74_Group":                ("#8e44ad", "rgba(142,68,173,.10)"),
    "E52_Time-Span":            ("#c0392b", "rgba(192,57,43,.10)"),
    "E30_Right":                ("#16a085", "rgba(22,160,133,.10)"),
    "E89_Propositional_Object": ("#16a085", "rgba(22,160,133,.10)"),
}
DEFAULT_COLOUR = ("#555", "rgba(85,85,85,.06)")


def _tag_map(root):
    """OtherTag ID -> (label, description, uri)."""
    tm = {}
    for ot in root.iter(f"{A}OtherTag"):
        tm[ot.get("ID")] = (ot.get("LABEL"), ot.get("DESCRIPTION"), ot.get("URI"))
    return tm


def _resolve_image(alto_path, page_image, override):
    """Find the scan image for this page. Tries the override, then the <Page
    IMAGE> stem with .png and .jpg in the ALTO file's directory."""
    cand = []
    if override:
        cand.append(Path(override))
    stem = Path(page_image).stem if page_image else None
    d = Path(alto_path).parent
    if stem:
        for ext in (".png", ".jpg", ".jpeg"):
            cand.append(d / f"{stem}{ext}")
    for c in cand:
        if c and c.exists():
            return c
    return None


def _data_uri(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _baseline_points(tl):
    """TextLine BASELINE="x,y x,y ..." -> list of (x, y) floats, or None."""
    bl = tl.get("BASELINE")
    if not bl:
        return None
    pts = []
    for tok in bl.split():
        xs, ys = tok.split(",")
        pts.append((float(xs), float(ys)))
    return pts or None


def _baseline_y(pts, x):
    """Linear-interpolate the baseline polyline y at column x (ALTO px).
    Clamps to the end points' y outside the polyline's x-range."""
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (px0, py0), (px1, py1) in zip(pts, pts[1:]):
        if px0 <= x <= px1:
            return py0 + (py1 - py0) * ((x - px0) / (px1 - px0) if px1 > px0 else 0)
    return pts[-1][1]


def render(alto_path, out_path, scale=0.5, image_override=None, show_text=True,
           font_min=9, font_max=16, font_fit=0.7, show_baseline=False,
           baseline_colour="#e00000", show_boxes=True, show_lineboxes=True,
           text_on_baseline=False, image_opacity=0.9, text_opacity=0.55):
    tree = ET.parse(alto_path)
    root = tree.getroot()
    page = root.find(f".//{A}Page")
    W = int(float(page.get("WIDTH") or 0))
    H = int(float(page.get("HEIGHT") or 0))
    phys = page.get("PHYSICAL_IMG_NR")
    img = page.get("IMAGE")
    tags = _tag_map(root)
    img_path = _resolve_image(alto_path, img, image_override)

    lines = []
    tagged = 0
    for tl in root.iter(f"{A}TextLine"):
        for st in tl.findall(f"{A}String"):
            x = float(st.get("HPOS")); y = float(st.get("VPOS"))
            w = float(st.get("WIDTH")); h = float(st.get("HEIGHT"))
            content = st.get("CONTENT") or ""
            tids = (st.get("TAGREFS") or "").split()
            if tids:
                tagged += 1
            lines.append((x, y, w, h, content, tids))

    # Legend: which labels appear on this page.
    labels_used = {}
    for _, _, _, _, _, tids in lines:
        for t in tids:
            lab = tags.get(t, (None, None, None))[0] or "?"
            labels_used[lab] = labels_used.get(lab, 0) + 1

    sw, sh = W * scale, H * scale
    bg = ""
    if img_path:
        uri = _data_uri(img_path)
        bg = (f"<img src='{uri}' style='position:absolute;left:0;top:0;"
              f"width:{sw:.0f}px;height:{sh:.0f}px;opacity:{image_opacity}'>")
        bg_note = f"image: {img_path.name} (embedded)"
    else:
        bg_note = "image: NOT FOUND (blank canvas)"

    font_size = 9 * scale if show_text else 0  # legacy default; overridden per-string
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>ALTO overlay — {html.escape(Path(alto_path).name)}</title>",
        "<style>",
        f"body{{font-family:monospace;margin:20px;background:#eee}}",
        f".canvas{{position:relative;width:{sw:.0f}px;height:{sh:.0f}px;"
        f"background:#fff;box-shadow:0 0 8px rgba(0,0,0,.25);overflow:hidden}}",
        f".str{{position:absolute;box-sizing:border-box;border:1px solid transparent;"
        f"display:flex;align-items:center;overflow:visible;white-space:nowrap}}",
        f".label{{line-height:1;color:#000;"
        f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;font-weight:600}}",
        f".tagged{{border-width:1.5px}}",
        ".linebox{position:absolute;border:1px dashed rgba(0,0,0,.10)}",
        ".bl-label{position:absolute;white-space:nowrap;line-height:1;color:#000;"
        "text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;font-weight:600}",
        ".legend{margin-top:12px;font-size:13px}",
        ".legend span{display:inline-block;margin-right:14px;padding:2px 6px;border:1.5px solid}",
        "h2{font-size:16px}",
        "</style></head><body>",
        f"<h2>{html.escape(Path(alto_path).name)} — page {phys} ({img})</h2>",
        f"<div>page size: {W}×{H}, strings: {len(lines)}, tagged: {tagged}, {bg_note}</div>",
        f"<div class='canvas'>",
        bg,
    ]
    # TextLine faint boxes (behind strings)
    if show_lineboxes:
        for tl in root.iter(f"{A}TextLine"):
            x = float(tl.get("HPOS")); y = float(tl.get("VPOS"))
            w = float(tl.get("WIDTH")); h = float(tl.get("HEIGHT"))
            parts.append(
                f"<div class='linebox' style='left:{x*scale:.1f}px;top:{y*scale:.1f}px;"
                f"width:{w*scale:.1f}px;height:{h*scale:.1f}px'></div>")
    # Strings
    if text_on_baseline and show_text:
        # Plot each word ON its TextLine's BASELINE: interpolate the polyline at
        # the word's start/end columns to get the local slant, then rotate the
        # label so its baseline sits on the polyline (handles slanted lines).
        for tl in root.iter(f"{A}TextLine"):
            pts = _baseline_points(tl)
            if not pts:
                continue
            for st in tl.findall(f"{A}String"):
                x0 = float(st.get("HPOS"))
                w = float(st.get("WIDTH"))
                h = float(st.get("HEIGHT"))
                x1 = x0 + w
                content = st.get("CONTENT") or ""
                y0 = _baseline_y(pts, x0)
                y1 = _baseline_y(pts, x1)
                if y0 is None or y1 is None:
                    continue
                angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
                fs = max(font_min, min(h * scale * font_fit, font_max))
                # Shift up by the label height so the box bottom (the pivot) sits
                # on the baseline polyline; rotate so the word follows the slope.
                top = y0 * scale - fs * 0.85
                style = (f"position:absolute;left:{x0*scale:.1f}px;"
                         f"top:{top:.1f}px;transform-origin:left bottom;"
                         f"transform:rotate({angle:.2f}deg)'")
                parts.append(
                    f"<div class='bl-label' style={style}>"
                    f"<span style='font-size:{fs:.1f}px;opacity:{text_opacity}'>"
                    f"{html.escape(content)}</span>"
                    f"</div>")
    else:
        for x, y, w, h, content, tids in lines:
            if tids and show_boxes:
                lab = tags.get(tids[0], (None, None, None))[0] or "?"
                bc, fc = LABEL_COLOURS.get(lab, DEFAULT_COLOUR)
                tip_parts = []
                for t in tids:
                    lg, desc, uri = tags.get(t, ("?","?","?"))
                    tip_parts.append(f"{t}  [{lg}]  {desc or ''}  {uri or ''}")
                tip = html.escape(" | ".join(tip_parts))
                cls = "str tagged"
                style = (f"left:{x*scale:.1f}px;top:{y*scale:.1f}px;width:{w*scale:.1f}px;"
                         f"height:{h*scale:.1f}px;border-color:{bc};background:{fc}' "
                         f"title='{tip}'")
            else:
                cls = "str"
                style = (f"left:{x*scale:.1f}px;top:{y*scale:.1f}px;width:{w*scale:.1f}px;"
                         f"height:{h*scale:.1f}px' ")
            inner = ""
            if show_text:
                # Size the label to its own box height so the transcription fills the
                # box — readable when zoomed in to check alignment against the scan.
                fs = max(font_min, min(h * scale * font_fit, font_max))
                inner = (f"<div class='label' style='font-size:{fs:.1f}px;"
                         f"opacity:{text_opacity}'>{html.escape(content)}</div>")
            parts.append(f"<div class='{cls}' style={style}>{inner}</div>")
    # Baselines (the ALTO BASELINE polyline on each TextLine) drawn on top, so
    # you can tell boxes from baselines and judge whether they line up.
    if show_baseline:
        polys = []
        for tl in root.iter(f"{A}TextLine"):
            bl = tl.get("BASELINE")
            if not bl:
                continue
            pts = []
            for tok in bl.split():
                xs, ys = tok.split(",")
                pts.append(f"{float(xs)*scale:.1f},{float(ys)*scale:.1f}")
            polys.append(
                f"<polyline points='{' '.join(pts)}' fill='none' "
                f"stroke='{baseline_colour}' stroke-width='1.2'/>")
        if polys:
            parts.append(
                f"<svg style='position:absolute;left:0;top:0;width:{sw:.0f}px;"
                f"height:{sh:.0f}px;pointer-events:none'>{''.join(polys)}</svg>")
    parts.append("</div>")

    # Legend
    parts.append("<div class='legend'>")
    for lab, n in sorted(labels_used.items(), key=lambda kv: -kv[1]):
        bc, _ = LABEL_COLOURS.get(lab, DEFAULT_COLOUR)
        parts.append(f"<span style='border-color:{bc}'>{html.escape(lab)} ({n})</span>")
    parts.append("</div></body></html>")

    Path(out_path).write_text("".join(parts), encoding="utf-8")
    return len(lines), tagged, bool(img_path)


def main():
    ap = argparse.ArgumentParser(description="Render an ALTO file as an HTML overlay.")
    ap.add_argument("alto_file", nargs="?", help="one ALTO file (default: all 10)")
    ap.add_argument("out_html", nargs="?", help="output HTML (default: <alto>.overlay.html)")
    ap.add_argument("--image", help="path to the page scan image (overrides <Page IMAGE>)")
    ap.add_argument("--no-text", action="store_true", help="draw boxes only, no text labels")
    ap.add_argument("--scale", type=float, default=0.5, help="display scale (default 0.5)")
    ap.add_argument("--font-min", type=float, default=11, help="min label font size in px (default 11)")
    ap.add_argument("--font-max", type=float, default=24, help="max label font size in px (default 24)")
    ap.add_argument("--font-fit", type=float, default=0.8,
                    help="label font = box height * this factor, clamped to min/max (default 0.8)")
    ap.add_argument("--baseline", action="store_true",
                    help="also draw each TextLine BASELINE polyline on top (red)")
    ap.add_argument("--no-boxes", action="store_true",
                    help="draw no String/TextLine boxes (use with --baseline for a "
                         "baselines-only view; text labels are still drawn)")
    ap.add_argument("--on-baseline", action="store_true",
                    help="plot each word rotated onto its TextLine BASELINE polyline "
                         "(follows slanted lines) instead of a horizontal box label")
    ap.add_argument("--image-opacity", type=float, default=0.9,
                    help="opacity of the background scan (default 0.9)")
    ap.add_argument("--text-opacity", type=float, default=0.55,
                    help="opacity of the transcription labels (default 0.55)")
    args = ap.parse_args()

    if args.alto_file:
        files = [args.alto_file]
        outs = [args.out_html or str(Path(args.alto_file).with_suffix(".overlay.html"))]
    else:
        files = sorted(glob.glob(str(ROOT / "output" / "alto"
                                     / "1816_third_letter_scan*.alto.xml")))
        outs = [str(Path(f).with_suffix(".overlay.html")) for f in files]
    for f, out in zip(files, outs):
        n, t, has_img = render(f, out, scale=args.scale, image_override=args.image,
                               show_text=not args.no_text, font_min=args.font_min,
                               font_max=args.font_max, font_fit=args.font_fit,
                               show_baseline=args.baseline,
                               show_boxes=not args.no_boxes,
                               show_lineboxes=not args.no_boxes,
                               text_on_baseline=args.on_baseline,
                               image_opacity=args.image_opacity,
                               text_opacity=args.text_opacity)
        img_note = "with image" if has_img else "no image"
        print(f"{Path(f).name} -> {Path(out).name}  ({n} strings, {t} tagged, {img_note})")


if __name__ == "__main__":
    main()