"""Render factual conditional completion curves as a self-contained SVG.

Panels show the configured PUBLIC NOMINAL service prior (not an inferred
hidden countdown), while predictors used the same observable adaptive
expected-duration feature. Observed empirical rate is the held-out label.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from html import escape

COLORS={"Observed":"#111827","V4":"#2563eb","V3-C frozen":"#e87921",
        "B0":"#8b5cf6","Rule":"#0891b2"}
ORDER=("Observed","V4","V3-C frozen","B0","Rule")


def plot(report,output):
    data=json.loads(Path(report).read_text())
    models=data["models"]
    present=sorted({int(row["public_ticks"]) for model in models.values()
                    for row in model["conditional_curve_by_nominal"]})
    if not present:raise ValueError("no factual model conditional curves")
    panels=len(present);cols=min(3,panels);rows=(panels+cols-1)//cols
    width=cols*345+32;height=rows*250+104
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<text x="20" y="30" font-size="17" fill="#111827" font-family="sans-serif">Factual service completion by nominal public ticks</text>',
       '<text x="20" y="49" font-size="11" fill="#374151" font-family="sans-serif">Next-step event probability versus observed effective progress. Different trajectories are NOT independent event rows.</text>']
    for i,name in enumerate(ORDER):
        x=20+i*195;y=height-25
        pieces.extend([f'<line x1="{x}" y1="{y}" x2="{x+20}" y2="{y}" stroke="{COLORS[name]}" stroke-width="3"/>',
                       f'<text x="{x+25}" y="{y+4}" font-size="11" font-family="sans-serif" fill="#333">{escape(name)}</text>'])
    for idx,prior in enumerate(present):
        col=idx%cols;row=idx//cols
        x0=32+col*345;y0=78+row*250
        chart_w=305;chart_h=174
        pieces.extend([f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#d1d5db"/>',
                       f'<text x="{x0+8}" y="{y0+14}" font-size="13" font-weight="bold" font-family="sans-serif" fill="#111827">Nominal {prior} ticks</text>'])
        xmax=max(10,max(int(z["active_elapsed"]) for key in models for z in models[key]["conditional_curve_by_nominal"]
                        if int(z["public_ticks"])==prior))
        def loc(e,p):
            return x0+20+(chart_w-34)*e/xmax,y0+chart_h-20-(chart_h-46)*p
        for yv in (0,0.5,1):
            y=loc(0,yv)[1]
            pieces.append(f'<line x1="{x0+20}" y1="{y}" x2="{x0+chart_w-14}" y2="{y}" stroke="#e5e7eb"/>')
            pieces.append(f'<text x="{x0+1}" y="{y+4}" font-size="9" fill="#666" font-family="sans-serif">{yv:g}</text>')
        for step in (0,xmax//2,xmax):
            x=loc(step,0)[0]
            pieces.append(f'<text x="{x-3}" y="{y0+chart_h-5}" font-size="9" fill="#555" font-family="sans-serif">{step}</text>')
        def entry(name):
            key="V4" if name=="V4" else name
            return [z for z in models[key]["conditional_curve_by_nominal"] if int(z["public_ticks"])==prior]
        for name in ORDER:
            keys=entry("V4" if name=="Observed" else name)
            attr="observed_rate" if name=="Observed" else "mean_predicted"
            coords=[loc(int(z["active_elapsed"]),float(z[attr])) for z in keys]
            if len(coords)>1:
                points=" ".join(f"{x:.1f},{y:.1f}" for x,y in coords)
                pieces.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[name]}" stroke-width="2" opacity=".9"/>')
            for x,y in coords:
                pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{COLORS[name]}"/>')
    pieces.append('</svg>')
    out=Path(output)
    if out.exists():raise FileExistsError("figure cannot overwrite")
    out.write_text("\n".join(pieces)+"\n",encoding="utf-8")
    print(json.dumps({"scope":data["evaluation_scope"],"nominal_panels":present,
                      "figure":str(out)},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metrics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();plot(a.metrics,a.output)


if __name__=="__main__":
    main()
