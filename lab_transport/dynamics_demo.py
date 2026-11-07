"""Generate an offline, one-step state-prediction playback on the clinic map."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .clinic_map import make_clinic_scenario
from .dynamics_experiment import check_predicted_step, compare, merge_counts, rates, zero_counts
from .dynamics_model import Geometry, HybridDynamicsModel
from .dynamics_revision import RevisedDynamicsModel
from .multisite_experiment import make_case
from .simulation import Simulator
from .transition_schema import Transition


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>样本转运状态预测回放</title><style>
body{font-family:system-ui,sans-serif;max-width:1250px;margin:auto;padding:24px;color:#17283b;background:#f4f7fb}
h1{margin-bottom:6px}.banner{background:#fff3db;color:#60470f;padding:12px;border-radius:9px}
.controls{display:flex;gap:12px;align-items:center;margin:18px 0;flex-wrap:wrap}input{flex:1;min-width:160px}
button{background:#1769aa;color:white;border:0;padding:8px 16px;border-radius:7px}.panels{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px}
.panel{background:white;padding:12px;border:1px solid #dce4ef;border-radius:12px}.panel h2{font-size:16px;margin:4px}
svg{width:100%;min-height:190px;background:#f5f8fb}.hint{color:#526479}table{width:100%;border-collapse:collapse;background:white}
td,th{border-bottom:1px solid #e8eef4;padding:8px;text-align:left}@media(max-width:700px){.panels{grid-template-columns:1fr}}
</style></head><body>
<h1>样本转运：下一步状态预测</h1>
<p class="banner">公开诊所演示地图；取送点和任务是仿真假设。页面按当步实际分配做离线预测，不预知新任务或故障，也不代表临床或真实机器人验证。</p>
<div class="controls"><button type="button" id="play">播放</button><input type="range" id="tick" min="0" value="0"><strong id="time"></strong></div>
<p id="action" class="hint"></p><p id="external" class="hint"></p>
<div id="panels" class="panels"></div><h2>已知任务状态</h2>
<table><thead><tr><th>任务</th><th>当前</th><th>规则预测</th><th>学习预测</th><th>下一步实际</th></tr></thead><tbody id="samples"></tbody></table>
<p class="hint">机器人位置和已知任务状态是离散仿真结果；预测受到地图邻接与离散占点约束，不代表物理安全。</p>
<script>
const data=__DATA__;
const slider=document.getElementById('tick'), play=document.getElementById('play');let timer=null;
slider.max=data.rows.length-1;
const names={waiting:'待分配',assigned:'前往取样',in_transit:'运输中',completed:'已送达'};
function point(p){return p.join(',')}
function draw(panel,state){
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
 const xs=data.graph.nodes.map(n=>n.x),ys=data.graph.nodes.map(n=>n.y), minx=Math.min(...xs)-120,miny=Math.min(...ys)-120;
 svg.setAttribute('viewBox',[minx,miny,Math.max(...xs)-minx+120,Math.max(...ys)-miny+120].join(' '));
 svg.setAttribute('role','img');svg.setAttribute('aria-label','诊所示例路网与机器人位置');
 const lookup=new Map(data.graph.nodes.map(n=>[point(n.position),n]));
 for(const edge of data.graph.edges){const a=lookup.get(point(edge[0])),b=lookup.get(point(edge[1]));const line=document.createElementNS(ns,'line');
  for(const [k,v] of Object.entries({x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#9aaabc','stroke-width':6}))line.setAttribute(k,v);svg.appendChild(line)}
 for(const node of data.graph.nodes){const station=state.stations.find(s=>point(s.position)===point(node.position));
  const robot=state.robots.find(r=>point(r.position)===point(node.position));const circle=document.createElementNS(ns,'circle');
  for(const [k,v] of Object.entries({cx:node.x,cy:node.y,r:robot?28:station?22:10,fill:robot?'#1769aa':station?'#dc7d27':'#64788a'}))circle.setAttribute(k,v);
  if(robot)circle.setAttribute('class','robot');svg.appendChild(circle);
  if(robot||station){const label=document.createElementNS(ns,'text');label.setAttribute('x',node.x+30);label.setAttribute('y',node.y-18);
   label.setAttribute('font-size','45');label.textContent=[station&&station.id,robot&&robot.id].filter(Boolean).join(' ');svg.appendChild(label)}}
 panel.appendChild(svg);
}
function render(index){const row=data.rows[index];slider.value=index;
 document.getElementById('time').textContent='第 '+row.current.tick+' 步 → 第 '+row.actual.tick+' 步';
 document.getElementById('action').textContent=row.assignments.length?'本步实际分配：'+row.assignments.map(a=>a.robot_id+' → '+a.sample_id).join('，'):'本步没有新分配';
 const known=new Set(row.current.samples.map(s=>s.id)),newCount=row.actual.samples.filter(s=>!known.has(s.id)).length;
 document.getElementById('external').textContent=newCount?'下一步实际出现 '+newCount+' 个新任务，模型不预测新到任务；表格仅比较当前已知任务。':'本步无新任务需要说明；表格仅比较当前已知任务。';
 const panels=document.getElementById('panels');panels.replaceChildren();
 for(const [label,state] of [['当前状态',row.current],['规则预测的下一步',row.heuristic],['学习模型预测的下一步',row.learned],['下一步实际状态',row.actual]]){
  const panel=document.createElement('section');panel.className='panel';const title=document.createElement('h2');title.textContent=label;panel.appendChild(title);draw(panel,state);panels.appendChild(panel)}
 const body=document.getElementById('samples');body.replaceChildren();
 for(const sample of row.current.samples){const rowElement=document.createElement('tr');
  const status=(state)=>{const item=state.samples.find(s=>s.id===sample.id);return item?(names[item.status]||item.status):'未知'};
  for(const value of [sample.id,status(row.current),status(row.heuristic),status(row.learned),status(row.actual)]){const td=document.createElement('td');td.textContent=value;rowElement.appendChild(td)}body.appendChild(rowElement)}
}
slider.addEventListener('input',()=>render(Number(slider.value)));
play.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return}play.textContent='暂停';timer=setInterval(()=>{
 const next=Number(slider.value)+1;if(next>=data.rows.length){clearInterval(timer);timer=null;play.textContent='播放';return}render(next)},300)});
const requested=Number(new URLSearchParams(window.location.search).get('step')||0);
render(Number.isFinite(requested)?Math.max(0,Math.min(data.rows.length-1,Math.floor(requested))):0);
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic-map", required=True)
    parser.add_argument("--clinic-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--regime", choices=("normal", "short_outage"), default="normal")
    parser.add_argument("--policy", choices=("nearest", "world"), default="world")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        base = make_clinic_scenario(Path(args.clinic_map).read_bytes(), Path(args.clinic_config).read_bytes())
        scenario = make_case(base, "clinic", args.seed, args.regime)
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8"))
        geometry = Geometry(scenario.grid)
        learned, heuristic = RevisedDynamicsModel(geometry, parameters), HybridDynamicsModel(geometry, mode="heuristic")
        transitions: list[Transition] = []
        result = Simulator(scenario, args.policy, args.seed).run(on_transition=transitions.append)
        rows, totals = [], {"learned": zero_counts(), "heuristic": zero_counts()}
        for transition in transitions:
            if transition.next_boundary != "before_dispatch":
                continue
            predictions = {name: model.predict_next(transition.observation, transition.assignments)
                           for name, model in (("learned", learned), ("heuristic", heuristic))}
            for name, prediction in predictions.items():
                check_predicted_step(transition.observation, prediction, geometry)
                merge_counts(totals[name], compare(transition.observation, transition.next_observation, prediction))
            rows.append({"current": asdict(transition.observation), "assignments": [asdict(a) for a in transition.assignments],
                         "actual": asdict(transition.next_observation),
                         "learned": asdict(predictions["learned"]), "heuristic": asdict(predictions["heuristic"])})
        if not rows:
            raise ValueError("no nonterminal transitions to display")
        graph = {"nodes": [{"position": list(p), "x": xy[0], "y": xy[1]}
                           for p, xy in scenario.grid.coordinates.items()],
                 "edges": [[list(a), list(b)] for a, b in scenario.grid.edges]}
        data = {"graph": graph, "rows": rows}
        encoded = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        summary = {"note": "单场公开诊所演示图上的合成任务；一步离线预测，不是现场验证。",
                   "policy": args.policy, "seed": args.seed, "regime": args.regime,
                   "transition_count": len(rows), "completed": result["summary"]["completed_count"],
                   "sample_count": result["summary"]["sample_count"],
                   "one_step": {name: {"counts": counts, "rates": rates(counts)} for name, counts in totals.items()}}
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.html").write_text(PAGE.replace("__DATA__", encoded), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"completed": summary["completed"], "sample_count": summary["sample_count"],
                          "transition_count": len(rows)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"状态预测回放失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
