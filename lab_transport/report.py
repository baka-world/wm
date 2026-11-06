"""Write reproducible machine-readable results and a self-contained playback."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17283b;background:#f3f6fa}
body{max-width:1080px;margin:0 auto;padding:24px} h1{margin-bottom:4px} p{color:#526479}
.banner{background:#fff3db;padding:12px 16px;border-radius:10px;color:#60470f}
.layout{display:grid;grid-template-columns:minmax(320px,1fr) minmax(260px,0.8fr);gap:18px;margin-top:18px}
.panel{background:white;border:1px solid #dce4ef;border-radius:14px;padding:18px;overflow:auto}
.grid{display:grid;gap:3px;width:max-content;max-width:100%;margin:14px 0}
.graph-map{width:100%;margin:14px 0}.graph-map svg{width:100%;min-height:260px;background:#f5f8fb}
.cell{width:55px;height:48px;background:#edf3f7;border-radius:5px;display:flex;flex-direction:column;
align-items:center;justify-content:center;font-size:11px;position:relative;overflow:hidden}
.cell.blocked{background:#526479}.cell.station{background:#dbeafe}.cell.unavailable{background:#ffe3d8}
.robot{background:#1769aa;color:#fff;border-radius:12px;padding:2px 5px;font-weight:bold}
.robot.r2{background:#b14c31}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button{border:0;border-radius:7px;background:#1769aa;color:white;padding:8px 15px;cursor:pointer}
input{flex:1;min-width:130px} table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;border-bottom:1px solid #e8eef4;padding:8px 4px} th{color:#53667a}
.mono{font-variant-numeric:tabular-nums}.foot{font-size:13px;color:#526479}
@media(max-width:760px){.layout{grid-template-columns:1fr}.cell{width:36px;height:38px;font-size:9px}}
</style></head><body>
<h1>样本转运仿真回放</h1><p id="subtitle"></p>
<div class="banner" id="banner">示例数据与离散网格仿真；不代表真实实验室、物理机器人或安全认证。</div>
<div class="layout"><section class="panel"><h2>实验室地图</h2><div class="controls">
<button id="play" type="button">播放</button><input id="tick" type="range" min="0" value="0">
<strong id="time" class="mono"></strong></div><div id="grid" class="grid"></div>
<p class="foot" id="map-note">蓝色格为站点，橙色格表示当前站点不可用，深色格为障碍；机器人按离散时间步移动。</p>
</section><section class="panel"><h2>任务状态</h2><p id="metrics"></p><table><thead><tr><th>样本</th><th>状态</th><th>机器人</th><th>时限</th></tr></thead>
<tbody id="tasks"></tbody></table><h2>最近调度决策</h2><p id="decision"></p>
<ul id="forecasts"></ul><p class="foot">预测基于当时已知状态；详细事件见同目录 JSON 文件。时间单位为仿真步。</p></section></div>
<script>
const data = __RESULT_JSON__;
const slider = document.getElementById('tick'), grid = document.getElementById('grid');
const play = document.getElementById('play'); let timer = null;
slider.max = data.history.length - 1;
document.getElementById('subtitle').textContent = data.scenario.name + ' · 调度策略：' + data.policy + ' · 随机种子：' + data.seed;
document.getElementById('metrics').textContent = '最终完成：' + data.summary.completed_count + '/' + data.summary.sample_count +
  '；超时：' + (data.summary.late_count===null?'未设置时限':data.summary.late_count) + '；结束步数：' + data.summary.ticks;
if(data.scenario.graph){
  document.getElementById('banner').textContent='公开医院路网的拓扑仿真；任务、双机器人与计时为功能测试设置，不代表医院实测或安全认证。';
  document.getElementById('map-note').textContent='箭头为原始路网允许的方向；连线只表示节点连接，每条边按一个抽象步计算，并非真实行驶时间。';
}
const statusNames = {unreleased:'未到达',waiting:'待分配',assigned:'前往取样',in_transit:'运输中',completed:'已送达'};
function renderGraph(frame){
  const ns='http://www.w3.org/2000/svg', source=data.scenario.graph;
  const nodeFor=p=>source.nodes.find(n=>n.position[0]===p[0]&&n.position[1]===p[1]);
  const svg=document.createElementNS(ns,'svg');
  const xs=source.nodes.map(n=>n.x), ys=source.nodes.map(n=>n.y);
  const minx=Math.min(...xs)-120, miny=Math.min(...ys)-120;
  svg.setAttribute('viewBox',[minx,miny,Math.max(...xs)-minx+120,Math.max(...ys)-miny+120].join(' '));
  svg.setAttribute('role','img');svg.setAttribute('aria-label','医院路网及机器人状态');
  const defs=document.createElementNS(ns,'defs'), marker=document.createElementNS(ns,'marker');
  for(const [key,value] of Object.entries({id:'arrow',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'5',markerHeight:'5',orient:'auto-start-reverse'}))marker.setAttribute(key,value);
  const arrow=document.createElementNS(ns,'path');arrow.setAttribute('d','M 0 0 L 10 5 L 0 10 z');arrow.setAttribute('fill','#718096');marker.appendChild(arrow);defs.appendChild(marker);svg.appendChild(defs);
  for(const edge of source.edges){const a=nodeFor(edge[0]),b=nodeFor(edge[1]);const line=document.createElementNS(ns,'line');
    for(const [key,value] of Object.entries({x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#718096','stroke-width':'7','marker-end':'url(#arrow)'}))line.setAttribute(key,value);svg.appendChild(line)}
  for(const node of source.nodes){const station=data.scenario.stations.find(s=>s.position[0]===node.position[0]&&s.position[1]===node.position[1]);
    const robot=frame.robots.find(r=>r.position[0]===node.position[0]&&r.position[1]===node.position[1]);
    const circle=document.createElementNS(ns,'circle');circle.setAttribute('cx',node.x);circle.setAttribute('cy',node.y);
    circle.setAttribute('r',robot?'28':station?'23':'12');circle.setAttribute('fill',robot?'#1769aa':station?'#dc7d27':'#425b72');svg.appendChild(circle);
    if(robot||station){const label=document.createElementNS(ns,'text');label.setAttribute('x',node.x+30);label.setAttribute('y',node.y-22);
      label.setAttribute('font-size','55');label.setAttribute('fill','#17283b');label.textContent=[station&&station.id,robot&&robot.id].filter(Boolean).join(' ');svg.appendChild(label)}}
  grid.className='graph-map';grid.appendChild(svg);
}
function render(index) {
  const frame = data.history[index]; slider.value = index;
  document.getElementById('time').textContent = '第 ' + frame.tick + ' 步';
  grid.replaceChildren(); grid.style.gridTemplateColumns = 'repeat(' + data.scenario.width + ', auto)';
  if(data.scenario.graph)renderGraph(frame);
  else for (let y=0; y<data.scenario.height; y++) for (let x=0; x<data.scenario.width; x++) {
    const cell=document.createElement('div'); cell.className='cell';
    if (data.scenario.blocked.some(p=>p[0]===x&&p[1]===y)) cell.classList.add('blocked');
    const station=data.scenario.stations.find(s=>s.position[0]===x&&s.position[1]===y);
    if(station){cell.classList.add('station');const state=frame.stations.find(s=>s.id===station.id);
      if(state&&!state.available)cell.classList.add('unavailable');const label=document.createElement('span');
      label.textContent=station.id;cell.appendChild(label);}
    const robot=frame.robots.find(r=>r.position[0]===x&&r.position[1]===y);
    if(robot){const label=document.createElement('span');label.className='robot '+(robot.id==='R2'?'r2':'');
      label.textContent=robot.id;cell.appendChild(label);}
    grid.appendChild(cell);
  }
  const body=document.getElementById('tasks');body.replaceChildren();
  for(const sample of frame.samples){const row=document.createElement('tr');
    for(const value of [sample.id,statusNames[sample.status]||sample.status,sample.robot_id||'—',sample.deadline_tick===null?'未设置':sample.deadline_tick]){
      const td=document.createElement('td');td.textContent=String(value);row.appendChild(td)}body.appendChild(row)}
  const decisions=data.events.filter(e=>e.type==='assigned'&&e.tick<frame.tick), latest=decisions[decisions.length-1];
  const decision=document.getElementById('decision'), forecasts=document.getElementById('forecasts');forecasts.replaceChildren();
  if(!latest){decision.textContent='该时刻尚无已执行的任务分配。';return}
  const selected=latest.selected, sample=data.samples.find(s=>s.id===latest.sample_id);
  let message='第 '+latest.tick+' 步：将 '+latest.sample_id+' 分配给 '+latest.robot_id+'。';
  if(latest.policy==='world') {message+=' 预测第 '+selected.expected_completion_tick+' 步完成。';
    if(selected.predicted_lateness!==null)message+=' 预计逾期 '+selected.predicted_lateness+' 步。';}
  if(sample&&sample.completed_tick!==null&&sample.completed_tick<=frame.tick) message+=' 实际第 '+sample.completed_tick+' 步完成。';
  decision.textContent=message;
  for(const candidate of latest.candidates){const item=document.createElement('li');
    item.textContent=candidate.robot_id+' / '+candidate.sample_id+(latest.policy==='world'
      ? '：预测完成 '+candidate.expected_completion_tick+' 步，评分 '+candidate.score
      : '：距取样点 '+candidate.distance_to_pickup+' 格');forecasts.appendChild(item)}
}
slider.addEventListener('input',()=>render(Number(slider.value)));
play.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return}
  play.textContent='暂停';timer=setInterval(()=>{let next=Number(slider.value)+1;
    if(next>=data.history.length){clearInterval(timer);timer=null;play.textContent='播放';return}render(next)},250)});
const requested=Number(new URLSearchParams(window.location.search).get('step')||0);
render(Number.isFinite(requested)?Math.max(0,Math.min(data.history.length-1,Math.floor(requested))):0);
</script></body></html>'''


def check_output_directory(output: Path) -> None:
    for name in ("result.json", "summary.json", "events.jsonl", "samples.csv", "report.html"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output / name}")


def write_run(result: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    check_output_directory(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in result["events"]),
                                       encoding="utf-8")
    with (path / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["samples"][0]))
        writer.writeheader()
        writer.writerows(result["samples"])
    encoded = json.dumps(result, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = _HTML.replace("__TITLE__", html.escape(result["scenario"]["name"])).replace("__RESULT_JSON__", encoded)
    (path / "report.html").write_text(page, encoding="utf-8")
