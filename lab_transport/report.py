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
<div class="banner">示例数据与离散网格仿真；不代表真实实验室、物理机器人或安全认证。</div>
<div class="layout"><section class="panel"><h2>实验室地图</h2><div class="controls">
<button id="play" type="button">播放</button><input id="tick" type="range" min="0" value="0">
<strong id="time" class="mono"></strong></div><div id="grid" class="grid"></div>
<p class="foot">蓝色格为站点，橙色格表示当前站点不可用，深色格为障碍；机器人按离散时间步移动。</p>
</section><section class="panel"><h2>任务状态</h2><p id="metrics"></p><table><thead><tr><th>样本</th><th>状态</th><th>机器人</th><th>时限</th></tr></thead>
<tbody id="tasks"></tbody></table><p class="foot">时间单位为仿真步；详细事件与预测见同目录 JSON 文件。</p></section></div>
<script>
const data = __RESULT_JSON__;
const slider = document.getElementById('tick'), grid = document.getElementById('grid');
const play = document.getElementById('play'); let timer = null;
slider.max = data.history.length - 1;
document.getElementById('subtitle').textContent = data.scenario.name + ' · 调度策略：' + data.policy + ' · 随机种子：' + data.seed;
document.getElementById('metrics').textContent = '最终完成：' + data.summary.completed_count + '/' + data.summary.sample_count +
  '；超时：' + data.summary.late_count + '；结束步数：' + data.summary.ticks;
const statusNames = {unreleased:'未到达',waiting:'待分配',assigned:'前往取样',in_transit:'运输中',completed:'已送达'};
function render(index) {
  const frame = data.history[index]; slider.value = index;
  document.getElementById('time').textContent = '第 ' + frame.tick + ' 步';
  grid.replaceChildren(); grid.style.gridTemplateColumns = 'repeat(' + data.scenario.width + ', auto)';
  for (let y=0; y<data.scenario.height; y++) for (let x=0; x<data.scenario.width; x++) {
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
    for(const value of [sample.id,statusNames[sample.status]||sample.status,sample.robot_id||'—',sample.deadline_tick]){
      const td=document.createElement('td');td.textContent=String(value);row.appendChild(td)}body.appendChild(row)}
}
slider.addEventListener('input',()=>render(Number(slider.value)));
play.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return}
  play.textContent='暂停';timer=setInterval(()=>{let next=Number(slider.value)+1;
    if(next>=data.history.length){clearInterval(timer);timer=null;play.textContent='播放';return}render(next)},250)});
render(0);
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
