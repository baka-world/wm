// Check the four-panel offline state-prediction playback with no browser packages.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
assert(html.includes('取送点和任务是仿真假设'));
assert(html.includes('不预知新任务或故障'));
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert(script, 'missing playback script');
const encoded = script[1].match(/const data=(\{.*\});/);
assert(encoded, 'missing embedded prediction data');
const data = JSON.parse(encoded[1]);
assert.equal(data.graph.nodes.length, 62);
assert.equal(data.graph.edges.length, 126);
assert(data.rows.length > 0);
for(const row of data.rows){
  const ids = state => state.samples.map(s=>s.id).sort().join(',');
  assert.equal(ids(row.current),ids(row.learned));
  assert.equal(ids(row.current),ids(row.heuristic));
  assert.equal(new Set(row.learned.robots.map(r=>r.position.join(','))).size,row.learned.robots.length);
  assert.equal(row.learned.tick,row.current.tick+1);
}
class Element {
  constructor(tag=''){this.tag=tag;this.children=[];this.listeners={};this.attributes={};this.textContent='';this.value='0';this.className='';}
  appendChild(child){this.children.push(child)}
  replaceChildren(){this.children=[]}
  addEventListener(name,callback){this.listeners[name]=callback}
  setAttribute(name,value){this.attributes[name]=String(value)}
}
const ids=['tick','play','time','action','external','panels','samples'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
let interval;
vm.runInNewContext(script[1],{
  document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(_ns,tag)=>new Element(tag)},
  window:{location:{search:'?step=1'}},URLSearchParams,
  setInterval:callback=>{interval=callback;return 1},clearInterval:()=>{},
});
assert.equal(elements.panels.children.length,4);
for(const panel of elements.panels.children){
  const svg=panel.children.find(item=>item.tag==='svg');
  assert(svg);
  assert.equal(svg.children.filter(item=>item.tag==='circle').length,62);
  assert.equal(svg.children.filter(item=>item.tag==='line').length,126);
  assert.equal(svg.children.filter(item=>item.attributes.class==='robot').length,2);
}
assert(elements.time.textContent.includes('→'));
assert(elements.samples.children.length>0);
elements.tick.value='0';elements.tick.listeners.input();
elements.play.listeners.click();interval();
assert(elements.time.textContent.includes('第 1 步'));
console.log('State-prediction playback verified: four panels, known tasks, slider and play');
