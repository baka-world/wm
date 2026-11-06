// Exercise the source-backed graph report without installing browser packages.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert(script, 'report script is missing');
class Element {
  constructor(tag = '') {this.tag = tag;this.children=[];this.listeners={};this.attributes={};this.textContent='';this.value='0';this.style={};
    this.classList={add:()=>{}};}
  appendChild(child){this.children.push(child)}
  replaceChildren(){this.children=[]}
  addEventListener(name, callback){this.listeners[name]=callback}
  setAttribute(name, value){this.attributes[name]=String(value)}
}
const ids=['tick','grid','play','subtitle','metrics','time','tasks','decision','forecasts','banner','map-note'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
vm.runInNewContext(script[1], {
  document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),
    createElementNS:(_ns,tag)=>new Element(tag)},
  window:{location:{search:'?step=1'}},URLSearchParams,setInterval:()=>1,clearInterval:()=>{},
});
const svg=elements.grid.children[0];
assert.equal(svg.tag,'svg');
assert.equal(svg.children.filter(c=>c.tag==='circle').length,32);
assert.equal(svg.children.filter(c=>c.tag==='line').length,63);
assert.equal(elements.tasks.children.length,4);
assert(elements.metrics.textContent.includes('未设置时限'));
if(process.argv[3]==='mimic'){
  assert(elements.banner.textContent.includes('跨机构仿真'));
  assert(elements['map-note'].textContent.includes('不是实际'));
  assert(elements.tasks.children.every(row=>row.children[4].textContent==='blood_unspecified'));
}else{
  assert(elements.banner.textContent.includes('功能测试设置'));
}
assert(elements.decision.textContent.includes('预测第'));
elements.tick.value=elements.tick.max;elements.tick.listeners.input();
assert(elements.tasks.children.every(row=>row.children[1].textContent==='已送达'));
console.log('Graph playback verified: 32 waypoints, 63 directed lanes, tasks and missing deadlines');
