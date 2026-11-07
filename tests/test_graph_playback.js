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
const ids=['tick','grid','play','subtitle','metrics','time','tasks','decision','forecasts','banner','map-note','deadline-header'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
vm.runInNewContext(script[1], {
  document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),
    createElementNS:(_ns,tag)=>new Element(tag)},
  window:{location:{search:'?step=1'}},URLSearchParams,setInterval:()=>1,clearInterval:()=>{},
});
const svg=elements.grid.children[0];
assert.equal(svg.tag,'svg');
const mode=process.argv[3]||'original';
const clinic=['clinic','clinic_learned','clinic_rollout'].includes(mode);
assert.equal(svg.children.filter(c=>c.tag==='circle').length,clinic?62:['nominal','learned_nominal','rule'].includes(mode)?76:32);
assert.equal(svg.children.filter(c=>c.tag==='line').length,clinic?126:['nominal','learned_nominal','rule'].includes(mode)?112:63);
assert.equal(elements.tasks.children.length,clinic?6:mode==='rule'?2:4);
if(mode==='rule'){
  assert(elements.banner.textContent.includes('按时交付不等于'));
  assert(elements['deadline-header'].textContent.includes('代理'));
  assert(elements.metrics.textContent.includes('仿真交付超过代理上限'));
}else{
  assert(elements.metrics.textContent.includes('未设置时限'));
}
if(mode==='mimic'||mode==='nominal'){
  assert(elements.banner.textContent.includes('跨机构仿真'));
  assert(elements['map-note'].textContent.includes('不是实际'));
  assert(elements.tasks.children.every(row=>row.children[4].textContent==='blood_unspecified'));
}else if(mode==='learned_nominal'){
  assert(elements.banner.textContent.includes('学习预测器在仿真数据上训练'));
  assert(elements['map-note'].textContent.includes('不是实际'));
}else if(clinic){
  assert(elements['map-note'].textContent.includes('公开诊所演示地图'));
  assert(elements['map-note'].textContent.includes('样本取送点是仿真假设'));
  if(mode==='clinic_learned')assert(elements.banner.textContent.includes('学习预测器在仿真数据上训练'));
  if(mode==='clinic_rollout')assert(elements.forecasts.children.every(item=>!item.textContent.includes('undefined')));
}else if(mode==='original'){
  assert(elements.banner.textContent.includes('功能测试设置'));
}
assert(elements.decision.textContent.includes('预测第'));
elements.tick.value=elements.tick.max;elements.tick.listeners.input();
assert(elements.tasks.children.every(row=>row.children[1].textContent==='已送达'));
console.log('Graph playback verified:',mode,'source scope, tasks and timing labels');
