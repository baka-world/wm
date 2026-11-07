// Exercise the offline report's controls without installing browser packages.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const path = process.argv[2] || 'examples/evidence/seed42-world/report.html';
const html = fs.readFileSync(path, 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/);
assert(match, 'report script is missing');

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.className = '';
    this.textContent = '';
    this.value = '0';
    this.style = {};
    this.classList = {add: name => { this.className += ' ' + name; }};
  }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
}

const ids = ['tick', 'grid', 'play', 'subtitle', 'metrics', 'time', 'tasks', 'decision', 'forecasts', 'banner', 'map-note'];
const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
let interval;
vm.runInNewContext(match[1], {
  document: {getElementById: id => elements[id], createElement: () => new Element()},
  window: {location: {search: '?step=1'}},
  URLSearchParams,
  setInterval: callback => { interval = callback; return 1; },
  clearInterval: () => {},
});
assert.equal(elements.grid.children.length, 63);
assert.equal(elements.tasks.children.length, 6);
if(process.argv[3]==='external'){
  assert(elements.decision.textContent.includes('分配给'));
  assert(elements.forecasts.children.every(item=>item.textContent.includes('可选择')&&!item.textContent.includes('undefined')));
} else assert(elements.decision.textContent.includes('预测第'));
assert(elements.forecasts.children.length > 0);
elements.tick.value = elements.tick.max;
elements.tick.listeners.input();
assert(elements.tasks.children.every(row => row.children[1].textContent === '已送达'));
elements.tick.value = '0';
elements.tick.listeners.input();
elements.play.listeners.click();
interval();
assert.equal(elements.time.textContent, '第 1 步');
console.log('Playback controls verified:',process.argv[3]==='external'?'injected policy candidates':'candidate predictions',', slider and play');
