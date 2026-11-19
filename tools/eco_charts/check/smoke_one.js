// usage: node smoke_one.js PAGE_HTML GAME_DATA  -> runs the replay page headless with one game's data (its JSON, or its
// site/eco/data/<game>.js), exercising every control. Needs jsdom (npm install jsdom).
const fs = require('fs'); const { JSDOM, VirtualConsole } = require('jsdom');
const [page, data] = process.argv.slice(2); const raw = fs.readFileSync(data, 'utf8'), G = JSON.parse(data.endsWith('.js') ? raw.slice(raw.indexOf('"] = ') + 5, raw.lastIndexOf('}') + 1) : raw); const errors = [];
const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(String(e.message || e)));
let html = fs.readFileSync(page, 'utf8').replace(/<link[^>]*>/g, ''); const i = html.indexOf('<script>');
html = html.slice(0, i) + '<script>window.__INLINE__ = {"' + G.id + '": ' + JSON.stringify(G).replace(/<\//g, '<\\/') + '};</script>' + html.slice(i);
const dom = new JSDOM('<!doctype html><html><head></head><body>' + html + '</body></html>', { runScripts: 'dangerously', virtualConsole: vc, pretendToBeVisual: true, url: 'https://x.test/',
  beforeParse(w) { Object.defineProperty(w.HTMLElement.prototype, 'clientWidth', { get() { return 760; } });
    w.SVGElement.prototype.getBoundingClientRect = () => ({ left: 0, top: 0, width: 760, height: 260 }); w.addEventListener('error', e => errors.push(String(e.message))); } });
const w = dom.window, d = w.document, sleep = ms => new Promise(r => setTimeout(r, ms)), ev = (t, x) => new w.MouseEvent(t, { clientX: x, clientY: 60, bubbles: true });
(async () => { await sleep(50); await w.__tradingReplay.show(G.id); await sleep(30);
  const res = { id: G.id, steps: G.steps.length };
  const svgs = [...d.querySelectorAll('#charts svg')]; res.panels = d.querySelectorAll('#charts .panel').length;
  res.svgElements = svgs.map(s => s.querySelectorAll('rect,path,circle,line,text').length);
  for (const s of svgs.filter(s => s.classList.contains('plot'))) { s.dispatchEvent(ev('mousemove', 380)); s.dispatchEvent(ev('mousedown', 380)); s.dispatchEvent(ev('mouseup', 380)); }
  const plots = d.querySelectorAll('#charts svg.plot'); if (plots[1]) { plots[1].dispatchEvent(ev('mousedown', 150)); plots[1].dispatchEvent(ev('mousemove', 600)); plots[1].dispatchEvent(ev('mouseup', 600)); }
  res.zoomWorks = !d.getElementById('reset').disabled || G.x.n < 3; d.getElementById('reset').click();
  d.querySelectorAll('.legend .item').forEach(x => { x.click(); x.click(); });
  const seats = d.getElementById('seats'); if (seats) { [...seats.children].forEach(b => { b.click(); d.getElementById('next').click(); }); seats.children[0].click(); }
  d.querySelectorAll('.ttabs button').forEach(b => b.click());
  for (let k = 0; k < Math.min(G.steps.length, 40); k++) d.getElementById('next').click();
  w.__tradingReplay.play(); await sleep(1200); w.__tradingReplay.stop();
  res.ticketEmpty = ['head', 'pill', 'chips', 'state', 'count'].filter(k => !(d.getElementById(k) || {}).textContent);
  const view = d.getElementById('view').innerHTML; res.nanOrUndefined = (view.match(/NaN|undefined|Infinity|\[object Object\]/g) || []).length;
  res.nonEnglish = (view.match(/[　-〿㐀-鿿＀-￯]/g) || []).length; res.errors = [...new Set(errors)].slice(0, 5);
  res.ok = !res.errors.length && !res.nanOrUndefined && !res.nonEnglish && !res.ticketEmpty.length; console.log(JSON.stringify(res)); })();
