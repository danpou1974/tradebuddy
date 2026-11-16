// Analiza las billeteras de PRESET_WATCH reutilizando la lógica exacta del dashboard (index.html).
// Salida: results/analysis.json y results/summary.csv
import fs from 'fs';
import vm from 'vm';

const html = fs.readFileSync('index.html', 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];
const ctx = { console, fetch: globalThis.fetch, setTimeout, Math, Date, JSON, Map, Set, Promise };
vm.createContext(ctx);
vm.runInContext(js + '\n;this.A={CFG,S,PRESET_WATCH,parseLeaderboard,info,getFills,mergedAvHistory,computeMetrics,simulateCopy,DAY};', ctx);
const A = ctx.A;
A.CFG.WEIGHT_PER_MIN = 1000;

const now = Date.now(), DAY = A.DAY;
let lbIndex = new Map();
try {
  const r = await fetch(A.CFG.LB_URL);
  const rows = A.parseLeaderboard(await r.json());
  lbIndex = new Map(rows.map(x => [x.addr, x]));
  console.log('leaderboard rows', rows.length);
} catch (e) { console.log('leaderboard error', e.message); }

const out = [], raw = {};
const entries = Object.entries(A.PRESET_WATCH);
for (const [i, [addr, name]] of entries.entries()) {
  try {
    const state = await A.info({ type: 'clearinghouseState', user: addr }, 2);
    const portfolio = await A.info({ type: 'portfolio', user: addr }, 20);
    const fills = await A.getFills(addr, now - 180 * DAY);
    const f90 = fills.filter(f => f.time >= now - 90 * DAY);
    const m = A.computeMetrics({ addr, name, fills: f90, portfolio, state, lb: lbIndex.get(addr), start: now - 90 * DAY, now });
    const avH = A.mergedAvHistory(portfolio);
    raw[addr] = portfolio;   // guardado para validar métricas sin re-consultar
    const base = { capital: 1000, target: 1000, feeRate: 0.0005, slip: 0.0005, minOrder: 5, maxLev: 10 };
    const sim = d => { const r = A.simulateCopy(fills, avH, { ...base, start: now - d * DAY });
      return { roi: r.roi, pnl: r.pnl, dd: r.dd, hitDays: r.hitTs ? (r.hitTs - (now - d * DAY)) / DAY : null, skippedPct: r.skippedPct, copied: r.copied, wiped: r.wiped }; };
    m.sim90 = sim(90); m.sim180 = sim(180);
    m.fills180 = fills.length; m.firstFill = fills.length ? Math.min(...fills.map(f => f.time)) : null;
    m.inLeaderboard = lbIndex.has(addr);
    out.push(m);
    console.log(`${i + 1}/${entries.length} ${name} ok trades=${m.trades} copy=${m.copy}`);
  } catch (e) {
    out.push({ addr, name, error: e.message });
    console.log(`${i + 1}/${entries.length} ${name} ERROR ${e.message}`);
  }
}

fs.mkdirSync('results', { recursive: true });
fs.writeFileSync('results/portfolios.json', JSON.stringify(raw));
fs.writeFileSync('results/analysis.json', JSON.stringify({ generated: new Date(now).toISOString(), wallets: out }));
const cols = ['name', 'addr', 'copy', 'aggr', 'av', 'roiW', 'roiM', 'roiA', 'pnlM', 'pnlA', 'wr', 'pf', 'payoff', 'trades', 'tpw', 'medHold', 'tim', 'avgLev', 'curLev',
  'ddAll', 'worstTradePct', 'worstWeekPct', 'maxLossStreak', 'longPct', 'entriesDay', 'maxEntriesDay', 'weeksPos', 'liqs', 'burned', 'flags', 'sim90', 'sim180', 'error'];
const cell = (m, c) => {
  let v = m[c];
  if (c === 'flags') v = (v || []).join('|');
  if (c === 'sim90' || c === 'sim180') v = v ? `${(v.roi * 100).toFixed(1)}%/${v.hitDays == null ? 'no' : v.hitDays.toFixed(0) + 'd'}/dd${(v.dd * 100).toFixed(0)}%/skip${(v.skippedPct * 100).toFixed(0)}%` : '';
  if (typeof v === 'number') v = Number.isFinite(v) ? +v.toFixed(4) : '';
  return v == null ? '' : `"${String(v).replace(/"/g, '""')}"`;
};
fs.writeFileSync('results/summary.csv', [cols.join(',')].concat(out.map(m => cols.map(c => cell(m, c)).join(','))).join('\n'));
console.log('done', out.length, 'errors', out.filter(x => x.error).length);
