// Vuelca fills, órdenes y velas de billeteras seleccionadas para estudiar su sistema
import fs from 'fs';
const API = 'https://api.hyperliquid.xyz/info';
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function post(b) {
  for (let i = 0; i < 6; i++) {
    const r = await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
    if (r.status === 429) { await sleep(3000 * (i + 1)); continue; }
    return r.json();
  }
  throw new Error('429');
}
const W = { n23: '0x441c8cbe1aa06beea4c84cfff5e12753f37822fb', n42: '0xed7c45acfd7d2d3b21f3039d86e1da540596d23a' };
const now = Date.now(), D = 864e5;
fs.mkdirSync('results/trades', { recursive: true });
for (const [name, a] of Object.entries(W)) {
  const fills = []; let from = now - 180 * D;
  for (let p = 0; p < 5; p++) {
    const r = await post({ type: 'userFillsByTime', user: a, startTime: from, aggregateByTime: true });
    if (!r.length) break; fills.push(...r); if (r.length < 2000) break; from = Math.max(...r.map(f => f.time)) + 1; await sleep(500);
  }
  const hist = await post({ type: 'historicalOrders', user: a });
  const open = await post({ type: 'frontendOpenOrders', user: a });
  const state = await post({ type: 'clearinghouseState', user: a });
  const portfolio = await post({ type: 'portfolio', user: a });
  const funding = await post({ type: 'userFunding', user: a, startTime: now - 180 * D });
  const cnt = {}; for (const f of fills) cnt[f.coin] = (cnt[f.coin] || 0) + 1;
  const coins = Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(x => x[0]).slice(0, 12);
  const candles = {};
  for (const c of coins) {
    candles[c] = {};
    for (const [iv, back] of [['1h', 185], ['1d', 450]]) {
      try { candles[c][iv] = await post({ type: 'candleSnapshot', req: { coin: c, interval: iv, startTime: now - back * D, endTime: now } }); } catch (e) { candles[c][iv] = []; }
      await sleep(300);
    }
  }
  fs.writeFileSync(`results/trades/${name}.json`, JSON.stringify({ addr: a, generated: new Date(now).toISOString(), fills, hist, open, state, portfolio, funding, candles }));
  console.log(name, 'fills', fills.length, 'orders', hist.length, 'open', open.length, 'coins', coins.join(','));
}
