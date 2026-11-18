// Datos actuales de HYPE + escalera de órdenes de n42 para calcular un Futures Grid
import fs from 'fs';
const API = 'https://api.hyperliquid.xyz/info', D = 864e5, now = Date.now();
const post = async b => (await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })).json();
const n42 = '0xed7c45acfd7d2d3b21f3039d86e1da540596d23a';
const h1 = await post({ type: 'candleSnapshot', req: { coin: 'HYPE', interval: '1h', startTime: now - 45 * D, endTime: now } });
const d1 = await post({ type: 'candleSnapshot', req: { coin: 'HYPE', interval: '1d', startTime: now - 200 * D, endTime: now } });
const mids = await post({ type: 'allMids' });
const open = await post({ type: 'frontendOpenOrders', user: n42 });
const st = await post({ type: 'clearinghouseState', user: n42 });
const fills = await post({ type: 'userFillsByTime', user: n42, startTime: now - 7 * D, aggregateByTime: true });
let pionex = null;
try { const j = await (await fetch('https://api.pionex.com/api/v1/common/symbols?symbols=HYPE_USDT_PERP')).json(); pionex = j; } catch (e) { pionex = { error: e.message }; }
let pxTicker = null;
try { pxTicker = await (await fetch('https://api.pionex.com/api/v1/market/tickers?symbol=HYPE_USDT_PERP')).json(); } catch (e) { pxTicker = { error: e.message }; }
fs.mkdirSync('results', { recursive: true });
fs.writeFileSync('results/grid_hype.json', JSON.stringify({ generated: new Date(now).toISOString(), mid: mids.HYPE, h1, d1,
  n42: { open: open.filter(o => o.coin === 'HYPE').map(o => ({ side: o.side, px: +o.limitPx, sz: +o.sz })), pos: (st.assetPositions || []).map(a => a.position), av: st.marginSummary,
    fills7d: fills.map(f => ({ t: f.time, px: +f.px, sz: +f.sz, side: f.side, dir: f.dir })) }, pionex, pxTicker }));
console.log('HYPE mid', mids.HYPE, 'n42 open', open.length);
