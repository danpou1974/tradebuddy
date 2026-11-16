// Vuelca datos crudos de algunas billeteras para validar métricas
import fs from 'fs';
const API = 'https://api.hyperliquid.xyz/info';
const post = async b => (await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })).json();
const W = { n6: '0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00', n1: '0x939f95036d2e7b6d7419ec072bf9d967352204d2', n12: '0x5559da6ec434c5723d0ce9c4da7f29e3f8a3d43b',
  n34: '0xc0b2a1ce425d0cd8bc321c445fa6ae0eaed2e688', n42: '0xed7c45acfd7d2d3b21f3039d86e1da540596d23a', g1: '0x1c287f73c566b3382463a58e5e44f1336ed013c0', t1: '0x0bd27fac898a59680b9dc92bb7378df610825e8d' };
const out = {};
for (const [k, a] of Object.entries(W)) {
  const pf = await post({ type: 'portfolio', user: a });
  const st = await post({ type: 'clearinghouseState', user: a });
  const sp = await post({ type: 'spotClearinghouseState', user: a });
  const fills = await post({ type: 'userFillsByTime', user: a, startTime: Date.now() - 90 * 864e5, aggregateByTime: true });
  const liq = (fills || []).filter(f => f.liquidation);
  const summ = Object.fromEntries((pf || []).map(([p, v]) => [p, { nAv: v.accountValueHistory.length, av: [v.accountValueHistory[0], v.accountValueHistory.at(-1)], avMax: Math.max(...v.accountValueHistory.map(x => +x[1])), nPnl: v.pnlHistory.length,
    pnl: [v.pnlHistory[0], v.pnlHistory.at(-1)], pnlMax: Math.max(...v.pnlHistory.map(x => +x[1])), pnlMin: Math.min(...v.pnlHistory.map(x => +x[1])) }]));
  out[k] = { portfolio: summ, margin: st.marginSummary, nPos: st.assetPositions?.length, spot: (sp.balances || []).filter(b => +b.total > 0).slice(0, 5),
    nFills: fills.length, sampleFill: fills[0], nLiq: liq.length, liqSample: liq.slice(0, 2), liqSelf: liq.filter(f => f.liquidation.liquatedUser === a || f.liquidation.liquidatedUser === a).length,
    pnlAllTimeSample: (pf.find(x => x[0] === 'perpAllTime') || pf.find(x => x[0] === 'allTime'))[1].pnlHistory.filter((_, i, arr) => i % Math.ceil(arr.length / 25) === 0),
    avAllTimeSample: pf.find(x => x[0] === 'allTime')[1].accountValueHistory.filter((_, i, arr) => i % Math.ceil(arr.length / 25) === 0) };
}
fs.mkdirSync('results', { recursive: true });
fs.writeFileSync('results/debug.json', JSON.stringify(out, null, 1));
