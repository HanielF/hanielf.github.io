(function(){
  'use strict';
  const $=id=>document.getElementById(id);
  const core=window.RotationCoreV8;
  const state={conceptCatalog:null,barsByCode:{},benchmarkBars:[],benchmarkCode:'000300',run:null,activePeriod:-1,loadKey:'',failures:[],stale:false};
  const benchmarkNames={'000300':'沪深300','000001':'上证指数','000905':'中证500','399006':'创业板指','000688':'科创50'};
  const esc=value=>String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const pct=value=>(value>=0?'+':'')+(Number(value||0)*100).toFixed(2)+'%';
  const num=(value,digits=2)=>Number(value||0).toFixed(digits);
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const uniqueStocks=concepts=>[...new Map(concepts.flatMap(concept=>concept.stocks).map(stock=>[stock.code,{...stock,type:'stock'}])).values()];
  const nameMap=()=>new Map(state.conceptCatalog.concepts.flatMap(concept=>concept.stocks).map(stock=>[stock.code,stock.name]));

  function switchMode(mode){
    document.querySelectorAll('[data-mode-panel]').forEach(panel=>{panel.hidden=panel.dataset.modePanel!==mode});
    document.querySelectorAll('.modeTab').forEach(button=>{const active=button.dataset.mode===mode;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active))});
    window.scrollTo({top:0,behavior:'smooth'});
  }

  function setStatus(id,text,kind=''){
    const node=$(id);if(!node)return;node.className='v8Status '+kind;node.innerHTML=`<i></i><span>${esc(text)}</span>`;
  }

  async function initializeCatalog(){
    const response=await fetch(new URL('data/v8/concepts.json',document.baseURI),{cache:'force-cache'});if(!response.ok)throw new Error('概念研究底稿加载失败');
    state.conceptCatalog=await response.json();
    const dataStatus=window.MarketDataStoreV8?await window.MarketDataStoreV8.status():{dataVersion:'runtime-only',staticSymbols:0,lastTradingDate:null,source:'东方财富即时日线'};
    $('dataStatusBar').innerHTML=`
      <div><span>数据版本</span><b>${esc(dataStatus.dataVersion)}</b></div>
      <div><span>静态覆盖</span><b>${dataStatus.staticSymbols} 个标的 · ${dataStatus.lastTradingDate||'等待首轮固化'}</b></div>
      <div><span>历史口径</span><b class="gold">C级近似重建 · ${state.conceptCatalog.concepts.length}个概念</b></div>
      <div><span>增量与正式性</span><b>本机持久缓存 · 禁止演示数据</b></div>`;
  }

  function config(){
    return{
      start:$('rotationStart').value,end:$('rotationEnd').value,
      conceptCount:+$('conceptsPerPeriod').value||3,leaderCount:+$('leadersPerConcept').value||3,pureCount:+$('purePerConcept').value||3,
      portfolioSize:+$('portfolioSize').value||9,weighting:$('weighting').value,stockCap:+$('stockCap').value||20,
      bufferRank:+$('bufferRank').value||10,turnoverCap:+$('turnoverCap').value||100,
      riskOverlay:$('riskOverlay').value!=='none',riskMode:$('riskOverlay').value==='drawdown'?'drawdown':'trend',
      maxPortfolioDD:+$('maxPortfolioDD').value||30,benchmarkCode:$('mainBenchmark').value,capital:1000000,feeBp:3,slipBp:5,maxPerConcept:3
    };
  }

  async function loadResearchData(options){
    const stocks=uniqueStocks(state.conceptCatalog.concepts),key=[options.start,options.end,options.benchmarkCode].join('|');
    if(state.loadKey===key&&state.benchmarkBars.length&&Object.keys(state.barsByCode).length>20)return;
    if(typeof window.getMarketDataV8!=='function')throw new Error('行情读取模块尚未就绪，请刷新页面');
    state.barsByCode={};state.failures=[];state.benchmarkCode=options.benchmarkCode;
    setStatus('rotationStatus',`正在读取主基准 ${benchmarkNames[options.benchmarkCode]}…`,'busy');
    const benchmark=await window.getMarketDataV8({code:options.benchmarkCode,name:benchmarkNames[options.benchmarkCode],type:'index'},options.start,options.end);
    state.benchmarkBars=benchmark.bars;state.barsByCode[options.benchmarkCode]=benchmark.bars;
    const chunkSize=5;
    for(let offset=0;offset<stocks.length;offset+=chunkSize){
      const chunk=stocks.slice(offset,offset+chunkSize);
      const loaded=await Promise.all(chunk.map(async stock=>{try{return{stock,data:await window.getMarketDataV8(stock,options.start,options.end)}}catch(error){return{stock,error}}}));
      loaded.forEach(item=>{if(item.error)state.failures.push({code:item.stock.code,name:item.stock.name,error:String(item.error.message||item.error)});else state.barsByCode[item.stock.code]=item.data.bars});
      const completed=Math.min(stocks.length,offset+chunk.length);
      setStatus('rotationStatus',`读取真实行情 ${completed}/${stocks.length} · 成功 ${Object.keys(state.barsByCode).length-1} · 失败 ${state.failures.length}`,'busy');
      await wait(45);
    }
    if(Object.keys(state.barsByCode).length<25)throw new Error(`完整行情仅 ${Object.keys(state.barsByCode).length-1} 只，低于正式组合最低要求`);
    const marketStatus=window.MarketDataStoreV8?await window.MarketDataStoreV8.status():{dataVersion:'runtime-only'};
    state.dataVersion=marketStatus.dataVersion;state.loadKey=key;
  }

  async function executeResearch(openPortfolio=false){
    const options=config(),button=openPortfolio?$('runPortfolio'):$('buildRotation');
    if(!options.start||!options.end||options.start>options.end){setStatus(openPortfolio?'portfolioStatus':'rotationStatus','日期范围无效，请检查开始与结束时期。','error');return}
    button.disabled=true;button.classList.add('busy');button.lastElementChild.textContent='⟳';
    setStatus(openPortfolio?'portfolioStatus':'rotationStatus','准备真实行情与月度时点数据…','busy');
    try{
      await loadResearchData(options);
      setStatus(openPortfolio?'portfolioStatus':'rotationStatus','训练期配置搜索、共享现金组合与封存测试运行中…','busy');
      await wait(30);
      state.run=core.runResearch({concepts:state.conceptCatalog.concepts,barsByCode:state.barsByCode,benchmarkBars:state.benchmarkBars,benchmarkCode:options.benchmarkCode,dataVersion:state.dataVersion,options});
      state.activePeriod=state.run.periods.length-1;state.stale=false;window.rotationSnapshot=state.run;
      renderAll();
      const valid=state.run.periods.filter(period=>period.concepts.length).length;
      setStatus('rotationStatus',`完成 ${valid}/${state.run.periods.length} 个时期 · ${state.run.universeSize} 只研究股票 · ${state.failures.length} 项行情失败`,'done');
      setStatus('portfolioStatus',`运行 ${state.run.runId} 已冻结 · 数据 ${state.run.dataVersion} · 样本外结果仅验收不重排`,'done');
      try{localStorage.setItem('a-share-v8-last-run',JSON.stringify(state.run))}catch{/* 结果仍保留在当前标签页 */}
      if(openPortfolio)switchMode('portfolio');
    }catch(error){setStatus(openPortfolio?'portfolioStatus':'rotationStatus',`正式回测停止：${String(error.message||error)}`,'error')}
    finally{button.disabled=false;button.classList.remove('busy');button.lastElementChild.textContent='↗'}
  }

  function renderAll(){renderPeriods();renderPeriod(state.activePeriod);renderPortfolio();renderReport()}

  function renderPeriods(){
    $('periodRail').innerHTML=state.run.periods.map((period,index)=>`<button type="button" class="${index===state.activePeriod?'active':''}" data-period="${index}" role="listitem"><span>${esc(period.id)}</span><b>${period.concepts.map(row=>row.name).join(' · ')||'数据不足'}</b><small>${period.quality==='historical-approximation'?'近似重建':'覆盖不足'} · 信号 ${period.signalDate.slice(5)}</small></button>`).join('');
    $('periodRail').querySelectorAll('button').forEach(button=>button.onclick=()=>{state.activePeriod=+button.dataset.period;renderPeriods();renderPeriod(state.activePeriod)});
    $('periodRail').lastElementChild?.scrollIntoView({inline:'end',block:'nearest'});
  }

  function renderPeriod(index){
    const period=state.run.periods[index];if(!period)return;
    $('conceptTop3').innerHTML=period.concepts.length?period.concepts.map((concept,conceptIndex)=>`<button type="button" class="conceptResult ${conceptIndex===0?'active':''}" data-concept="${conceptIndex}"><span class="conceptRank">TOP ${concept.rank}</span><b>${esc(concept.name)}</b><strong>${num(concept.score,1)}</strong><small>20日相对 ${pct(concept.factors.relative20)} · 60日 ${pct(concept.factors.relative60)}<br>趋势广度 ${(concept.factors.breadth*100).toFixed(0)}% · 成交升温 ${pct(concept.factors.amountHeat)}</small><i>${Math.round(concept.confidence*100)}% 底稿可信度 · 历史近似</i></button>`).join(''):'<div class="v8EmptyState"><b>本期数据不足</b><span>至少需要每个概念两只股票和65个历史交易日。</span></div>';
    $('conceptTop3').querySelectorAll('button').forEach(button=>button.onclick=()=>{document.querySelectorAll('#conceptTop3 button').forEach(node=>node.classList.toggle('active',node===button));renderEvidence(period.concepts[+button.dataset.concept])});
    renderEvidence(period.concepts[0]);
    const all=period.concepts.flatMap(concept=>concept.picks.map(stock=>({...stock,concept:concept.name}))),deduped=[...new Map(all.map(stock=>[stock.code,stock])).values()],previous=state.run.periods[index-1],previousCodes=new Set(previous?previous.concepts.flatMap(concept=>concept.picks.map(stock=>stock.code)):[]),added=deduped.filter(stock=>!previousCodes.has(stock.code));
    $('universeDigest').innerHTML=`<div class="digestStats"><article><span>原始槽位</span><b>${period.concepts.length*(+$('leadersPerConcept').value+ +$('purePerConcept').value)}</b><small>龙头与正宗角色可重合</small></article><article><span>本期去重</span><b>${deduped.length}</b><small>${deduped.map(row=>row.name).join('、')}</small></article><article><span>相对上期新增</span><b>${added.length}</b><small>${added.map(row=>row.name).join('、')||'无'}</small></article><article><span>执行口径</span><b>次日开盘</b><small>${period.signalDate} 收盘生成 → ${period.tradeDate}</small></article></div><p class="qualityNotice">本期为 C 级历史重建：行情特征严格截止信号日，但概念成分与业务相关度来自免费研究底稿，存在幸存者偏差；不可作为已证明的无偏历史结论。</p>`;
  }

  function renderEvidence(concept){
    if(!concept){$('pickEvidence').innerHTML='<div class="v8EmptyState"><b>无可用概念</b></div>';return}
    const rows=(stocks,role)=>stocks.map((stock,index)=>`<li><span>${index+1}</span><div><b>${esc(stock.name)} <i>${stock.code}</i></b><small>${role==='leader'?`相对强度评分 ${num(stock.leaderScore,1)}`:`业务相关度 ${num(stock.purity,0)} / 100`} · 截止 ${stock.asOf}</small><em>${esc(role==='leader'?'行情龙头：动量、趋势与流动性':'业务相关度近似：'+stock.reason)}</em></div></li>`).join('');
    $('pickEvidence').innerHTML=`<article class="evidenceList"><header><span>龙头 Top${concept.leaders.length}</span><b>${esc(concept.name)}</b></header><ol>${rows(concept.leaders,'leader')}</ol></article><article class="evidenceList"><header><span>正宗 Top${concept.pure.length}</span><b>业务证据近似</b></header><ol>${rows(concept.pure,'pure')}</ol></article>`;
  }

  function metricCard(label,value,detail,className=''){return`<article><span>${esc(label)}</span><b class="${className}">${esc(value)}</b><small>${esc(detail)}</small></article>`}
  function renderPortfolio(){
    const run=state.run,sealed=run.sealed,strategy=sealed.strategy,benchmark=sealed.benchmark,excess=strategy.return-benchmark.return,ddImprove=Math.abs(benchmark.drawdown)-Math.abs(strategy.drawdown),gates=run.validation.gates;
    $('portfolioSummary').innerHTML=[
      metricCard('封存样本外收益',pct(strategy.return),`${run.validation.sealedTest[0]}—${run.validation.sealedTest[1]}`,strategy.return>=0?'up':'down'),
      metricCard('同期'+benchmarkNames[run.options.benchmarkCode],pct(benchmark.return),'主比较基准',benchmark.return>=0?'gold':'down'),
      metricCard('样本外超额',pct(excess),'策略减主基准',excess>=0?'up':'down'),
      metricCard('样本外最大回撤',pct(strategy.drawdown),`基准 ${pct(benchmark.drawdown)} · 改善 ${pct(ddImprove)}`,Math.abs(strategy.drawdown)<=run.options.maxPortfolioDD/100?'up':'down'),
      metricCard('样本外 Calmar',num(strategy.calmar),`基准 ${num(benchmark.calmar)}`,strategy.calmar>=benchmark.calmar?'up':'gold'),
      metricCard('Walk-forward',pct(run.validation.outer.passRate),`${run.validation.outer.folds.filter(row=>row.passed).length}/${run.validation.outer.folds.length} 个外层测试跑赢`,run.validation.outer.passRate>=.6?'up':'gold'),
      metricCard('全周期诊断收益',pct(run.selected.stats.return),'包含训练与测试，不是独立证据',run.selected.stats.return>=0?'up':'down'),
      metricCard('换手 / 成交',num(run.selected.turnover,1)+'×',`${run.selected.trades.length}笔 · 阻断${run.selected.blocked.length}次`)
    ].join('');
    const policy=document.querySelector('.portfolioResults .v8Policy');policy.className='v8Policy '+(Object.values(gates).every(Boolean)?'':'mutedPolicy');policy.textContent=Object.values(gates).every(Boolean)?'封存门槛通过':'封存门槛未全部通过';
    drawPortfolioChart();drawDrawdown();renderTimeline();renderAssetResults();
  }

  function alignedSeries(){
    const run=state.run,sets=[['策略',run.selected.equity,'#65e7ad'],[benchmarkNames[run.options.benchmarkCode],run.benchmark.equity,'#e5bd70'],['同批选股不风控',run.selectionOnly.equity,'#76a8ff'],['静态股票池持有',run.staticPool.equity,'#9a7bdc']];
    return sets.map(([name,rows,color])=>{const base=rows[0]?.nav||1;return{name,color,rows:rows.map(row=>({date:row.date,value:row.nav/base}))}}).filter(series=>series.rows.length>1);
  }
  function drawPortfolioChart(){
    const svg=$('portfolioChart'),series=alignedSeries(),all=series.flatMap(item=>item.rows.map(row=>row.value)),width=960,height=330,pad=38,min=Math.min(...all)*.97,max=Math.max(...all)*1.03,start=series[0].rows[0].date,end=series[0].rows.at(-1).date,span=Math.max(1,new Date(end)-new Date(start)),x=date=>pad+(new Date(date)-new Date(start))/span*(width-pad*2),y=value=>height-pad-(value-min)/Math.max(.001,max-min)*(height-pad*2),path=rows=>rows.map((row,index)=>(index?'L':'M')+x(row.date).toFixed(1)+','+y(row.value).toFixed(1)).join(' '),marks=state.run.selected.timeline.map(row=>`<circle cx="${x(row.tradeDate).toFixed(1)}" cy="${y(series[0].rows.find(point=>point.date>=row.tradeDate)?.value||1).toFixed(1)}" r="2.2" fill="#65e7ad"><title>${row.period} 换仓 · ${row.holdings.map(item=>item.name).join('、')}</title></circle>`).join('');
    svg.innerHTML=`${[.2,.4,.6,.8].map(q=>`<line x1="${pad}" x2="${width-pad}" y1="${height*q}" y2="${height*q}" stroke="rgba(213,238,226,.08)"/>`).join('')}${series.map(item=>`<path d="${path(item.rows)}" fill="none" stroke="${item.color}" stroke-width="${item.name==='策略'?2.6:1.4}" ${item.name==='策略'?'':'stroke-dasharray="5 5"'}/>`).join('')}${marks}<text x="${pad}" y="${height-8}" fill="#60756a">${start}</text><text x="${width-pad}" y="${height-8}" text-anchor="end" fill="#60756a">${end}</text>${series.map((item,index)=>`<g transform="translate(${pad+index*180},18)"><line x1="0" x2="18" stroke="${item.color}" stroke-width="2"/><text x="24" y="4" fill="#81968b">${item.name}</text></g>`).join('')}`;
  }
  function drawDrawdown(){
    const svg=$('drawdownChart'),rows=state.run.selected.equity,width=960,height=150,pad=32,start=rows[0].date,end=rows.at(-1).date,span=Math.max(1,new Date(end)-new Date(start)),min=Math.min(-.01,...rows.map(row=>row.drawdown)),x=date=>pad+(new Date(date)-new Date(start))/span*(width-pad*2),y=value=>pad+(0-value)/Math.max(.01,-min)*(height-pad*2),path=rows.map((row,index)=>(index?'L':'M')+x(row.date).toFixed(1)+','+y(row.drawdown).toFixed(1)).join(' '),limit=-(state.run.options.maxPortfolioDD||30)/100;
    svg.innerHTML=`<line x1="${pad}" x2="${width-pad}" y1="${y(0)}" y2="${y(0)}" stroke="rgba(213,238,226,.12)"/><line x1="${pad}" x2="${width-pad}" y1="${y(Math.max(limit,min))}" y2="${y(Math.max(limit,min))}" stroke="#ff8278" stroke-dasharray="6 5"/><path d="${path}" fill="none" stroke="#e5bd70" stroke-width="2"/><text x="${width-pad}" y="${Math.max(14,y(Math.max(limit,min))-5)}" text-anchor="end" fill="#ff8278">目标 ${pct(limit)}</text>`;
  }

  function renderTimeline(){
    const rows=state.run.selected.timeline;
    $('holdingsTimeline').innerHTML=`<table><thead><tr><th>时期 / 信号</th><th>热门概念</th><th>目标持仓与权重</th><th>新增 / 移除</th><th>总风险仓位</th></tr></thead><tbody>${rows.map((row,index)=>`<tr data-period="${index}"><td><b>${row.period}</b><br><small>${row.signalDate} → ${row.tradeDate}</small></td><td>${row.concepts.join('、')}</td><td>${row.holdings.map(item=>`${item.name} ${(item.weight*100).toFixed(1)}%`).join(' · ')}</td><td><span class="up">+${row.added.map(code=>nameMap().get(code)||code).join('、')||'无'}</span><br><span class="down">−${row.removed.map(code=>nameMap().get(code)||code).join('、')||'无'}</span></td><td>${(row.gross*100).toFixed(0)}%</td></tr>`).join('')}</tbody></table>`;
    $('holdingsTimeline').querySelectorAll('tbody tr').forEach(row=>row.onclick=()=>{const periodId=rows[+row.dataset.period].period,index=state.run.periods.findIndex(period=>period.id===periodId);if(index>=0){state.activePeriod=index;switchMode('rotation');renderPeriods();renderPeriod(index)}});
  }

  function assetRows(){
    const names=nameMap(),run=state.run;
    return run.selected.assetStats.map(row=>({...row,name:names.get(row.code)||row.code})).sort((a,b)=>b.contribution-a.contribution);
  }
  function renderAssetResults(){
    const rows=assetRows();
    $('assetResults').innerHTML=`<table><thead><tr><th>股票</th><th>入选期数</th><th>实际持有日</th><th>买入金额</th><th>卖出+期末市值</th><th>净贡献</th><th>组合贡献率</th><th>状态</th></tr></thead><tbody>${rows.map(row=>`<tr><td><b>${esc(row.name)}</b><br><small>${row.code}</small></td><td>${row.periods}</td><td>${row.days}</td><td>${Math.round(row.buyValue).toLocaleString()}</td><td>${Math.round(row.sellValue+row.endValue).toLocaleString()}</td><td class="${row.contribution>=0?'up':'down'}">${Math.round(row.contribution).toLocaleString()}</td><td class="${row.contributionRate>=0?'up':'down'}">${pct(row.contributionRate)}</td><td>${row.endValue>0?'期末持有':'已退出'}</td></tr>`).join('')}</tbody></table>`;
  }

  function gate(label,value){return`<li class="${value?'pass':'fail'}"><b>${value?'通过':'未通过'}</b><span>${esc(label)}</span></li>`}
  function renderReport(){
    const run=state.run,g=run.validation.gates,s=run.sealed.strategy,b=run.sealed.benchmark,allPassed=Object.values(g).every(Boolean),periodCount=run.periods.length;
    $('reportBody').innerHTML=`
      <article class="reportLead"><span>${allPassed?'封存量化门槛全部通过':'封存量化门槛未全部通过'} · 但历史成分仅为C级近似</span><h2>${allPassed?'本次组合在封存样本满足收益与回撤目标':'本次组合尚不能宣称稳定跑赢持有基准'}</h2><p>样本外收益 ${pct(s.return)}，${benchmarkNames[run.options.benchmarkCode]} ${pct(b.return)}，超额 ${pct(s.return-b.return)}；最大回撤 ${pct(s.drawdown)}。历史概念成分为免费档案重建，因此结果只能作为策略探索，不能冒充无偏证明。</p></article>
      <section class="reportChapter"><span>01 / 目标与验收门槛</span><h3>预注册的封存测试规则</h3><ul class="gateList">${gate('样本外净收益 > 0',g.positive)}${gate('样本外收益高于主基准',g.beatsBenchmark)}${gate(`样本外最大回撤 ≤ ${run.options.maxPortfolioDD}%`,g.drawdownCap)}${gate('回撤小于主基准',g.drawdownImproved)}${gate('外层滚动窗口至少60%跑赢',g.walkForward)}</ul></section>
      <section class="reportChapter"><span>02 / 数据清单</span><h3>${esc(run.dataVersion)}</h3><p>行情：东方财富公开日线，静态清单与浏览器持久缓存优先，缺失区间才即时补齐。股票信号使用前复权日线；成交按开盘价、3bp费率、5bp滑点、100股整手近似。研究期 ${run.options.start}—${run.options.end}，${periodCount} 个时期，${run.universeSize} 只去重股票，读取失败 ${state.failures.length} 只。</p></section>
      <section class="reportChapter"><span>03 / 历史选股方法</span><h3>概念 Top${run.options.conceptCount} → 龙头 Top${run.options.leaderCount} + 正宗 Top${run.options.pureCount}</h3><p>概念热度使用信号日前20/60日相对强度、成交升温、站上MA20广度与波动惩罚；龙头使用相对强度、趋势和流动性；“正宗”使用静态业务相关度底稿。所有行情特征严格 ≤ signalDate，tradeDate 为下一月首个交易日。</p></section>
      <section class="reportChapter"><span>04 / 组合与成交模型</span><h3>${run.options.portfolioSize}只 · ${run.options.weighting} · 单股上限${run.options.stockCap}%</h3><p>共享现金、先卖后买、100股整手、最低5元佣金、卖出印花税分段近似；停牌/缺开盘价订单阻断。月度保留缓冲排名 ${run.options.bufferRank}，换手上限 ${run.options.turnoverCap}%，趋势与组合回撤共同决定总风险仓位。</p></section>
      <section class="reportChapter"><span>05 / 参数搜索与冻结边界</span><h3>开发期 ${run.validation.development.join('—')} · 封存期 ${run.validation.sealedTest.join('—')}</h3><p>候选配置只按开发期 CAGR、Calmar、相对回撤和30%回撤惩罚评分；封存期不参与配置选择、过滤或重排。另用36个月开发＋3个月测试滚动拼接外层 OOS，资金连续、不在窗口间重置。</p></section>
      <section class="reportChapter"><span>06 / 结果与多基准</span><div class="reportMetricGrid">${metricCard('封存策略',pct(s.return),`回撤 ${pct(s.drawdown)}`,s.return>=0?'up':'down')}${metricCard('主基准',pct(b.return),benchmarkNames[run.options.benchmarkCode],b.return>=0?'gold':'down')}${metricCard('封存超额',pct(s.return-b.return),`Calmar ${num(s.calmar)}`,s.return>b.return?'up':'down')}${metricCard('全周期诊断',pct(run.selected.stats.return),'非独立证据')}</div></section>
      <section class="reportChapter"><span>07 / 时期与持仓附录</span><h3>${periodCount} 个时期 · ${run.selected.timeline.length} 次月度构建</h3><p>${run.selected.timeline.slice(-12).map(row=>`${row.period}：${row.concepts.join('/')}; ${row.holdings.map(item=>item.name).join('、')}`).join('<br>')}</p></section>
      <section class="reportChapter"><span>08 / 逐股贡献与失败</span><h3>${assetRows().length} 只实际持有 · ${run.selected.blocked.length} 次成交阻断</h3><p>${assetRows().slice(0,12).map(row=>`${row.name} ${pct(row.contributionRate)}（${row.periods}期）`).join(' · ')}</p></section>
      <section class="reportChapter warningChapter"><span>09 / 偏差与限制</span><h3>当前历史结果不具备“已消除偏差”的资格</h3><p>2021年至今的概念成分使用当前研究底稿向历史重建，仍含幸存者偏差、概念覆盖偏差和业务相关度发布时间偏差；涨跌停、上市初期、退市、历史ST与复权成交价仍为近似。从本版本上线起保存每月点时快照，未来前向样本才可逐步升级为正式证据。</p></section>
      <section class="reportChapter"><span>10 / 复现信息</span><h3>${esc(run.runId)}</h3><p>strategyVersion=${core.VERSION}<br>configHash=${esc(run.configHash)}<br>dataVersion=${esc(run.dataVersion)}<br>quality=${run.quality}<br>generatedAt=${run.generatedAt}</p></section>`;
  }

  function csvEscape(value){const text=String(value??'');return/[",\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text}
  function download(name,type,content){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
  function csv(name,rows){download(name,'text/csv;charset=utf-8','\ufeff'+rows.map(row=>row.map(csvEscape).join(',')).join('\n'))}
  function requireRun(){if(state.run)return true;alert('请先生成时期并运行组合回测。');return false}
  function exportPicks(){if(!requireRun())return;const rows=[['时期','信号截止日','执行日','概念排名','概念','概念分','股票','代码','身份','龙头分','业务相关度','证据口径']];state.run.periods.forEach(period=>period.concepts.forEach(concept=>concept.picks.forEach(stock=>rows.push([period.id,period.signalDate,period.tradeDate,concept.rank,concept.name,concept.score,stock.name,stock.code,stock.roles.join('+'),stock.leaderScore,stock.purity,'历史近似重建']))));csv('v8-monthly-picks.csv',rows)}
  function exportHoldings(){if(!requireRun())return;const rows=[['时期','信号日','执行日','概念','股票','代码','权重','新增','移除','总仓位']];state.run.selected.timeline.forEach(period=>period.holdings.forEach(stock=>rows.push([period.period,period.signalDate,period.tradeDate,period.concepts.join('|'),stock.name,stock.code,stock.weight,period.added.includes(stock.code),period.removed.join('|'),period.gross])));rows.push([]);rows.push(['成交日','代码','动作','数量','价格','原因']);state.run.selected.trades.forEach(trade=>rows.push([trade.date,trade.code,trade.action,trade.shares,trade.price,trade.reason]));csv('v8-holdings-and-trades.csv',rows)}
  function exportAssets(){if(!requireRun())return;csv('v8-per-asset-results.csv',[['股票','代码','入选期数','持有日','买入金额','卖出金额','期末市值','净贡献','贡献率'],...assetRows().map(row=>[row.name,row.code,row.periods,row.days,row.buyValue,row.sellValue,row.endValue,row.contribution,row.contributionRate])])}

  function markStale(){if(!state.run)return;state.stale=true;setStatus('portfolioStatus','配置已变化：上一次运行已冻结，请重新运行后再使用新配置。','error')}

  async function init(){
    if(!core){setStatus('rotationStatus','策略核心加载失败，请刷新页面。','error');return}
    document.querySelectorAll('.modeTab').forEach(button=>button.onclick=()=>switchMode(button.dataset.mode));
    $('buildRotation').onclick=()=>executeResearch(false);$('runPortfolio').onclick=()=>executeResearch(true);
    ['rotationStart','rotationEnd','conceptsPerPeriod','leadersPerConcept','purePerConcept','portfolioSize','weighting','stockCap','bufferRank','turnoverCap','riskOverlay','maxPortfolioDD','mainBenchmark'].forEach(id=>$(id)?.addEventListener('change',markStale));
    $('exportPicksCsv').onclick=exportPicks;$('exportHoldingsCsv').onclick=exportHoldings;$('exportAssetsCsv').onclick=exportAssets;$('exportFullJson').onclick=()=>{if(requireRun())download('v8-strategy-report.json','application/json;charset=utf-8',JSON.stringify(state.run,null,2))};$('printV8Report').onclick=()=>{if(requireRun()){switchMode('report');setTimeout(()=>window.print(),120)}};
    try{await initializeCatalog()}catch(error){setStatus('rotationStatus',String(error.message||error),'error')}
  }
  init();
})();
