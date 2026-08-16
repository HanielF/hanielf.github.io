(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.RotationCoreV8=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';

  const VERSION='rotation-core-v8.0.0';
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;
  const std=values=>{if(values.length<2)return 0;const average=mean(values);return Math.sqrt(values.reduce((sum,value)=>sum+(value-average)**2,0)/(values.length-1))};
  const monthId=date=>date.slice(0,7);
  const last=values=>values[values.length-1];
  const byDate=bars=>new Map((bars||[]).map(bar=>[bar.date,bar]));
  const safeReturn=(end,start)=>Number.isFinite(end)&&Number.isFinite(start)&&start>0?end/start-1:0;
  const stableHash=value=>{
    const text=typeof value==='string'?value:JSON.stringify(value,Object.keys(value||{}).sort());
    let hash=2166136261;
    for(let index=0;index<text.length;index++){hash^=text.charCodeAt(index);hash=Math.imul(hash,16777619)}
    return('00000000'+(hash>>>0).toString(16)).slice(-8);
  };

  function normalize(rows,key,outKey=key+'Z'){
    const values=rows.map(row=>Number(row[key])||0),average=mean(values),deviation=std(values)||1;
    rows.forEach(row=>{row[outKey]=clamp(((Number(row[key])||0)-average)/deviation,-3,3)});
    return rows;
  }

  function indexAtOrBefore(bars,date){
    let low=0,high=(bars||[]).length-1,answer=-1;
    while(low<=high){const middle=(low+high)>>1;if(bars[middle].date<=date){answer=middle;low=middle+1}else high=middle-1}
    return answer;
  }

  function valueAt(bars,index,key='close'){return index>=0&&bars[index]?Number(bars[index][key]):NaN}
  function momentum(bars,index,lookback){return index>=lookback?safeReturn(valueAt(bars,index),valueAt(bars,index-lookback)):0}
  function movingAverage(bars,index,period,key='close'){
    if(index<period-1)return NaN;
    let total=0;for(let offset=0;offset<period;offset++)total+=Number(bars[index-offset][key])||0;
    return total/period;
  }
  function volatility(bars,index,period=20){
    if(index<period)return .5;
    const returns=[];for(let i=index-period+1;i<=index;i++)returns.push(safeReturn(bars[i].close,bars[i-1].close));
    return std(returns)*Math.sqrt(252);
  }
  function liquidityAcceleration(bars,index){
    if(index<40)return 0;
    const recent=mean(bars.slice(index-19,index+1).map(bar=>Number(bar.amount)||Number(bar.volume)*Number(bar.close)||0));
    const prior=mean(bars.slice(index-39,index-19).map(bar=>Number(bar.amount)||Number(bar.volume)*Number(bar.close)||0));
    return prior?safeReturn(recent,prior):0;
  }

  function buildPeriods(calendar,start,end){
    const allDates=(calendar||[]).map(row=>typeof row==='string'?row:row.date).filter(date=>date<=end).sort();
    const dates=allDates.filter(date=>date>=start);
    const buckets=new Map();
    dates.forEach(date=>{const id=monthId(date);if(!buckets.has(id))buckets.set(id,[]);buckets.get(id).push(date)});
    const periods=[];
    for(const [id,monthDates] of buckets){
      const tradeDate=monthDates[0],calendarIndex=allDates.indexOf(tradeDate),signalDate=calendarIndex>0?allDates[calendarIndex-1]:null;
      if(!signalDate)continue;
      periods.push({id,signalDate,tradeDate,holdEnd:last(monthDates),quality:'historical-approximation'});
    }
    return periods;
  }

  function stockFeature(stock,bars,signalDate,benchmarkBars){
    const index=indexAtOrBefore(bars,signalDate),benchmarkIndex=indexAtOrBefore(benchmarkBars,signalDate);
    if(index<65||benchmarkIndex<65)return null;
    const m20=momentum(bars,index,20),m60=momentum(bars,index,60),bm20=momentum(benchmarkBars,benchmarkIndex,20),bm60=momentum(benchmarkBars,benchmarkIndex,60);
    const ma20=movingAverage(bars,index,20),ma60=movingAverage(bars,index,60),close=valueAt(bars,index),vol=volatility(bars,index,20),amount=mean(bars.slice(index-19,index+1).map(bar=>Number(bar.amount)||Number(bar.volume)*Number(bar.close)||0));
    return{
      code:stock.code,name:stock.name,purity:Number(stock.purity)||0,reason:stock.reason||'',
      rel20:m20-bm20,rel60:m60-bm60,m20,m60,above20:close>ma20?1:0,above60:close>ma60?1:0,
      amount,amountHeat:liquidityAcceleration(bars,index),volatility:vol,close,asOf:bars[index].date
    };
  }

  function rankPeriod(period,concepts,barsByCode,benchmarkBars,options={}){
    const conceptRows=[];
    for(const concept of concepts){
      if(concept.availableFrom&&concept.availableFrom>period.signalDate)continue;
      const stocks=concept.stocks.map(stock=>stockFeature(stock,barsByCode[stock.code]||[],period.signalDate,benchmarkBars)).filter(Boolean);
      if(stocks.length<2)continue;
      const breadth=mean(stocks.map(stock=>stock.above20)),rel20=mean(stocks.map(stock=>stock.rel20)),rel60=mean(stocks.map(stock=>stock.rel60)),amountHeat=mean(stocks.map(stock=>clamp(stock.amountHeat,-1,2))),vol=mean(stocks.map(stock=>stock.volatility));
      conceptRows.push({concept,stocks,rel20,rel60,breadth,amountHeat,vol,confidence:Number(concept.confidence)||.5});
    }
    ['rel20','rel60','breadth','amountHeat','vol'].forEach(key=>normalize(conceptRows,key));
    conceptRows.forEach(row=>{row.score=40*row.rel20Z+25*row.rel60Z+20*row.breadthZ+10*row.amountHeatZ-5*row.volZ+3*(row.confidence-.5)});
    conceptRows.sort((a,b)=>b.score-a.score);
    const top=conceptRows.slice(0,options.conceptCount||3).map((row,conceptRank)=>{
      const candidates=row.stocks.map(stock=>({...stock}));
      ['rel20','rel60','amount','above20'].forEach(key=>normalize(candidates,key));
      candidates.forEach(stock=>{
        stock.leaderScore=45*stock.rel20Z+25*stock.rel60Z+15*stock.amountZ+15*stock.above20Z;
        stock.purityScore=.7*stock.purity+.2*clamp(50+stock.leaderScore,0,100)+.1*clamp(50+stock.amountZ*15,0,100);
      });
      const leaders=[...candidates].sort((a,b)=>b.leaderScore-a.leaderScore).slice(0,options.leaderCount||3);
      const pure=[...candidates].sort((a,b)=>b.purityScore-a.purityScore).slice(0,options.pureCount||3);
      const roleMap=new Map();
      leaders.forEach((stock,index)=>roleMap.set(stock.code,{...stock,roles:['leader'],leaderRank:index+1}));
      pure.forEach((stock,index)=>{const current=roleMap.get(stock.code);if(current){current.roles.push('authentic');current.pureRank=index+1}else roleMap.set(stock.code,{...stock,roles:['authentic'],pureRank:index+1})});
      return{
        id:row.concept.id,name:row.concept.name,rank:conceptRank+1,score:row.score,
        factors:{relative20:row.rel20,relative60:row.rel60,breadth:row.breadth,amountHeat:row.amountHeat,volatility:row.vol},
        confidence:row.confidence,leaders,pure,picks:[...roleMap.values()],quality:'historical-approximation'
      };
    });
    return{...period,concepts:top,quality:top.length?'historical-approximation':'insufficient-data'};
  }

  function selectHoldings(period,previousCodes=[],options={}){
    const all=[];
    for(const concept of period.concepts){
      for(const stock of concept.picks){
        const roleBoost=stock.roles.includes('leader')&&stock.roles.includes('authentic')?18:stock.roles.includes('leader')?12:7;
        all.push({...stock,conceptId:concept.id,conceptName:concept.name,conceptRank:concept.rank,selectionScore:concept.score+stock.leaderScore*.55+stock.purityScore*.18+roleBoost});
      }
    }
    const seen=new Set(),deduped=all.sort((a,b)=>b.selectionScore-a.selectionScore).filter(stock=>{if(seen.has(stock.code))return false;seen.add(stock.code);return true});
    const prior=new Set(previousCodes),bufferRank=options.bufferRank||10,portfolioSize=options.portfolioSize||9,maxPerConcept=options.maxPerConcept||3;
    const kept=deduped.filter((stock,index)=>prior.has(stock.code)&&index<bufferRank),result=[],counts=new Map();
    const tryAdd=stock=>{if(result.some(row=>row.code===stock.code))return;const count=counts.get(stock.conceptId)||0;if(count>=maxPerConcept)return;result.push(stock);counts.set(stock.conceptId,count+1)};
    kept.forEach(tryAdd);deduped.forEach(stock=>{if(result.length<portfolioSize)tryAdd(stock)});
    const maxNew=Math.max(1,Math.ceil(portfolioSize*clamp((options.turnoverCap??100)/100,.1,2))),newRows=result.filter(row=>!prior.has(row.code));
    if(previousCodes.length&&newRows.length>maxNew){
      const keepNew=new Set(newRows.slice(0,maxNew).map(row=>row.code)),reduced=result.filter(row=>prior.has(row.code)||keepNew.has(row.code));
      deduped.filter(row=>prior.has(row.code)).forEach(row=>{if(reduced.length<portfolioSize&&!reduced.some(item=>item.code===row.code))reduced.push(row)});
      result.splice(0,result.length,...reduced.slice(0,portfolioSize));
    }
    return{holdings:result.slice(0,portfolioSize),ranked:deduped,added:result.filter(row=>!prior.has(row.code)).map(row=>row.code),removed:previousCodes.filter(code=>!result.some(row=>row.code===code))};
  }

  function inverseVolWeights(holdings,barsByCode,signalDate,stockCap=.18){
    if(!holdings.length)return{};
    const raw=holdings.map(stock=>{const bars=barsByCode[stock.code]||[],index=indexAtOrBefore(bars,signalDate),vol=Math.max(.12,volatility(bars,index,20));return{code:stock.code,value:1/vol}});
    let weights=Object.fromEntries(raw.map(row=>[row.code,row.value/raw.reduce((sum,item)=>sum+item.value,0)]));
    for(let loop=0;loop<8;loop++){
      const capped=Object.entries(weights).filter(([,weight])=>weight>stockCap),excess=capped.reduce((sum,[,weight])=>sum+weight-stockCap,0);
      if(!capped.length)break;capped.forEach(([code])=>{weights[code]=stockCap});
      const open=Object.entries(weights).filter(([,weight])=>weight<stockCap-.000001),base=open.reduce((sum,[,weight])=>sum+weight,0);
      if(!base)break;open.forEach(([code,weight])=>{weights[code]+=excess*weight/base});
    }
    const total=Object.values(weights).reduce((sum,value)=>sum+value,0)||1;
    Object.keys(weights).forEach(code=>{weights[code]/=total});return weights;
  }

  function equalWeights(holdings){return holdings.length?Object.fromEntries(holdings.map(stock=>[stock.code,1/holdings.length])):{} }
  function scoreWeights(holdings,stockCap=.18){
    if(!holdings.length)return{};const floor=Math.min(...holdings.map(row=>row.selectionScore)),raw=holdings.map(row=>({code:row.code,value:Math.max(1,row.selectionScore-floor+5)})),total=raw.reduce((sum,row)=>sum+row.value,0);const weights=Object.fromEntries(raw.map(row=>[row.code,Math.min(stockCap,row.value/total)])),used=Object.values(weights).reduce((sum,value)=>sum+value,0)||1;Object.keys(weights).forEach(code=>{weights[code]/=used});return weights;
  }

  function portfolioStats(equity,start,end,key='nav'){
    const rows=(equity||[]).filter(row=>(!start||row.date>=start)&&(!end||row.date<=end));
    if(rows.length<2)return{return:0,cagr:0,drawdown:0,sharpe:0,calmar:0,volatility:0};
    const base=rows[0][key]||1,normalized=rows.map(row=>({...row,_value:(row[key]||0)/base}));
    let peak=normalized[0]._value,maxDrawdown=0;const returns=[];
    normalized.forEach((row,index)=>{peak=Math.max(peak,row._value);maxDrawdown=Math.min(maxDrawdown,row._value/peak-1);if(index)returns.push(safeReturn(row._value,normalized[index-1]._value))});
    const total=last(normalized)._value-1,days=Math.max(1,(new Date(last(rows).date)-new Date(rows[0].date))/864e5),years=Math.max(days/365.25,1/252),cagr=Math.pow(Math.max(.0001,1+total),1/years)-1,vol=std(returns)*Math.sqrt(252),sharpe=vol?mean(returns)*252/vol:0,calmar=maxDrawdown<0?cagr/Math.abs(maxDrawdown):cagr>0?9.99:0;
    return{return:total,cagr,drawdown:maxDrawdown,sharpe,calmar,volatility:vol};
  }

  function trendGross(benchmarkBars,date,previousNav,peakNav,maxDrawdown=.3,mode='trend'){
    const index=indexAtOrBefore(benchmarkBars,date);if(index<125)return .75;
    const close=benchmarkBars[index].close,ma20=movingAverage(benchmarkBars,index,20),ma120=movingAverage(benchmarkBars,index,120);
    let gross=1;
    if(mode!=='drawdown'){
      if(close>ma20&&close>ma120)gross=1;
      else if(close>ma20||close>ma120)gross=.75;
      else gross=.45;
    }
    const drawdown=peakNav?safeReturn(previousNav,peakNav):0;
    const scale=clamp(maxDrawdown/.3,.5,1.3);
    if(drawdown<=-.18*scale)gross=Math.min(gross,.35);else if(drawdown<=-.12*scale)gross=Math.min(gross,.6);else if(drawdown<=-.08*scale)gross=Math.min(gross,.8);
    return gross;
  }

  function simulatePortfolio(periods,barsByCode,benchmarkBars,options={}){
    const calendar=benchmarkBars.filter(bar=>bar.date>=options.start&&bar.date<=options.end),maps=Object.fromEntries(Object.entries(barsByCode).map(([code,bars])=>[code,byDate(bars)]));
    const periodMap=new Map(periods.map(period=>[period.tradeDate,period])),capital=options.capital||1000000,fee=(options.feeBp??3)/10000,slip=(options.slipBp??5)/10000,stockCap=clamp((options.stockCap||18)/100,.05,.3);
    let cash=capital,positions={},lastPrices={},peakNav=capital,previousNav=capital,currentSelection=[],currentBaseWeights={},currentGross=0,previousCodes=[];
    const equity=[],trades=[],timeline=[],assetStats={},blocked=[];
    const mark=(date,field='close')=>cash+Object.entries(positions).reduce((sum,[code,shares])=>{const bar=maps[code]?.get(date),price=bar?.[field]||bar?.close||lastPrices[code]||0;if(bar?.close)lastPrices[code]=bar.close;return sum+shares*price},0);
    const commission=value=>Math.max(5,value*fee);
    function targetWeights(period){
      const selected=selectHoldings(period,previousCodes,options);previousCodes=selected.holdings.map(row=>row.code);currentSelection=selected.holdings;
      const raw=options.weighting==='inverseVol'?inverseVolWeights(currentSelection,barsByCode,period.signalDate,stockCap):options.weighting==='score'?scoreWeights(currentSelection,stockCap):equalWeights(currentSelection);
      currentBaseWeights=raw;return selected;
    }
    function rebalance(date,period,reason,forceGross){
      const selection=period?targetWeights(period):{holdings:currentSelection,added:[],removed:[]};
      const gross=options.riskOverlay===false?1:forceGross;
      const openValue=mark(date,'open'),targets=Object.fromEntries(Object.entries(currentBaseWeights).map(([code,weight])=>[code,weight*gross]));
      const codes=new Set([...Object.keys(positions),...Object.keys(targets)]);
      const desired={};
      for(const code of codes){const bar=maps[code]?.get(date);if(!bar||!bar.open||bar.volume===0){blocked.push({date,code,reason:'停牌或缺少开盘价'});continue}desired[code]=Math.floor(openValue*(targets[code]||0)/(bar.open*(1+slip))/100)*100}
      for(const code of codes){const held=positions[code]||0,want=desired[code]??held;if(want>=held)continue;const bar=maps[code]?.get(date);if(!bar)continue;const qty=held-want,price=bar.open*(1-slip),value=qty*price,stamp=date<'2023-08-28'?.001:.0005;cash+=value-commission(value)-value*stamp;positions[code]=want;if(!want)delete positions[code];trades.push({date,code,action:'sell',shares:qty,price,reason});const stats=assetStats[code]||(assetStats[code]={code,buyValue:0,sellValue:0,cost:0,periods:0,days:0});stats.sellValue+=value;stats.cost+=commission(value)+value*stamp}
      for(const code of codes){const held=positions[code]||0,want=desired[code]??held;if(want<=held)continue;const bar=maps[code]?.get(date);if(!bar)continue;let qty=want-held,price=bar.open*(1+slip),value=qty*price,cost=value+commission(value);if(cost>cash){qty=Math.floor((cash-5)/price/100)*100;value=qty*price;cost=value+commission(value)}if(qty<=0)continue;cash-=cost;positions[code]=held+qty;trades.push({date,code,action:'buy',shares:qty,price,reason});const stats=assetStats[code]||(assetStats[code]={code,buyValue:0,sellValue:0,cost:0,periods:0,days:0});stats.buyValue+=value;stats.cost+=commission(value)}
      currentGross=gross;
      if(period){currentSelection.forEach(stock=>{const stats=assetStats[stock.code]||(assetStats[stock.code]={code:stock.code,buyValue:0,sellValue:0,cost:0,periods:0,days:0});stats.periods++});timeline.push({period:period.id,signalDate:period.signalDate,tradeDate:date,concepts:period.concepts.map(row=>row.name),holdings:currentSelection.map(stock=>({code:stock.code,name:stock.name,concept:stock.conceptName,roles:stock.roles,weight:(currentBaseWeights[stock.code]||0)*gross})),added:selection.added,removed:selection.removed,gross})}
    }
    for(let index=0;index<calendar.length;index++){
      const bar=calendar[index],date=bar.date,priorDate=index?calendar[index-1].date:date,period=periodMap.get(date),gross=options.riskOverlay===false?1:trendGross(benchmarkBars,priorDate,previousNav,peakNav,(options.maxPortfolioDD||30)/100,options.riskMode||'trend');
      if(period)rebalance(date,period,'月度换仓',gross);else if(currentSelection.length&&gross<currentGross-.19)rebalance(date,null,'组合风险降仓',gross);else if(currentSelection.length&&gross>currentGross+.24&&index%5===0)rebalance(date,null,'趋势恢复加仓',gross);
      Object.keys(positions).forEach(code=>{if(positions[code]>0){const stats=assetStats[code]||(assetStats[code]={code,buyValue:0,sellValue:0,cost:0,periods:0,days:0});stats.days++}});
      const nav=mark(date,'close');peakNav=Math.max(peakNav,nav);previousNav=nav;equity.push({date,nav,drawdown:nav/peakNav-1,cash,gross:nav?1-cash/nav:0});
    }
    Object.entries(assetStats).forEach(([code,row])=>{row.endValue=(positions[code]||0)*(lastPrices[code]||0);row.contribution=row.sellValue+row.endValue-row.buyValue-row.cost;row.contributionRate=row.contribution/capital});
    const stats=portfolioStats(equity);return{equity,trades,timeline,assetStats:Object.values(assetStats),blocked,stats,turnover:trades.reduce((sum,trade)=>sum+trade.shares*trade.price,0)/capital,config:{...options},endCash:cash};
  }

  function simulateBuyHold(codes,barsByCode,calendar,options={}){
    const start=options.start,end=options.end,dates=calendar.filter(bar=>bar.date>=start&&bar.date<=end),capital=options.capital||1000000,maps=Object.fromEntries(codes.map(code=>[code,byDate(barsByCode[code]||[])]));
    let cash=capital,positions={},entered=false,lastPrices={};const equity=[];
    for(const day of dates){
      if(!entered){const available=codes.filter(code=>maps[code].get(day.date)?.open>0),weight=available.length?1/available.length:0;for(const code of available){const bar=maps[code].get(day.date),qty=capital*weight/bar.open;positions[code]=qty;cash-=qty*bar.open}entered=available.length>0}
      const nav=cash+Object.entries(positions).reduce((sum,[code,qty])=>{const close=maps[code].get(day.date)?.close||lastPrices[code]||0;if(close)lastPrices[code]=close;return sum+qty*close},0);equity.push({date:day.date,nav})
    }
    return{equity,stats:portfolioStats(equity)};
  }

  function sliceReturnSeries(result,start,end){
    const rows=result.equity.filter(row=>row.date>=start&&row.date<=end);if(rows.length<2)return[];const values=[];for(let index=1;index<rows.length;index++)values.push({date:rows[index].date,ret:safeReturn(rows[index].nav,rows[index-1].nav)});return values;
  }
  function scoreStats(stats,benchmarkStats){return(stats.cagr-benchmarkStats.cagr)*180+stats.calmar*5+(Math.abs(benchmarkStats.drawdown)-Math.abs(stats.drawdown))*60-Math.max(0,Math.abs(stats.drawdown)-.3)*250}

  function stitchOuterOos(candidateRuns,benchmark,periods,options={}){
    const developmentMonths=options.developmentMonths||36,testMonths=options.testMonths||3,folds=[];let nav=options.capital||1000000;const equity=[];
    for(let testStartIndex=developmentMonths;testStartIndex<periods.length;testStartIndex+=testMonths){
      const developmentStart=periods[Math.max(0,testStartIndex-developmentMonths)].tradeDate,developmentEnd=periods[testStartIndex-1].holdEnd,testStart=periods[testStartIndex].tradeDate,testEnd=periods[Math.min(periods.length-1,testStartIndex+testMonths-1)].holdEnd;
      const benchmarkDevelopment=portfolioStats(benchmark.equity,developmentStart,developmentEnd);
      const ranked=candidateRuns.map(row=>({row,score:scoreStats(portfolioStats(row.result.equity,developmentStart,developmentEnd),benchmarkDevelopment)})).sort((a,b)=>b.score-a.score);
      const chosen=ranked[0].row,series=sliceReturnSeries(chosen.result,testStart,testEnd),startNav=nav;
      series.forEach(point=>{nav*=1+point.ret;equity.push({date:point.date,nav})});
      const testStats=portfolioStats(chosen.result.equity,testStart,testEnd),benchmarkTest=portfolioStats(benchmark.equity,testStart,testEnd);
      folds.push({development:[developmentStart,developmentEnd],test:[testStart,testEnd],configHash:chosen.hash,config:chosen.config,strategy:testStats,benchmark:benchmarkTest,excess:testStats.return-benchmarkTest.return,passed:testStats.return>benchmarkTest.return});
      if(!series.length)nav=startNav;
    }
    return{equity,folds,stats:portfolioStats(equity),passRate:folds.length?folds.filter(fold=>fold.passed).length/folds.length:0};
  }

  function runResearch(input){
    const options={conceptCount:3,leaderCount:3,pureCount:3,portfolioSize:9,bufferRank:10,maxPerConcept:3,weighting:'inverseVol',stockCap:18,riskOverlay:true,capital:1000000,feeBp:3,slipBp:5,...input.options};
    const benchmarkBars=input.benchmarkBars||[],periods=buildPeriods(benchmarkBars,options.start,options.end).map(period=>rankPeriod(period,input.concepts,input.barsByCode,benchmarkBars,options));
    const universe=[...new Set(input.concepts.flatMap(concept=>concept.stocks.map(stock=>stock.code)))];
    const candidates=input.candidates||[
      {portfolioSize:6,weighting:'inverseVol',bufferRank:9,riskOverlay:true},
      {portfolioSize:9,weighting:'inverseVol',bufferRank:10,riskOverlay:true},
      {portfolioSize:12,weighting:'inverseVol',bufferRank:14,riskOverlay:true},
      {portfolioSize:9,weighting:'equal',bufferRank:10,riskOverlay:true},
      {portfolioSize:9,weighting:'inverseVol',bufferRank:12,riskOverlay:false}
    ];
    const benchmark=simulateBuyHold([input.benchmarkCode||'000300'],{...input.barsByCode,[input.benchmarkCode||'000300']:benchmarkBars},benchmarkBars,options),staticPool=simulateBuyHold(universe,input.barsByCode,benchmarkBars,options);
    const candidateRuns=candidates.map(candidate=>{const config={...options,...candidate},result=simulatePortfolio(periods,input.barsByCode,benchmarkBars,config);return{config,hash:stableHash(config),result}});
    const cut=Math.max(1,Math.floor(periods.length*.7)),developmentEnd=periods[Math.min(cut-1,periods.length-1)]?.holdEnd||options.end,sealedStart=periods[Math.min(cut,periods.length-1)]?.tradeDate||options.end,benchmarkDevelopment=portfolioStats(benchmark.equity,options.start,developmentEnd);
    const rankedDevelopment=candidateRuns.map(row=>({...row,development:portfolioStats(row.result.equity,options.start,developmentEnd)})).map(row=>({...row,developmentScore:scoreStats(row.development,benchmarkDevelopment)})).sort((a,b)=>b.developmentScore-a.developmentScore);
    const selected=rankedDevelopment[0]||candidateRuns[0],selectionOnly=simulatePortfolio(periods,input.barsByCode,benchmarkBars,{...selected.config,riskOverlay:false}),sealed={strategy:portfolioStats(selected.result.equity,sealedStart,options.end),benchmark:portfolioStats(benchmark.equity,sealedStart,options.end),selectionOnly:portfolioStats(selectionOnly.equity,sealedStart,options.end),staticPool:portfolioStats(staticPool.equity,sealedStart,options.end)},outer=stitchOuterOos(candidateRuns,benchmark,periods,options);
    const dataVersion=input.dataVersion||'runtime-unpersisted',configHash=stableHash(selected.config),runId=`v8-${Date.now().toString(36)}-${configHash}`;
    const validation={
      development:[options.start,developmentEnd],sealedTest:[sealedStart,options.end],outer,
      gates:{positive:sealed.strategy.return>0,beatsBenchmark:sealed.strategy.return>sealed.benchmark.return,drawdownCap:Math.abs(sealed.strategy.drawdown)<=(options.maxPortfolioDD||30)/100,drawdownImproved:Math.abs(sealed.strategy.drawdown)<Math.abs(sealed.benchmark.drawdown),walkForward:outer.passRate>=.6},
      eligibleForHeadline:false,reason:'历史概念成分为免费档案重建（C级近似），仅作探索；从上线后的点时快照开始累计正式前向验证。'
    };
    return{version:VERSION,runId,dataVersion,configHash,generatedAt:new Date().toISOString(),quality:'historical-approximation',options:selected.config,periods,selected:selected.result,selectionOnly,benchmark,staticPool,sealed,validation,candidateAudit:rankedDevelopment.map(row=>({config:row.config,hash:row.hash,development:row.development,score:row.developmentScore})),universeSize:universe.length};
  }

  return{VERSION,stableHash,indexAtOrBefore,buildPeriods,stockFeature,rankPeriod,selectHoldings,portfolioStats,simulatePortfolio,simulateBuyHold,stitchOuterOos,runResearch};
});
