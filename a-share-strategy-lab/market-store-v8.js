(function(root){
  'use strict';
  const DB_NAME='a-share-lab-market-v8',DB_VERSION=1,STORE='symbols',MANIFEST_KEY='__manifest__';
  let manifestPromise=null;
  const memory=new Map();
  const baseUrl=()=>new URL('data/market/',document.baseURI);
  const nextDay=date=>{const value=new Date(date+'T00:00:00Z');value.setUTCDate(value.getUTCDate()+1);return value.toISOString().slice(0,10)};
  const previousDay=date=>{const value=new Date(date+'T00:00:00Z');value.setUTCDate(value.getUTCDate()-1);return value.toISOString().slice(0,10)};
  const mergeBars=(...groups)=>[...new Map(groups.flat().filter(Boolean).map(bar=>[bar.date,bar])).values()].sort((a,b)=>a.date.localeCompare(b.date));
  const covers=(bars,start,end)=>{if(!bars.length||bars[0].date>start)return false;const gap=(new Date(end+'T00:00:00Z')-new Date(bars.at(-1).date+'T00:00:00Z'))/864e5;return gap<=7};
  function openDb(){return new Promise((resolve,reject)=>{if(!('indexedDB'in root)){resolve(null);return}const request=indexedDB.open(DB_NAME,DB_VERSION);request.onupgradeneeded=()=>{if(!request.result.objectStoreNames.contains(STORE))request.result.createObjectStore(STORE)};request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error)})}
  async function readIdb(key){try{const db=await openDb();if(!db)return null;return await new Promise((resolve,reject)=>{const tx=db.transaction(STORE,'readonly'),request=tx.objectStore(STORE).get(key);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error)})}catch{return null}}
  async function writeIdb(key,value){try{const db=await openDb();if(!db)return;await new Promise((resolve,reject)=>{const tx=db.transaction(STORE,'readwrite');tx.objectStore(STORE).put(value,key);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error)})}catch{/* private mode or quota: memory cache still works */}}
  async function manifest(){
    if(manifestPromise)return manifestPromise;
    manifestPromise=(async()=>{try{const response=await fetch(new URL('manifest.json',baseUrl()),{cache:'no-store'});if(!response.ok)throw new Error('manifest unavailable');const value=await response.json();await writeIdb(MANIFEST_KEY,value);return value}catch{return await readIdb(MANIFEST_KEY)||{schemaVersion:1,dataVersion:'runtime-only',entries:{},updatedAt:null}}})();
    return manifestPromise;
  }
  async function staticParts(entry){
    if(!entry)return[];const parts=(entry.parts||entry.files||[]).filter(part=>typeof part==='string'||part?.path);
    const groups=await Promise.all(parts.map(async part=>{const path=typeof part==='string'?part:part.path,response=await fetch(new URL(path,baseUrl()),{cache:'force-cache'});if(!response.ok)throw new Error(`static part ${path} unavailable`);const payload=await response.json();return Array.isArray(payload)?payload:payload.rows||[]}));
    return mergeBars(...groups);
  }
  async function load(asset,start,end,remoteLoader){
    const key=`${asset.type}:${asset.code}`,memo=memory.get(key)||[],cached=await readIdb(key),meta=await manifest();let bars=mergeBars(cached?.bars||[],memo),source=cached?.source||'浏览器持久缓存',persisted=Boolean(bars.length),entry=meta.entries?.[key]||meta.entries?.[asset.code];
    if(entry){try{bars=mergeBars(bars,await staticParts(entry));source=`GitHub静态历史库 · ${meta.source||'已校验日线'}`;persisted=true}catch(error){if(!bars.length)throw error}}
    if(!covers(bars,start,end)&&typeof remoteLoader==='function'){
      const gaps=[];
      if(!bars.length)gaps.push([start,end]);
      else{
        if(bars[0].date>start)gaps.push([start,previousDay(bars[0].date)]);
        if(bars.at(-1).date<end)gaps.push([nextDay(bars.at(-1).date),end]);
      }
      for(const [gapStart,gapEnd] of gaps){if(gapStart<=gapEnd){const value=await remoteLoader(gapStart,gapEnd);bars=mergeBars(bars,value.bars||[])}}
      if(gaps.length){source=persisted?'静态历史 + 东方财富缺口补齐':'东方财富即时日线（已存本机）';persisted=false}
    }
    if(!bars.length)throw new Error('请求区间没有可用行情');memory.set(key,bars);await writeIdb(key,{bars,source,updatedAt:new Date().toISOString(),dataVersion:meta.dataVersion||'runtime-only'});
    return{bars:bars.filter(bar=>bar.date>=start&&bar.date<=end),allBars:bars,name:asset.name,source,dataVersion:meta.dataVersion||'runtime-only',persisted,asOf:bars.at(-1)?.date||null,formal:Boolean(entry&&persisted)};
  }
  async function status(){const value=await manifest();return{dataVersion:value.dataVersion||'runtime-only',updatedAt:value.updatedAt,lastTradingDate:value.lastTradingDate||null,staticSymbols:Object.keys(value.entries||{}).length,source:value.source||'本机缓存 + 东方财富增量'}}
  root.MarketDataStoreV8={load,status,manifest,mergeBars,clearMemory:()=>memory.clear()};
})(typeof window!=='undefined'?window:globalThis);
