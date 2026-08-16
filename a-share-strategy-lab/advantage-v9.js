(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  const pct = (value, digits = 1) =>
    Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
  const pp = (value, digits = 1) =>
    Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}pp` : "—";
  const dd = (value, digits = 1) =>
    Number.isFinite(Number(value)) ? `${(Math.abs(Number(value)) * 100).toFixed(digits)}%` : "—";
  const num = (value, digits = 2) =>
    Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  const cls = (value) => (Number(value) > 0 ? "positive" : Number(value) < 0 ? "negative" : "is-neutral");
  const state = { report: null, cap: "30", filter: "all" };

  const gateNames = {
    developmentDrawdownCap: "开发期组合回撤达标",
    developmentEveryStockDrawdownCap: "开发期逐股回撤达标",
    sealedPositive: "时间OOS收益为正",
    sealedExcessPositive: "时间OOS跑赢同期持有",
    sealedDrawdownCap: "时间OOS回撤达标",
    sealedEveryStockDrawdownCap: "时间OOS逐股回撤达标",
    sealedDrawdownImproved: "时间OOS回撤优于持有",
    sealedCalmarImproved: "时间OOS Calmar不低于持有",
    crossStockWinRate: "跨股票胜率达标",
    doubleCostRobust: "双倍成本仍有超额",
    oneDayDelayRobust: "额外延迟1日仍有超额",
    holdoutSealedPositive: "留出股票OOS收益为正",
    holdoutSealedExcessPositive: "留出股票OOS跑赢持有",
    holdoutAggregateDrawdownCap: "留出股票组合回撤达标",
    holdoutEveryStockDrawdownCap: "留出股票逐股回撤达标",
    holdoutAssetWinRate: "留出股票胜率达标",
    timeFoldWinRate: "OOS时间折胜率≥60%",
    portfolioDrawdownCap: "开发期组合回撤达标",
    sealedPortfolioDrawdownCap: "封存期组合回撤达标",
    holdoutExcessPositive: "留出股票超额为正",
    developmentPortfolioDrawdownCap: "开发期组合回撤达标",
    developmentUniverseSealedPassed: "原开发股票池的时间OOS通过",
    expandedSealedPositive: "扩展股票池OOS收益为正",
    expandedSealedExcessPositive: "扩展股票池OOS跑赢同期持有",
    expandedSealedDrawdownCap: "扩展股票池OOS回撤达标",
    expandedSealedDrawdownImproved: "扩展股票池OOS回撤优于持有",
    expandedSealedCalmarImproved: "扩展股票池OOS Calmar优于持有",
    holdoutPoolPositive: "16只完全留出股票独立成池收益为正",
    holdoutPoolExcessPositive: "完全留出股票池跑赢同期持有",
    holdoutPoolDrawdownCap: "完全留出股票池回撤达标",
    holdoutPoolDrawdownImproved: "完全留出股票池回撤优于持有",
    holdoutPoolCalmarImproved: "完全留出股票池 Calmar 优于持有",
  };

  function strategies(tier) {
    return tier.portfolioStudy?.top5 || tier.portfolioStudy?.top3 || tier.portfolioTop3 || tier.top3 || [];
  }

  function featuredStrategy(tier) {
    const rows = strategies(tier);
    return rows.find(strategyPassed) || rows[0] || null;
  }

  function aggregateOf(strategy, period) {
    const node = strategy?.[period];
    return node?.aggregate || node || null;
  }

  function mainOos(strategy) {
    return aggregateOf(strategy, "allSealed") || aggregateOf(strategy, "sealed");
  }

  function perStock(strategy) {
    return strategy?.allSealed?.perStock || strategy?.sealed?.perStock || strategy?.assetResults || [];
  }

  function strategyPassed(strategy) {
    return strategy?.extendedPassed ?? strategy?.gates?.passed ?? false;
  }

  function download(name, type, content) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function metricTriplet(aggregate) {
    if (!aggregate) return "<dl><div><dt>策略收益</dt><dd>—</dd></div><div><dt>超额</dt><dd>—</dd></div><div><dt>最大回撤</dt><dd>—</dd></div></dl>";
    return `<dl><div><dt>OOS策略</dt><dd class="${cls(aggregate.strategy?.return)}">${pct(aggregate.strategy?.return)}</dd></div><div><dt>相对持有</dt><dd class="${cls(aggregate.excessReturn)}">${pp(aggregate.excessReturn)}</dd></div><div><dt>最大回撤</dt><dd>${dd(aggregate.strategy?.drawdown)}</dd></div></dl>`;
  }

  function renderComparison() {
    $("capComparison").innerHTML = ["20", "30"]
      .map((cap) => {
        const tier = state.report.tiers[cap];
        const rows = strategies(tier);
        const leader = featuredStrategy(tier);
        const aggregate = mainOos(leader);
        const qualified = rows.filter(strategyPassed).length;
        return `<article data-cap="${cap}"><header><div><span>${cap}% 回撤档</span><h3>${leader ? `#${leader.rank} ${esc(leader.familyName || leader.name)}` : "无开发期可行策略"}</h3></div><span class="strategyStatus ${qualified ? "pass" : "fail"}">${qualified ? `${qualified} 个严格通过` : "未严格通过"}</span></header>${metricTriplet(aggregate)}<small>开发期冻结 ${rows.length} 个候选；默认展开开发排名最靠前的通过者，OOS 不改变原排名。</small></article>`;
      })
      .join("");
  }

  function renderTopCards(tier) {
    const rows = strategies(tier);
    if (!rows.length) {
      const closest = tier.candidateAudit?.[0];
      return `<article class="topStrategyCard"><header><span class="strategyRank">—</span><span class="strategyStatus fail">无可行候选</span></header><h4>当前硬约束下没有可冻结策略</h4><small>${closest ? `最近候选：${esc(closest.familyName)}；最差单股开发回撤 ${pct(-Math.abs(closest.worstDevelopmentStockDrawdown || 0))}` : "请查看失败门槛。"}</small><p class="frozenParams">系统不会为了展示好看结果而使用封存期重新选参。</p></article>`;
    }
    return rows
      .map((row, index) => {
        const development = aggregateOf(row, "development");
        const oos = mainOos(row);
        const passed = strategyPassed(row);
        return `<article class="topStrategyCard"><header><span class="strategyRank">${row.rank || index + 1}</span><span class="strategyStatus ${passed ? "pass" : "fail"}">${passed ? "严格通过" : "仅观察"}</span></header><h4>${esc(row.familyName || row.name || row.id)}</h4><small>开发期排名已冻结 · ${esc(row.id)}</small><div class="topStrategyMetrics"><div><span>开发期收益</span><b class="${cls(development?.strategy?.return)}">${pct(development?.strategy?.return)}</b></div><div><span>OOS收益</span><b class="${cls(oos?.strategy?.return)}">${pct(oos?.strategy?.return)}</b></div><div><span>OOS超额</span><b class="${cls(oos?.excessReturn)}">${pp(oos?.excessReturn)}</b></div><div><span>OOS回撤</span><b>${dd(oos?.strategy?.drawdown)}</b></div><div><span>股票贡献胜率</span><b>${pct(oos?.assetWinRate ?? row.assetWinRate)}</b></div><div><span>平均仓位</span><b>${pct(oos?.averageExposure ?? row.averageExposure)}</b></div></div><p class="frozenParams"><b>冻结参数：</b>${esc(JSON.stringify(row.params || row.config || {}))}</p></article>`;
      })
      .join("");
  }

  function renderOosTable(tier) {
    const rows = strategies(tier);
    if (!rows.length) return `<div class="detailsBody"><p>开发期没有满足该档回撤硬约束的候选，因此没有打开封存 OOS 进行“补选”。</p></div>`;
    return `<table><thead><tr><th>开发排名 / 家族</th><th>OOS策略</th><th>同期持有</th><th>超额</th><th>策略回撤</th><th>持有回撤</th><th>Calmar</th><th>结论</th></tr></thead><tbody>${rows
      .map((row, index) => {
        const a = mainOos(row);
        return `<tr><td data-label="开发排名 / 家族"><b>#${row.rank || index + 1} ${esc(row.familyName || row.name)}</b><br><small>${esc(row.id)}</small></td><td data-label="OOS策略" class="${cls(a?.strategy?.return)}">${pct(a?.strategy?.return)}</td><td data-label="同期持有">${pct(a?.benchmark?.return)}</td><td data-label="超额" class="${cls(a?.excessReturn)}">${pp(a?.excessReturn)}</td><td data-label="策略回撤">${dd(a?.strategy?.drawdown)}</td><td data-label="持有回撤">${dd(a?.benchmark?.drawdown)}</td><td data-label="Calmar">${num(a?.strategy?.calmar)} / ${num(a?.benchmark?.calmar)}</td><td data-label="结论"><span class="strategyStatus ${strategyPassed(row) ? "pass" : "fail"}">${strategyPassed(row) ? "通过" : "未通过"}</span></td></tr>`;
      })
      .join("")}</tbody></table>`;
  }

  function normalizedStockRow(row) {
    const strategy = row.strategy || row.strategyStats || {};
    const benchmark = row.benchmark || row.benchmarkStats || {};
    const excessReturn = row.excessReturn ?? row.excess ?? (strategy.return - benchmark.return);
    return {
      code: row.code,
      name: row.name || row.code,
      strategy,
      benchmark,
      excessReturn,
      turnover: row.turnover,
      tradeCount: row.tradeCount,
      contribution: row.contribution,
    };
  }

  function renderAssets(panel, tier, cap) {
    const primary = featuredStrategy(tier);
    const holdoutCodes = new Set(state.report.protocol.holdoutStocks.map((row) => row.code));
    let rows = perStock(primary).map(normalizedStockRow);
    if (state.filter === "win") rows = rows.filter((row) => row.excessReturn > 0);
    if (state.filter === "lose") rows = rows.filter((row) => row.excessReturn <= 0);
    if (state.filter === "holdout") rows = rows.filter((row) => holdoutCodes.has(row.code));
    const contributionMode = rows.some((row) => Number.isFinite(Number(row.contribution)));
    panel.querySelector("[data-role='asset-table']").innerHTML = primary
      ? contributionMode
        ? `<table><thead><tr><th>股票</th><th>策略组合贡献</th><th>同池持有贡献</th><th>贡献差</th><th>成交次数</th><th>股票分组</th><th>结果</th></tr></thead><tbody>${rows
            .map((row) => {
              const passed = row.excessReturn > 0;
              return `<tr><td data-label="股票"><b>${esc(row.name)}</b><br><small>${esc(row.code)}</small></td><td data-label="策略组合贡献" class="${cls(row.strategy.return)}">${pct(row.strategy.return)}</td><td data-label="同池持有贡献">${pct(row.benchmark.return)}</td><td data-label="贡献差" class="${cls(row.excessReturn)}">${pp(row.excessReturn)}</td><td data-label="成交次数">${row.tradeCount ?? "—"}</td><td data-label="股票分组">${holdoutCodes.has(row.code) ? "完全留出" : "开发股票"}</td><td data-label="结果"><span class="strategyStatus ${passed ? "pass" : "fail"}">${passed ? "正超额贡献" : "负超额贡献"}</span></td></tr>`;
            })
            .join("")}</tbody></table>`
        : `<table><thead><tr><th>股票</th><th>OOS策略</th><th>同期持有</th><th>超额</th><th>策略 / 持有回撤</th><th>换手 / 成交</th><th>股票分组</th><th>验收</th></tr></thead><tbody>${rows
          .map((row) => {
            const passed = row.excessReturn > 0 && Math.abs(row.strategy.drawdown || 0) <= Number(cap) / 100;
            return `<tr><td data-label="股票"><b>${esc(row.name)}</b><br><small>${esc(row.code)}</small></td><td data-label="OOS策略" class="${cls(row.strategy.return)}">${pct(row.strategy.return)}</td><td data-label="同期持有">${pct(row.benchmark.return)}</td><td data-label="超额" class="${cls(row.excessReturn)}">${pp(row.excessReturn)}</td><td data-label="策略 / 持有回撤">${dd(row.strategy.drawdown)} / ${dd(row.benchmark.drawdown)}</td><td data-label="换手 / 成交">${Number.isFinite(row.turnover) ? `${num(row.turnover, 1)}×` : "—"} / ${row.tradeCount ?? "—"}</td><td data-label="股票分组">${holdoutCodes.has(row.code) ? "完全留出" : "开发股票"}</td><td data-label="验收"><span class="strategyStatus ${passed ? "pass" : "fail"}">${passed ? "优势" : "未通过"}</span></td></tr>`;
          })
          .join("")}</tbody></table>`
      : `<div class="detailsBody"><p>无逐股 OOS 结果。</p></div>`;
    panel.querySelector("[data-role='asset-count']").textContent = `${rows.length} 只股票`;
  }

  function allGates(strategy) {
    return { ...(strategy?.gates || {}), ...(strategy?.holdoutGates || {}) };
  }

  function renderGates(primary) {
    const gates = allGates(primary);
    if (!Object.keys(gates).length) return `<li class="fail"><b>未运行</b><span>没有可行候选可进入封存验收</span></li>`;
    return Object.entries(gates)
      .filter(([key]) => key !== "passed")
      .map(([key, value]) => `<li class="${value ? "pass" : "fail"}"><b>${value ? "通过" : "未通过"}</b><span>${esc(gateNames[key] || key)}</span></li>`)
      .join("");
  }

  function renderDiagnostics(primary) {
    if (!primary) return `<p>没有可诊断的封存策略。</p>`;
    const normal = mainOos(primary);
    const doubleCost = aggregateOf(primary.sensitivity?.doubleCost, "aggregate") || primary.sensitivity?.doubleCost?.aggregate;
    const delay = aggregateOf(primary.sensitivity?.oneDayDelay, "aggregate") || primary.sensitivity?.oneDayDelay?.aggregate;
    return `<dl><div><dt>基础成本超额</dt><dd class="${cls(normal?.excessReturn)}">${pp(normal?.excessReturn)}</dd></div><div><dt>双倍成本超额</dt><dd class="${cls(doubleCost?.excessReturn)}">${pp(doubleCost?.excessReturn)}</dd></div><div><dt>延迟1日超额</dt><dd class="${cls(delay?.excessReturn)}">${pp(delay?.excessReturn)}</dd></div><div><dt>基础换手</dt><dd>${num(normal?.averageTurnover, 1)}×</dd></div></dl>`;
  }

  function renderStability(primary) {
    if (!primary) return `<p>没有可诊断的封存策略。</p>`;
    const holdout = primary.assetHoldout?.sealed?.aggregate;
    const folds = primary.timeFolds || [];
    return `<dl><div><dt>OOS时间折胜率</dt><dd>${pct(primary.timeFoldWinRate)}</dd></div><div><dt>留出股票胜率</dt><dd>${pct(holdout?.assetWinRate ?? primary.holdoutAssetWinRate)}</dd></div><div><dt>留出股票超额</dt><dd class="${cls(holdout?.excessReturn ?? primary.holdoutExcessReturn)}">${pp(holdout?.excessReturn ?? primary.holdoutExcessReturn)}</dd></div><div><dt>时间折数量</dt><dd>${folds.length || "—"}</dd></div></dl>`;
  }

  function renderFailures(primary, tier) {
    const failed = Object.entries(allGates(primary)).filter(([key, value]) => key !== "passed" && !value);
    if (!primary) {
      const closest = tier.candidateAudit?.slice(0, 8) || [];
      return `<div class="failureList">${closest
        .map((row) => `<div class="failureRow"><b>${esc(row.familyName)}</b><span>开发期最差单股回撤 ${pct(-Math.abs(row.worstDevelopmentStockDrawdown || 0))}</span><span class="failureReason">不满足该档开发期硬约束，未打开OOS补选</span></div>`)
        .join("") || "<p>没有候选审计记录。</p>"}</div>`;
    }
    return `<div class="failureList">${failed
      .map(([key]) => `<div class="failureRow"><b>${esc(primary.familyName || primary.name)}</b><span>${esc(key)}</span><span class="failureReason">${esc(gateNames[key] || "量化验收门槛未通过")}</span></div>`)
      .join("") || "<p class='positive'>所有预注册门槛均通过。</p>"}</div>`;
  }

  function renderPanel(cap) {
    const tier = state.report.tiers[cap];
    const study = tier.portfolioStudy || tier;
    const primary = featuredStrategy(tier);
    const passed = strategies(tier).filter(strategyPassed).length;
    const panel = document.createElement("section");
    panel.dataset.capPanel = cap;
    panel.dataset.printTitle = `${cap}%回撤策略报告`;
    panel.hidden = cap !== state.cap;
    panel.innerHTML = `
      <div class="reportSectionHead"><div><p class="kicker">DEVELOPMENT-FROZEN TOP5</p><h3>${cap}% 回撤档：开发期冻结策略</h3></div><span>${passed ? `${passed} 个通过全部封存门槛` : "没有策略通过全部门槛"}</span></div>
      <div class="advantageTopStrategies">${renderTopCards(tier)}</div>
      <div class="reportSectionHead"><div><p class="kicker">SEALED OOS</p><h3>统一股票池 OOS 与持有比较</h3></div><span>不按 OOS 收益重新排序</span></div>
      <div class="advantageTableWrap">${renderOosTable(tier)}</div>
      <ul class="gateGrid">${renderGates(primary)}</ul>
      <div class="reportSectionHead"><div><p class="kicker">CROSS-STOCK EVIDENCE</p><h3>逐股封存验证</h3></div><span data-role="asset-count">0 只股票</span></div>
      <div class="assetFilters" data-role="asset-filters"><button class="active" type="button" data-filter="all" aria-pressed="true">全部</button><button type="button" data-filter="win" aria-pressed="false">仅跑赢</button><button type="button" data-filter="lose" aria-pressed="false">未跑赢</button><button type="button" data-filter="holdout" aria-pressed="false">完全留出股票</button></div>
      <div class="advantageTableWrap" data-role="asset-table"></div>
      <div class="diagnosticGrid"><article class="diagnosticCard"><h3>成本与执行压力测试</h3>${renderDiagnostics(primary)}</article><article class="diagnosticCard"><h3>时间与股票稳定性</h3>${renderStability(primary)}</article></div>
      <details class="failureSamples"><summary>查看未通过门槛与失败样本</summary><div class="detailsBody">${renderFailures(primary, tier)}</div></details>
      <details class="reproducibility"><summary>复现信息与数据限制</summary><dl class="reproducibilityGrid"><div><dt>runId</dt><dd>${esc(study.runId)}</dd></div><div><dt>selectionHash</dt><dd>${esc(study.selectionHash)}</dd></div><div><dt>dataVersion</dt><dd>${esc(state.report.data.dataVersion)}</dd></div><div><dt>策略核心</dt><dd>${esc(state.report.strategyVersion)}</dd></div><div><dt>开发期</dt><dd>${esc(study.split.development.join(" — "))}</dd></div><div><dt>封存OOS</dt><dd>${esc(study.split.sealed.join(" — "))}</dd></div><div><dt>股票拆分</dt><dd>${state.report.protocol.developmentStockCount} 开发 / ${state.report.protocol.holdoutStockCount} 留出</dd></div><div><dt>残余偏差</dt><dd>${esc(state.report.protocol.disclaimer)}</dd></div></dl></details>`;
    panel.querySelectorAll("[data-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        state.filter = button.dataset.filter;
        panel.querySelectorAll("[data-filter]").forEach((node) => {
          const active = node === button;
          node.classList.toggle("active", active);
          node.setAttribute("aria-pressed", String(active));
        });
        renderAssets(panel, tier, cap);
      });
    });
    renderAssets(panel, tier, cap);
    return panel;
  }

  function switchCap(cap) {
    state.cap = cap;
    state.filter = "all";
    document.querySelectorAll("#drawdownTabs [data-cap]").forEach((button) => {
      const active = button.dataset.cap === cap;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("#advantagePanels [data-cap-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.capPanel !== cap;
    });
  }

  function exportCsv() {
    const rows = [["回撤档", "开发排名", "策略", "股票", "代码", "OOS策略收益", "同期持有收益", "超额", "策略回撤", "持有回撤"]];
    for (const cap of ["20", "30"]) {
      const tier = state.report.tiers[cap];
      for (const strategy of strategies(tier)) {
        for (const raw of perStock(strategy)) {
          const row = normalizedStockRow(raw);
          rows.push([cap, strategy.rank, strategy.familyName || strategy.name, row.name, row.code, row.strategy.return, row.benchmark.return, row.excessReturn, row.strategy.drawdown, row.benchmark.drawdown]);
        }
      }
    }
    const cell = (value) => {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    download("v9-20-30-drawdown-per-stock.csv", "text/csv;charset=utf-8", `\ufeff${rows.map((row) => row.map(cell).join(",")).join("\n")}`);
  }

  async function init() {
    try {
      const response = await fetch(new URL("data/v9/advantage-report.json", document.baseURI), { cache: "no-store" });
      if (!response.ok) throw new Error(`报告 HTTP ${response.status}`);
      state.report = await response.json();
      $("advantageProtocolNote").textContent = `${state.report.protocol.selection}。研究 ${state.report.protocol.start}—${state.report.protocol.end}；${state.report.data.stockCount} 只股票，${state.report.protocol.developmentStockCount} 只开发、${state.report.protocol.holdoutStockCount} 只完全留出。`;
      renderComparison();
      const host = $("advantagePanels");
      host.replaceChildren(renderPanel("20"), renderPanel("30"));
      document.querySelectorAll("#drawdownTabs [data-cap]").forEach((button) => (button.onclick = () => switchCap(button.dataset.cap)));
      $("exportAdvantageJson").onclick = () => download("v9-20-30-drawdown-report.json", "application/json;charset=utf-8", JSON.stringify(state.report, null, 2));
      $("exportAdvantageCsv").onclick = exportCsv;
      $("printAdvantageReport").onclick = () => window.print();
      $("advantageStatus").className = "v8Status done";
      $("advantageStatus").innerHTML = `<i></i><span>报告已冻结 · 数据更新至 ${esc(state.report.data.lastTradingDate)} · reportHash ${esc(state.report.reportHash)}</span>`;
    } catch (error) {
      $("advantageStatus").className = "v8Status error";
      $("advantageStatus").innerHTML = `<i></i><span>${esc(error.message || error)}；正式报告不会回退到演示数据。</span>`;
    }
  }

  init();
})();
