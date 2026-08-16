(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.PortfolioAlphaCoreV9 = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const VERSION = "portfolio-alpha-core-v9.0.0";
  const EPSILON = 1e-10;
  const TRADING_DAYS = 252;

  const clamp = (value, minimum, maximum) =>
    Math.max(minimum, Math.min(maximum, Number(value) || 0));
  const mean = (values) =>
    values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const last = (values) => values[values.length - 1];
  const safeReturn = (end, start) =>
    Number.isFinite(end) && Number.isFinite(start) && start > 0 ? end / start - 1 : 0;

  function canonicalize(value) {
    if (Array.isArray(value)) return value.map(canonicalize);
    if (value && typeof value === "object") {
      return Object.fromEntries(
        Object.keys(value)
          .sort()
          .map((key) => [key, canonicalize(value[key])]),
      );
    }
    return value;
  }

  function stableHash(value) {
    const text = typeof value === "string" ? value : JSON.stringify(canonicalize(value));
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return ("00000000" + (hash >>> 0).toString(16)).slice(-8);
  }

  function normalizeTargetDrawdown(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 0.3;
    return clamp(numeric > 1 ? numeric / 100 : numeric, 0.05, 0.5);
  }

  function normalizeBars(rows) {
    const seen = new Map();
    for (const source of rows || []) {
      const bar = {
        date: String(source.date || ""),
        open: Number(source.open),
        close: Number(source.close),
        high: Number(source.high),
        low: Number(source.low),
        volume: Number(source.volume || 0),
      };
      if (
        /^\d{4}-\d{2}-\d{2}$/.test(bar.date) &&
        bar.open > 0 &&
        bar.close > 0 &&
        bar.high >= Math.max(bar.open, bar.close) &&
        bar.low > 0 &&
        bar.low <= Math.min(bar.open, bar.close)
      ) {
        seen.set(bar.date, bar);
      }
    }
    return [...seen.values()].sort((left, right) => left.date.localeCompare(right.date));
  }

  function normalizeAssets(input) {
    const source = Array.isArray(input)
      ? input
      : Object.entries(input || {}).map(([code, bars]) => ({ code, bars }));
    const used = new Set();
    const assets = source
      .map((asset, index) => ({
        code: String(asset.code || `asset-${index + 1}`),
        name: String(asset.name || asset.code || `资产${index + 1}`),
        bars: normalizeBars(asset.bars || []),
      }))
      .filter((asset) => {
        if (used.has(asset.code) || asset.bars.length < 2) return false;
        used.add(asset.code);
        return true;
      });
    if (!assets.length) throw new Error("没有可用的股票行情");
    return assets;
  }

  function prepareSeries(asset, calendar, dateToGlobalIndex) {
    const exactIndexByGlobal = new Int32Array(calendar.length);
    const latestIndexByGlobal = new Int32Array(calendar.length);
    exactIndexByGlobal.fill(-1);
    latestIndexByGlobal.fill(-1);
    asset.bars.forEach((bar, localIndex) => {
      const globalIndex = dateToGlobalIndex.get(bar.date);
      if (globalIndex != null) exactIndexByGlobal[globalIndex] = localIndex;
    });
    let latest = -1;
    for (let globalIndex = 0; globalIndex < calendar.length; globalIndex += 1) {
      if (exactIndexByGlobal[globalIndex] >= 0) latest = exactIndexByGlobal[globalIndex];
      latestIndexByGlobal[globalIndex] = latest;
    }
    const returns = new Float64Array(asset.bars.length);
    const returnPrefix = new Float64Array(asset.bars.length + 1);
    const squaredReturnPrefix = new Float64Array(asset.bars.length + 1);
    for (let index = 0; index < asset.bars.length; index += 1) {
      returns[index] = index
        ? safeReturn(asset.bars[index].close, asset.bars[index - 1].close)
        : 0;
      returnPrefix[index + 1] = returnPrefix[index] + returns[index];
      squaredReturnPrefix[index + 1] =
        squaredReturnPrefix[index] + returns[index] * returns[index];
    }
    return {
      ...asset,
      exactIndexByGlobal,
      latestIndexByGlobal,
      returns,
      returnPrefix,
      squaredReturnPrefix,
    };
  }

  function prepareUniverse(assetsInput, benchmarkInput) {
    const assets = normalizeAssets(assetsInput);
    const benchmarkBars = benchmarkInput?.bars ? normalizeBars(benchmarkInput.bars) : [];
    const dates = new Set();
    assets.forEach((asset) => asset.bars.forEach((bar) => dates.add(bar.date)));
    benchmarkBars.forEach((bar) => dates.add(bar.date));
    const calendar = [...dates].sort();
    const dateToGlobalIndex = new Map(calendar.map((date, index) => [date, index]));
    const preparedAssets = assets.map((asset) =>
      prepareSeries(asset, calendar, dateToGlobalIndex),
    );
    const benchmark = benchmarkBars.length
      ? prepareSeries(
          {
            code: String(benchmarkInput.code || "benchmark"),
            name: String(benchmarkInput.name || benchmarkInput.code || "市场基准"),
            bars: benchmarkBars,
          },
          calendar,
          dateToGlobalIndex,
        )
      : null;

    const equalMarketClose = new Float64Array(calendar.length);
    equalMarketClose[0] = 100;
    for (let globalIndex = 1; globalIndex < calendar.length; globalIndex += 1) {
      let sum = 0;
      let count = 0;
      for (const asset of preparedAssets) {
        const localIndex = asset.exactIndexByGlobal[globalIndex];
        if (localIndex > 0) {
          sum += asset.returns[localIndex];
          count += 1;
        }
      }
      equalMarketClose[globalIndex] =
        equalMarketClose[globalIndex - 1] * (1 + (count ? sum / count : 0));
    }
    const equalMarketPrefix = new Float64Array(calendar.length + 1);
    for (let index = 0; index < calendar.length; index += 1) {
      equalMarketPrefix[index + 1] = equalMarketPrefix[index] + equalMarketClose[index];
    }
    return {
      assets: preparedAssets,
      benchmark,
      calendar,
      dateToGlobalIndex,
      equalMarketClose,
      equalMarketPrefix,
    };
  }

  function candidate(params) {
    const normalized = canonicalize(params);
    return {
      id: `rotation-${stableHash(normalized)}`,
      family: normalized.momentumLookbacks.length === 1
        ? `momentum_${normalized.momentumLookbacks[0]}`
        : "momentum_blend",
      name: `${normalized.momentumLookbacks.join("/")}日动量 · Top${normalized.topN} · ${normalized.weighting === "inverseVol" ? "波动率倒数" : "等权"}`,
      params: normalized,
    };
  }

  function defaultDrawdownLadder(cap, style) {
    const templates = {
      defensive: [
        { at: cap * 0.25, exposure: 0.65 },
        { at: cap * 0.45, exposure: 0.4 },
        { at: cap * 0.7, exposure: 0.15 },
        { at: cap * 0.9, exposure: 0 },
      ],
      balanced: [
        { at: cap * 0.35, exposure: 0.8 },
        { at: cap * 0.6, exposure: 0.55 },
        { at: cap * 0.82, exposure: 0.25 },
        { at: cap * 0.95, exposure: 0 },
      ],
      loose: [
        { at: cap * 0.45, exposure: 0.9 },
        { at: cap * 0.7, exposure: 0.65 },
        { at: cap * 0.88, exposure: 0.35 },
        { at: cap * 0.98, exposure: 0 },
      ],
    };
    return templates[style] || templates.balanced;
  }

  function generateCandidates(targetDrawdown) {
    const cap = normalizeTargetDrawdown(targetDrawdown);
    const profiles = [[20], [60], [120], [20, 60], [60, 120], [20, 60, 120]];
    const rows = [];
    for (const rebalanceDays of [5, 10, 20]) {
      for (const momentumLookbacks of profiles) {
        for (const topN of [3, 5]) {
          for (const weighting of ["equal", "inverseVol"]) {
            const maximumLookback = Math.max(...momentumLookbacks);
            const styleSelector =
              (rebalanceDays + maximumLookback + topN + (weighting === "inverseVol" ? 1 : 0)) % 3;
            const drawdownStyle = ["defensive", "balanced", "loose"][styleSelector];
            const marketTrendMode = styleSelector % 2 ? "benchmark" : "equalWeight";
            rows.push(
              candidate({
                targetDrawdown: cap,
                rebalanceDays,
                momentumLookbacks,
                topN,
                retainBuffer: topN === 3 ? 1 : 2,
                weighting,
                volatilityLookback: 20,
                stockCap: topN === 3 ? 0.4 : 0.25,
                requirePositiveMomentum: true,
                marketTrendMode,
                marketMaPeriod: maximumLookback <= 20 ? 60 : 120,
                riskOffExposure: cap <= 0.2 ? 0 : 0.2,
                drawdownStyle,
                drawdownLadder: defaultDrawdownLadder(cap, drawdownStyle),
              }),
            );
          }
        }
      }
    }
    return rows;
  }

  function normalizeStrategy(source, targetDrawdown) {
    const input = source?.params ? { ...source.params } : { ...(source || {}) };
    const cap = normalizeTargetDrawdown(input.targetDrawdown ?? targetDrawdown);
    const lookbacks = (Array.isArray(input.momentumLookbacks)
      ? input.momentumLookbacks
      : [input.momentumLookback || 60])
      .map(Number)
      .filter((value) => [20, 60, 120].includes(value));
    const momentumLookbacks = lookbacks.length ? [...new Set(lookbacks)].sort((a, b) => a - b) : [60];
    const topN = Math.max(1, Math.round(Number(input.topN) || 5));
    const params = {
      targetDrawdown: cap,
      rebalanceDays: [5, 10, 20].includes(Number(input.rebalanceDays))
        ? Number(input.rebalanceDays)
        : 10,
      momentumLookbacks,
      topN,
      retainBuffer: Math.max(0, Math.round(Number(input.retainBuffer) || 0)),
      weighting: input.weighting === "inverseVol" ? "inverseVol" : "equal",
      volatilityLookback: Math.max(10, Math.round(Number(input.volatilityLookback) || 20)),
      stockCap: clamp(input.stockCap == null ? Math.max(0.2, 1 / topN) : input.stockCap, 0.05, 1),
      requirePositiveMomentum: input.requirePositiveMomentum !== false,
      marketTrendMode: ["benchmark", "equalWeight", "none"].includes(input.marketTrendMode)
        ? input.marketTrendMode
        : "benchmark",
      marketMaPeriod: [20, 60, 120].includes(Number(input.marketMaPeriod))
        ? Number(input.marketMaPeriod)
        : 120,
      riskOffExposure: clamp(input.riskOffExposure == null ? 0 : input.riskOffExposure, 0, 1),
      drawdownStyle: String(input.drawdownStyle || "balanced"),
      drawdownLadder: Array.isArray(input.drawdownLadder)
        ? input.drawdownLadder
            .map((row) => ({ at: clamp(row.at, 0, 0.95), exposure: clamp(row.exposure, 0, 1) }))
            .sort((left, right) => left.at - right.at)
        : defaultDrawdownLadder(cap, input.drawdownStyle || "balanced"),
    };
    const metadata = source?.params ? source : candidate(params);
    return {
      id: String(metadata.id || `rotation-${stableHash(params)}`),
      name: String(metadata.name || "跨股票动量轮动"),
      family: String(metadata.family || "momentum_rotation"),
      params,
    };
  }

  function periodBounds(prepared, options) {
    const calendar = prepared.calendar;
    const start = options.start || calendar[0];
    const end = options.end || last(calendar);
    let startIndex = 0;
    while (startIndex < calendar.length && calendar[startIndex] < start) startIndex += 1;
    let endIndex = calendar.length - 1;
    while (endIndex >= 0 && calendar[endIndex] > end) endIndex -= 1;
    if (startIndex > endIndex || startIndex >= calendar.length || endIndex < 0) {
      throw new Error("回测区间没有交易日");
    }
    return { startIndex, endIndex, start: calendar[startIndex], end: calendar[endIndex] };
  }

  function localMomentum(asset, localIndex, lookbacks) {
    if (localIndex < Math.max(...lookbacks)) return NaN;
    let score = 0;
    for (const lookback of lookbacks) {
      const periodReturn = safeReturn(
        asset.bars[localIndex].close,
        asset.bars[localIndex - lookback].close,
      );
      score += Math.pow(Math.max(EPSILON, 1 + periodReturn), TRADING_DAYS / lookback) - 1;
    }
    return score / lookbacks.length;
  }

  function localVolatility(asset, localIndex, period) {
    if (localIndex < period) return NaN;
    const start = localIndex - period + 1;
    const end = localIndex + 1;
    const total = asset.returnPrefix[end] - asset.returnPrefix[start];
    const squared = asset.squaredReturnPrefix[end] - asset.squaredReturnPrefix[start];
    const variance = Math.max(0, (squared - (total * total) / period) / (period - 1));
    return Math.sqrt(variance) * Math.sqrt(TRADING_DAYS);
  }

  function marketTrendExposure(prepared, globalIndex, params) {
    if (params.marketTrendMode === "none") return 1;
    const period = params.marketMaPeriod;
    if (params.marketTrendMode === "benchmark" && prepared.benchmark) {
      const benchmark = prepared.benchmark;
      const localIndex = benchmark.latestIndexByGlobal[globalIndex];
      if (localIndex < period - 1) return params.riskOffExposure;
      let total = 0;
      for (let offset = 0; offset < period; offset += 1) {
        total += benchmark.bars[localIndex - offset].close;
      }
      return benchmark.bars[localIndex].close >= total / period ? 1 : params.riskOffExposure;
    }
    if (globalIndex < period - 1) return params.riskOffExposure;
    const average =
      (prepared.equalMarketPrefix[globalIndex + 1] -
        prepared.equalMarketPrefix[globalIndex + 1 - period]) /
      period;
    return prepared.equalMarketClose[globalIndex] >= average ? 1 : params.riskOffExposure;
  }

  function drawdownExposure(drawdown, params) {
    let exposure = 1;
    for (const step of params.drawdownLadder) {
      if (drawdown + EPSILON >= step.at) exposure = Math.min(exposure, step.exposure);
    }
    return exposure;
  }

  function cappedWeights(rawRows, totalExposure, stockCap) {
    const output = new Map();
    if (!rawRows.length || totalExposure <= EPSILON) return output;
    const targetTotal = Math.min(totalExposure, rawRows.length * stockCap);
    const remaining = new Map(rawRows.map((row) => [row.code, Math.max(EPSILON, row.raw)]));
    let unallocated = targetTotal;
    while (remaining.size && unallocated > EPSILON) {
      const rawTotal = [...remaining.values()].reduce((sum, value) => sum + value, 0);
      let cappedAny = false;
      for (const [code, raw] of [...remaining.entries()]) {
        const proposed = unallocated * (raw / rawTotal);
        if (proposed >= stockCap - EPSILON) {
          output.set(code, stockCap);
          unallocated -= stockCap;
          remaining.delete(code);
          cappedAny = true;
        }
      }
      if (!cappedAny) {
        for (const [code, raw] of remaining.entries()) {
          output.set(code, unallocated * (raw / rawTotal));
        }
        unallocated = 0;
      }
    }
    return output;
  }

  function buildTargets(prepared, globalIndex, strategy, heldCodes, portfolioDrawdown) {
    const params = strategy.params;
    const ranked = [];
    for (const asset of prepared.assets) {
      const localIndex = asset.latestIndexByGlobal[globalIndex];
      if (localIndex < 0) continue;
      const score = localMomentum(asset, localIndex, params.momentumLookbacks);
      if (!Number.isFinite(score) || (params.requirePositiveMomentum && score <= 0)) continue;
      ranked.push({ asset, code: asset.code, score, localIndex });
    }
    ranked.sort((left, right) => right.score - left.score || left.code.localeCompare(right.code));
    const allowedRank = params.topN + params.retainBuffer;
    const selected = ranked
      .filter((row, index) => heldCodes.has(row.code) && index < allowedRank)
      .slice(0, params.topN);
    const selectedCodes = new Set(selected.map((row) => row.code));
    for (const row of ranked) {
      if (selected.length >= params.topN) break;
      if (!selectedCodes.has(row.code)) {
        selected.push(row);
        selectedCodes.add(row.code);
      }
    }
    const marketExposure = marketTrendExposure(prepared, globalIndex, params);
    const portfolioExposure = drawdownExposure(portfolioDrawdown, params);
    const exposure = Math.min(marketExposure, portfolioExposure);
    const rawRows = selected.map((row) => ({
      code: row.code,
      raw: params.weighting === "inverseVol"
        ? 1 / Math.max(0.05, localVolatility(row.asset, row.localIndex, params.volatilityLookback) || 1)
        : 1,
    }));
    return {
      weights: cappedWeights(rawRows, exposure, params.stockCap),
      selected: selected.map((row) => ({ code: row.code, score: row.score })),
      exposure,
      marketExposure,
      drawdownExposure: portfolioExposure,
    };
  }

  function standardDeviation(values) {
    if (values.length < 2) return 0;
    const average = mean(values);
    return Math.sqrt(
      values.reduce((sum, value) => sum + (value - average) ** 2, 0) /
        (values.length - 1),
    );
  }

  function summarizeNav(daily, initialCapital) {
    if (!daily.length) {
      return {
        return: 0,
        annualizedReturn: 0,
        drawdown: 0,
        sharpe: 0,
        sortino: 0,
        calmar: 0,
        positiveDayRate: 0,
      };
    }
    const finalNav = last(daily).nav;
    const totalReturn = safeReturn(finalNav, initialCapital);
    const annualizedReturn = Math.pow(Math.max(EPSILON, 1 + totalReturn), TRADING_DAYS / Math.max(1, daily.length)) - 1;
    const returns = daily.map((row) => row.return);
    const volatility = standardDeviation(returns);
    const downside = standardDeviation(returns.filter((value) => value < 0));
    const drawdown = daily.reduce((maximum, row) => Math.max(maximum, row.drawdown), 0);
    return {
      return: totalReturn,
      annualizedReturn,
      drawdown,
      sharpe: volatility > EPSILON ? (mean(returns) / volatility) * Math.sqrt(TRADING_DAYS) : 0,
      sortino: downside > EPSILON ? (mean(returns) / downside) * Math.sqrt(TRADING_DAYS) : 0,
      calmar: drawdown > EPSILON ? annualizedReturn / drawdown : annualizedReturn > 0 ? 99 : 0,
      positiveDayRate: returns.length
        ? returns.filter((value) => value > 0).length / returns.length
        : 0,
    };
  }

  function monthlyFromDaily(daily) {
    const byMonth = new Map();
    daily.forEach((row) => {
      byMonth.set(row.date.slice(0, 7), {
        month: row.date.slice(0, 7),
        date: row.date,
        nav: row.nav,
        drawdown: row.drawdown,
      });
    });
    return [...byMonth.values()];
  }

  function executeTargets(state, prepared, globalIndex, pending, rates, dailyTradeRows) {
    if (!pending) return { turnover: 0, cost: 0 };
    const openPrices = new Map();
    let openValue = state.cash;
    for (const asset of prepared.assets) {
      const holding = state.holdings.get(asset.code);
      const exactIndex = asset.exactIndexByGlobal[globalIndex];
      const localIndex = asset.latestIndexByGlobal[globalIndex];
      const price = exactIndex >= 0
        ? asset.bars[exactIndex].open
        : localIndex >= 0
          ? asset.bars[localIndex].close
          : NaN;
      if (Number.isFinite(price)) openPrices.set(asset.code, price);
      if (holding && Number.isFinite(price)) openValue += holding.quantity * price;
    }
    if (openValue <= EPSILON) return { turnover: 0, cost: 0 };

    let tradedNotional = 0;
    let dayCost = 0;
    const orders = [];
    for (const asset of prepared.assets) {
      const exactIndex = asset.exactIndexByGlobal[globalIndex];
      if (exactIndex < 0) continue;
      const open = asset.bars[exactIndex].open;
      const current = state.holdings.get(asset.code)?.quantity || 0;
      const targetWeight = pending.weights.get(asset.code) || 0;
      const target = (openValue * targetWeight) / open;
      const difference = target - current;
      if (Math.abs(difference * open) > openValue * 1e-8) {
        orders.push({ asset, code: asset.code, open, current, difference, targetWeight });
      }
    }

    const tradeDate = prepared.calendar[globalIndex];
    const stampRate = rates.stamp == null
      ? tradeDate < "2023-08-28" ? 0.001 : 0.0005
      : rates.stamp;
    for (const order of orders.filter((row) => row.difference < 0)) {
      const quantity = Math.min(order.current, -order.difference);
      const grossAtOpen = quantity * order.open;
      const executionPrice = order.open * (1 - rates.slip);
      const executionNotional = quantity * executionPrice;
      const fee = executionNotional * rates.fee;
      const stampTax = executionNotional * stampRate;
      const slippage = grossAtOpen - executionNotional;
      const totalCost = fee + stampTax + slippage;
      state.cash += executionNotional - fee - stampTax;
      const remaining = order.current - quantity;
      if (remaining <= EPSILON) state.holdings.delete(order.code);
      else state.holdings.set(order.code, { quantity: remaining });
      state.costs.fees += fee;
      state.costs.stampTax += stampTax;
      state.costs.slippage += slippage;
      state.costByCode.set(order.code, (state.costByCode.get(order.code) || 0) + totalCost);
      tradedNotional += grossAtOpen;
      dayCost += totalCost;
      dailyTradeRows.push({
        signalDate: pending.signalDate,
        tradeDate,
        code: order.code,
        side: "sell",
        quantity,
        open: order.open,
        executionPrice,
        notional: grossAtOpen,
        cost: totalCost,
      });
    }

    const buyOrders = orders.filter((row) => row.difference > 0);
    const totalCashNeeded = buyOrders.reduce(
      (sum, order) => sum + order.difference * order.open * (1 + rates.slip) * (1 + rates.fee),
      0,
    );
    const scale = totalCashNeeded > state.cash && totalCashNeeded > 0
      ? Math.max(0, state.cash / totalCashNeeded)
      : 1;
    for (const order of buyOrders) {
      const quantity = order.difference * scale;
      if (quantity <= EPSILON) continue;
      const grossAtOpen = quantity * order.open;
      const executionPrice = order.open * (1 + rates.slip);
      const executionNotional = quantity * executionPrice;
      const fee = executionNotional * rates.fee;
      const slippage = executionNotional - grossAtOpen;
      const totalCost = fee + slippage;
      state.cash -= executionNotional + fee;
      if (Math.abs(state.cash) < 1e-6) state.cash = 0;
      const current = state.holdings.get(order.code)?.quantity || 0;
      state.holdings.set(order.code, { quantity: current + quantity });
      state.costs.fees += fee;
      state.costs.slippage += slippage;
      state.costByCode.set(order.code, (state.costByCode.get(order.code) || 0) + totalCost);
      tradedNotional += grossAtOpen;
      dayCost += totalCost;
      dailyTradeRows.push({
        signalDate: pending.signalDate,
        tradeDate,
        code: order.code,
        side: "buy",
        quantity,
        open: order.open,
        executionPrice,
        notional: grossAtOpen,
        cost: totalCost,
      });
    }
    return { turnover: tradedNotional / openValue, cost: dayCost };
  }

  function markPortfolio(state, prepared, globalIndex, previousGlobalIndex) {
    let nav = state.cash;
    let invested = 0;
    for (const asset of prepared.assets) {
      const holding = state.holdings.get(asset.code);
      const quantityAfter = holding?.quantity || 0;
      const quantityBefore = state.preTradeQuantities.get(asset.code) || 0;
      if (quantityAfter <= EPSILON && quantityBefore <= EPSILON) continue;
      const localIndex = asset.latestIndexByGlobal[globalIndex];
      if (localIndex < 0) continue;
      const close = asset.bars[localIndex].close;
      const marketValue = quantityAfter * close;
      nav += marketValue;
      invested += marketValue;

      const previousLocalIndex = previousGlobalIndex >= 0
        ? asset.latestIndexByGlobal[previousGlobalIndex]
        : -1;
      const previousClose = previousLocalIndex >= 0
        ? asset.bars[previousLocalIndex].close
        : close;
      const exactIndex = asset.exactIndexByGlobal[globalIndex];
      const open = exactIndex >= 0 ? asset.bars[exactIndex].open : previousClose;
      const grossPnl = quantityBefore * (open - previousClose) + quantityAfter * (close - open);
      state.pnlByCode.set(asset.code, (state.pnlByCode.get(asset.code) || 0) + grossPnl);
    }
    return { nav, invested };
  }

  function simulatePrepared(prepared, strategyInput, options, outputDetail) {
    const strategy = normalizeStrategy(strategyInput, options.targetDrawdown);
    const bounds = periodBounds(prepared, options);
    const initialCapital = Math.max(1, Number(options.capital) || 1_000_000);
    const rates = {
      fee: Math.max(0, Number(options.feeBp ?? 3)) / 10_000,
      slip: Math.max(0, Number(options.slipBp ?? 5)) / 10_000,
      stamp: Number.isFinite(Number(options.stampTaxBp))
        ? Math.max(0, Number(options.stampTaxBp)) / 10_000
        : null,
    };
    const executionDelayDays = Math.max(0, Math.floor(Number(options.executionDelayDays || 0)));
    const state = {
      cash: initialCapital,
      holdings: new Map(),
      preTradeQuantities: new Map(),
      costs: { fees: 0, slippage: 0, stampTax: 0 },
      costByCode: new Map(),
      pnlByCode: new Map(),
    };
    const daily = [];
    const trades = [];
    let pending = null;
    let peak = initialCapital;
    let previousNav = initialCapital;
    let portfolioDrawdown = 0;
    let totalTurnover = 0;
    let targetCount = 0;
    let exposureSum = 0;

    if (bounds.startIndex > 0) {
      const signalIndex = bounds.startIndex - 1;
      const initialTarget = buildTargets(prepared, signalIndex, strategy, new Set(), 0);
      pending = {
        ...initialTarget,
        signalDate: prepared.calendar[signalIndex],
        executeIndex: bounds.startIndex + executionDelayDays,
      };
    }

    for (let globalIndex = bounds.startIndex; globalIndex <= bounds.endIndex; globalIndex += 1) {
      state.preTradeQuantities = new Map(
        [...state.holdings.entries()].map(([code, holding]) => [code, holding.quantity]),
      );
      const dayTrades = [];
      const due = pending && globalIndex >= pending.executeIndex;
      const execution = due
        ? executeTargets(state, prepared, globalIndex, pending, rates, dayTrades)
        : { turnover: 0, cost: 0 };
      if (due) pending = null;
      trades.push(...dayTrades);
      totalTurnover += execution.turnover;
      const mark = markPortfolio(state, prepared, globalIndex, globalIndex - 1);
      const nav = Math.max(EPSILON, mark.nav);
      peak = Math.max(peak, nav);
      portfolioDrawdown = peak > 0 ? Math.max(0, 1 - nav / peak) : 0;
      const dayReturn = safeReturn(nav, previousNav);
      const date = prepared.calendar[globalIndex];
      const weights = {};
      for (const asset of prepared.assets) {
        const holding = state.holdings.get(asset.code);
        const localIndex = asset.latestIndexByGlobal[globalIndex];
        if (holding && localIndex >= 0) {
          weights[asset.code] = (holding.quantity * asset.bars[localIndex].close) / nav;
        }
      }
      daily.push({
        date,
        nav,
        return: dayReturn,
        drawdown: portfolioDrawdown,
        cash: state.cash,
        invested: mark.invested,
        exposure: mark.invested / nav,
        turnover: execution.turnover,
        costs: execution.cost,
        weights,
      });
      exposureSum += mark.invested / nav;
      previousNav = nav;

      const daysSinceStart = globalIndex - bounds.startIndex + 1;
      if (!pending && globalIndex < bounds.endIndex && daysSinceStart % strategy.params.rebalanceDays === 0) {
        const heldCodes = new Set(state.holdings.keys());
        const targets = buildTargets(
          prepared,
          globalIndex,
          strategy,
          heldCodes,
          portfolioDrawdown,
        );
        pending = {
          ...targets,
          signalDate: date,
          executeIndex: globalIndex + 1 + executionDelayDays,
        };
        targetCount += 1;
      }
    }

    const metrics = summarizeNav(daily, initialCapital);
    const totalCosts = state.costs.fees + state.costs.slippage + state.costs.stampTax;
    const contributions = prepared.assets
      .map((asset) => {
        const grossPnl = state.pnlByCode.get(asset.code) || 0;
        const cost = state.costByCode.get(asset.code) || 0;
        return {
          code: asset.code,
          name: asset.name,
          grossPnl,
          cost,
          netPnl: grossPnl - cost,
          contributionReturn: (grossPnl - cost) / initialCapital,
        };
      })
      .filter((row) => Math.abs(row.grossPnl) > EPSILON || row.cost > EPSILON)
      .sort((left, right) => right.netPnl - left.netPnl);
    const finalHoldings = prepared.assets
      .map((asset) => {
        const holding = state.holdings.get(asset.code);
        const localIndex = asset.latestIndexByGlobal[bounds.endIndex];
        if (!holding || localIndex < 0) return null;
        const marketValue = holding.quantity * asset.bars[localIndex].close;
        return {
          code: asset.code,
          name: asset.name,
          quantity: holding.quantity,
          marketValue,
          weight: marketValue / last(daily).nav,
        };
      })
      .filter(Boolean)
      .sort((left, right) => right.weight - left.weight);
    const result = {
      version: VERSION,
      strategy,
      period: { start: bounds.start, end: bounds.end, tradingDays: daily.length },
      metrics: {
        ...metrics,
        averageExposure: daily.length ? exposureSum / daily.length : 0,
        turnover: totalTurnover,
        annualizedTurnover: totalTurnover * (TRADING_DAYS / Math.max(1, daily.length)),
        tradeCount: trades.length,
        rebalanceCount: targetCount + (bounds.startIndex > 0 ? 1 : 0),
      },
      costs: {
        ...state.costs,
        total: totalCosts,
        returnDrag: totalCosts / initialCapital,
      },
      contributions,
      finalHoldings,
      monthlyNav: monthlyFromDaily(daily),
    };
    if (outputDetail !== "summary") {
      result.dailyNav = daily;
      result.trades = trades;
    }
    return result;
  }

  function simulateBuyHoldPrepared(prepared, options, outputDetail) {
    const bounds = periodBounds(prepared, options);
    const initialCapital = Math.max(1, Number(options.capital) || 1_000_000);
    const rates = {
      fee: Math.max(0, Number(options.feeBp ?? 3)) / 10_000,
      slip: Math.max(0, Number(options.slipBp ?? 5)) / 10_000,
      stamp: Math.max(0, Number(options.stampTaxBp ?? 5)) / 10_000,
    };
    const eligible = prepared.assets.filter(
      (asset) => asset.exactIndexByGlobal[bounds.startIndex] >= 0,
    );
    if (!eligible.length) throw new Error("区间首日没有可买入的基准成分");
    const weight = 1 / eligible.length;
    let cash = initialCapital;
    const holdings = new Map();
    const costs = { fees: 0, slippage: 0, stampTax: 0 };
    const trades = [];
    for (const asset of eligible) {
      const localIndex = asset.exactIndexByGlobal[bounds.startIndex];
      const open = asset.bars[localIndex].open;
      const executionPrice = open * (1 + rates.slip);
      const budget = initialCapital * weight;
      const quantity = budget / (executionPrice * (1 + rates.fee));
      const executionNotional = quantity * executionPrice;
      const fee = executionNotional * rates.fee;
      const slippage = quantity * (executionPrice - open);
      cash -= executionNotional + fee;
      holdings.set(asset.code, quantity);
      costs.fees += fee;
      costs.slippage += slippage;
      trades.push({
        tradeDate: bounds.start,
        code: asset.code,
        side: "buy",
        quantity,
        open,
        executionPrice,
        notional: quantity * open,
        cost: fee + slippage,
      });
    }
    if (Math.abs(cash) < 1e-6) cash = 0;
    let peak = initialCapital;
    let previousNav = initialCapital;
    const daily = [];
    for (let globalIndex = bounds.startIndex; globalIndex <= bounds.endIndex; globalIndex += 1) {
      let invested = 0;
      for (const asset of eligible) {
        const localIndex = asset.latestIndexByGlobal[globalIndex];
        if (localIndex >= 0) invested += holdings.get(asset.code) * asset.bars[localIndex].close;
      }
      const nav = cash + invested;
      peak = Math.max(peak, nav);
      const drawdown = peak > 0 ? Math.max(0, 1 - nav / peak) : 0;
      daily.push({
        date: prepared.calendar[globalIndex],
        nav,
        return: safeReturn(nav, previousNav),
        drawdown,
        cash,
        invested,
        exposure: invested / nav,
      });
      previousNav = nav;
    }
    const totalCosts = costs.fees + costs.slippage;
    const result = {
      name: "同宇宙等权买入持有",
      universeSize: eligible.length,
      period: { start: bounds.start, end: bounds.end, tradingDays: daily.length },
      metrics: summarizeNav(daily, initialCapital),
      costs: { ...costs, total: totalCosts, returnDrag: totalCosts / initialCapital },
      monthlyNav: monthlyFromDaily(daily),
    };
    if (outputDetail !== "summary") {
      result.dailyNav = daily;
      result.trades = trades;
    }
    return result;
  }

  function simulateMarketAsset(benchmark, prepared, options, outputDetail) {
    if (!benchmark) return null;
    const bounds = periodBounds(prepared, options);
    const startLocal = benchmark.latestIndexByGlobal[bounds.startIndex];
    const endLocal = benchmark.latestIndexByGlobal[bounds.endIndex];
    if (startLocal < 0 || endLocal < 0) return null;
    const initialCapital = Math.max(1, Number(options.capital) || 1_000_000);
    const startClose = benchmark.bars[startLocal].close;
    const daily = [];
    let peak = initialCapital;
    let previousNav = initialCapital;
    for (let globalIndex = bounds.startIndex; globalIndex <= bounds.endIndex; globalIndex += 1) {
      const localIndex = benchmark.latestIndexByGlobal[globalIndex];
      if (localIndex < 0) continue;
      const nav = initialCapital * (benchmark.bars[localIndex].close / startClose);
      peak = Math.max(peak, nav);
      daily.push({
        date: prepared.calendar[globalIndex],
        nav,
        return: safeReturn(nav, previousNav),
        drawdown: Math.max(0, 1 - nav / peak),
      });
      previousNav = nav;
    }
    const result = {
      code: benchmark.code,
      name: benchmark.name,
      period: { start: bounds.start, end: bounds.end, tradingDays: daily.length },
      metrics: summarizeNav(daily, initialCapital),
      monthlyNav: monthlyFromDaily(daily),
    };
    if (outputDetail !== "summary") result.dailyNav = daily;
    return result;
  }

  function simulatePortfolio(assets, benchmarkAsset, strategy, options) {
    const settings = options || {};
    const prepared = prepareUniverse(assets, benchmarkAsset);
    const detail = settings.outputDetail === "summary" ? "summary" : "full";
    const portfolio = simulatePrepared(prepared, strategy, settings, detail);
    const buyHold = simulateBuyHoldPrepared(prepared, settings, detail);
    const marketAsset = simulateMarketAsset(prepared.benchmark, prepared, settings, detail);
    return {
      ...portfolio,
      benchmark: {
        sameUniverseBuyHold: buyHold,
        marketAsset,
      },
      comparison: {
        excessReturn: portfolio.metrics.return - buyHold.metrics.return,
        drawdownImprovement: buyHold.metrics.drawdown - portfolio.metrics.drawdown,
        calmarImprovement: portfolio.metrics.calmar - buyHold.metrics.calmar,
      },
    };
  }

  function developmentHash(prepared, endDate) {
    let hash = 2166136261;
    const push = (text) => {
      for (let index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }
    };
    for (const asset of prepared.assets) {
      push(asset.code);
      for (const bar of asset.bars) {
        if (bar.date > endDate) break;
        push(`${bar.date}:${bar.open}:${bar.close};`);
      }
    }
    if (prepared.benchmark) {
      push(prepared.benchmark.code);
      for (const bar of prepared.benchmark.bars) {
        if (bar.date > endDate) break;
        push(`${bar.date}:${bar.close};`);
      }
    }
    return ("00000000" + (hash >>> 0).toString(16)).slice(-8);
  }

  function buildResearchSplit(prepared, options) {
    const bounds = periodBounds(prepared, options);
    let sealedStart = options.sealedStart;
    if (!sealedStart) {
      const ratio = clamp(options.developmentRatio == null ? 0.7 : options.developmentRatio, 0.5, 0.9);
      const cut = Math.min(
        bounds.endIndex - 1,
        Math.max(bounds.startIndex + 2, bounds.startIndex + Math.floor((bounds.endIndex - bounds.startIndex + 1) * ratio)),
      );
      sealedStart = prepared.calendar[cut];
    }
    let sealedIndex = bounds.startIndex;
    while (sealedIndex <= bounds.endIndex && prepared.calendar[sealedIndex] < sealedStart) {
      sealedIndex += 1;
    }
    if (sealedIndex <= bounds.startIndex || sealedIndex > bounds.endIndex) {
      throw new Error("开发期或封存样本外区间不足");
    }
    return {
      full: { start: bounds.start, end: bounds.end },
      development: {
        start: bounds.start,
        end: prepared.calendar[sealedIndex - 1],
        tradingDays: sealedIndex - bounds.startIndex,
      },
      sealed: {
        start: prepared.calendar[sealedIndex],
        end: bounds.end,
        tradingDays: bounds.endIndex - sealedIndex + 1,
      },
    };
  }

  function developmentScore(portfolio, benchmark) {
    const excess = portfolio.metrics.return - benchmark.metrics.return;
    const drawdownGain = benchmark.metrics.drawdown - portfolio.metrics.drawdown;
    return (
      excess * 100 +
      portfolio.metrics.annualizedReturn * 18 +
      drawdownGain * 25 +
      (portfolio.metrics.calmar - benchmark.metrics.calmar) * 1.5 -
      portfolio.metrics.annualizedTurnover * 0.04
    );
  }

  function runResearch(input) {
    const assets = input?.assets || input?.barsByCode;
    const benchmarkAsset = input?.benchmarkAsset || input?.benchmark;
    const options = { ...(input?.options || {}) };
    const cap = normalizeTargetDrawdown(options.targetDrawdown);
    const prepared = prepareUniverse(assets, benchmarkAsset);
    const split = buildResearchSplit(prepared, options);
    const candidates = (input?.candidates || generateCandidates(cap)).map((row) =>
      normalizeStrategy(row, cap),
    );
    if (!candidates.length) throw new Error("候选策略为空");
    const developmentOptions = {
      ...options,
      start: split.development.start,
      end: split.development.end,
    };
    const developmentBenchmark = simulateBuyHoldPrepared(prepared, developmentOptions, "summary");
    const candidateAudit = candidates.map((strategy) => {
      const result = simulatePrepared(prepared, strategy, developmentOptions, "summary");
      const feasible = result.metrics.drawdown <= cap + 1e-9;
      return {
        id: strategy.id,
        name: strategy.name,
        family: strategy.family,
        params: strategy.params,
        developmentFeasible: feasible,
        developmentScore: developmentScore(result, developmentBenchmark),
        development: {
          return: result.metrics.return,
          annualizedReturn: result.metrics.annualizedReturn,
          drawdown: result.metrics.drawdown,
          calmar: result.metrics.calmar,
          turnover: result.metrics.turnover,
          annualizedTurnover: result.metrics.annualizedTurnover,
          costs: result.costs.total,
          excessReturn: result.metrics.return - developmentBenchmark.metrics.return,
        },
      };
    });
    candidateAudit.sort(
      (left, right) =>
        Number(right.developmentFeasible) - Number(left.developmentFeasible) ||
        right.developmentScore - left.developmentScore ||
        left.id.localeCompare(right.id),
    );
    candidateAudit.forEach((row, index) => {
      row.developmentRank = index + 1;
    });
    const frozenRows = candidateAudit.slice(0, Math.min(5, candidateAudit.length));
    const frozenTop5 = frozenRows.map((row, index) => ({
      rank: index + 1,
      id: row.id,
      name: row.name,
      family: row.family,
      params: row.params,
      developmentScore: row.developmentScore,
      developmentFeasible: row.developmentFeasible,
    }));
    const frozenTop3 = frozenTop5.slice(0, 3);
    const developmentDataHash = developmentHash(prepared, split.development.end);
    const selectionHash = stableHash({
      developmentDataHash,
      cap,
      frozenTop5,
      candidateAudit: candidateAudit.map((row) => ({
        id: row.id,
        rank: row.developmentRank,
        feasible: row.developmentFeasible,
        score: Number(row.developmentScore.toFixed(10)),
      })),
    });
    const sealedOptions = {
      ...options,
      start: split.sealed.start,
      end: split.sealed.end,
    };
    const sealedBenchmark = simulateBuyHoldPrepared(prepared, sealedOptions, "full");
    const top5 = frozenTop5.map((frozen) => {
      const strategy = candidates.find((row) => row.id === frozen.id);
      const development = simulatePrepared(prepared, strategy, developmentOptions, "full");
      const sealed = simulatePrepared(prepared, strategy, sealedOptions, "full");
      const sealedMarket = simulateMarketAsset(prepared.benchmark, prepared, sealedOptions, "summary");
      return {
        ...frozen,
        development,
        sealed: {
          ...sealed,
          benchmark: {
            sameUniverseBuyHold: sealedBenchmark,
            marketAsset: sealedMarket,
          },
          comparison: {
            excessReturn: sealed.metrics.return - sealedBenchmark.metrics.return,
            drawdownImprovement: sealedBenchmark.metrics.drawdown - sealed.metrics.drawdown,
            calmarImprovement: sealed.metrics.calmar - sealedBenchmark.metrics.calmar,
          },
        },
        gates: {
          drawdownCap: sealed.metrics.drawdown <= cap + 1e-9,
          positiveReturn: sealed.metrics.return > 0,
          positiveExcess: sealed.metrics.return > sealedBenchmark.metrics.return,
          betterDrawdown: sealed.metrics.drawdown < sealedBenchmark.metrics.drawdown,
        },
      };
    });
    top5.forEach((row) => {
      row.gates.passed = Object.values(row.gates).every(Boolean);
    });
    const top3 = top5.slice(0, 3);
    return {
      version: VERSION,
      protocol: "仅用开发期排序并冻结Top5；封存样本外结果不参与排名（Top3为兼容视图）",
      targetDrawdown: cap,
      split,
      universe: {
        assetCount: prepared.assets.length,
        benchmarkCode: prepared.benchmark?.code || null,
      },
      developmentDataHash,
      selectionHash,
      candidateCount: candidates.length,
      candidateAudit,
      frozenTop5,
      frozenTop3,
      top5,
      top3,
      primary: top5[0] || null,
      developmentBenchmark,
      sealedBenchmark,
    };
  }

  return Object.freeze({
    VERSION,
    generateCandidates,
    simulatePortfolio,
    runResearch,
    stableHash,
  });
});
