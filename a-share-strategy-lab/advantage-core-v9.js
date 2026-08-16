(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.AdvantageCoreV9 = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const VERSION = "advantage-core-v9.0.0";
  const EPSILON = 1e-10;
  const FAMILIES = Object.freeze([
    Object.freeze({ id: "trend_ladder", name: "趋势分级" }),
    Object.freeze({ id: "trend_reentry", name: "趋势再入" }),
    Object.freeze({ id: "dual_momentum", name: "双动量" }),
    Object.freeze({ id: "vol_target", name: "波动率目标" }),
    Object.freeze({ id: "donchian", name: "唐奇安突破" }),
    Object.freeze({ id: "pullback", name: "上升趋势回调" }),
  ]);
  const FAMILY_IDS = new Set(FAMILIES.map((family) => family.id));
  const STATEFUL_FAMILIES = new Set(["trend_reentry", "donchian", "pullback"]);

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

  function standardDeviation(values) {
    if (values.length < 2) return 0;
    const average = mean(values);
    return Math.sqrt(
      values.reduce((sum, value) => sum + (value - average) ** 2, 0) /
        (values.length - 1),
    );
  }

  function movingAverage(bars, index, period) {
    if (!Number.isInteger(period) || period < 1 || index < period - 1) return NaN;
    let total = 0;
    for (let offset = 0; offset < period; offset += 1) {
      total += Number(bars[index - offset].close) || 0;
    }
    return total / period;
  }

  function momentum(bars, index, period) {
    if (!Number.isInteger(period) || period < 1 || index < period) return NaN;
    return safeReturn(Number(bars[index].close), Number(bars[index - period].close));
  }

  function annualizedVolatility(bars, index, period) {
    if (!Number.isInteger(period) || period < 2 || index < period) return NaN;
    const returns = [];
    for (let cursor = index - period + 1; cursor <= index; cursor += 1) {
      returns.push(safeReturn(Number(bars[cursor].close), Number(bars[cursor - 1].close)));
    }
    return standardDeviation(returns) * Math.sqrt(252);
  }

  function buildIndicatorCache(bars) {
    const closePrefix = [0];
    const returnPrefix = [0];
    const squaredReturnPrefix = [0];
    bars.forEach((bar, index) => {
      closePrefix.push(closePrefix[index] + Number(bar.close));
      const dailyReturn = index ? safeReturn(Number(bar.close), Number(bars[index - 1].close)) : 0;
      returnPrefix.push(returnPrefix[index] + dailyReturn);
      squaredReturnPrefix.push(squaredReturnPrefix[index] + dailyReturn ** 2);
    });
    return {
      movingAverage(index, period) {
        if (!Number.isInteger(period) || period < 1 || index < period - 1) return NaN;
        return (closePrefix[index + 1] - closePrefix[index + 1 - period]) / period;
      },
      volatility(index, period) {
        if (!Number.isInteger(period) || period < 2 || index < period) return NaN;
        const start = index - period + 1;
        const end = index + 1;
        const total = returnPrefix[end] - returnPrefix[start];
        const squaredTotal = squaredReturnPrefix[end] - squaredReturnPrefix[start];
        const variance = Math.max(0, (squaredTotal - total ** 2 / period) / (period - 1));
        return Math.sqrt(variance) * Math.sqrt(252);
      },
    };
  }

  function cachedMovingAverage(bars, index, period, indicators) {
    return indicators ? indicators.movingAverage(index, period) : movingAverage(bars, index, period);
  }

  function cachedVolatility(bars, index, period, indicators) {
    return indicators ? indicators.volatility(index, period) : annualizedVolatility(bars, index, period);
  }

  function rollingExtreme(bars, index, period, key, mode, includeCurrent) {
    const end = includeCurrent ? index : index - 1;
    const start = end - period + 1;
    if (period < 1 || start < 0 || end < start) return NaN;
    let value = mode === "max" ? -Infinity : Infinity;
    for (let cursor = start; cursor <= end; cursor += 1) {
      const candidate = Number(bars[cursor][key]);
      if (!Number.isFinite(candidate)) continue;
      value = mode === "max" ? Math.max(value, candidate) : Math.min(value, candidate);
    }
    return Number.isFinite(value) ? value : NaN;
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
    const sourceAssets = Array.isArray(input.assets)
      ? input.assets
      : Object.entries(input.barsByCode || {}).map(([code, bars]) => ({ code, bars }));
    const assets = sourceAssets
      .map((asset, index) => ({
        code: String(asset.code || `asset-${index + 1}`),
        name: String(asset.name || asset.code || `资产${index + 1}`),
        type: String(asset.type || "stock"),
        bars: normalizeBars(asset.bars || input.barsByCode?.[asset.code] || []),
      }))
      .filter((asset) => asset.bars.length >= 3);
    if (!assets.length) throw new Error("没有可用的真实行情资产");
    return assets;
  }

  function researchDates(assets, start, end) {
    const dates = new Set();
    assets.forEach((asset) =>
      asset.bars.forEach((bar) => {
        if ((!start || bar.date >= start) && (!end || bar.date <= end)) dates.add(bar.date);
      }),
    );
    return [...dates].sort();
  }

  function buildSplit(assets, options) {
    const allDates = researchDates(assets, options.start, options.end);
    if (allDates.length < 4) throw new Error("研究区间共同日期不足");
    const start = options.start || allDates[0];
    const end = options.end || last(allDates);
    let sealedStart = options.sealedStart;
    if (!sealedStart) {
      const ratio = clamp(options.developmentRatio == null ? 0.7 : options.developmentRatio, 0.5, 0.9);
      const cut = Math.min(allDates.length - 2, Math.max(2, Math.floor(allDates.length * ratio)));
      sealedStart = allDates[cut];
    }
    const developmentDates = allDates.filter((date) => date >= start && date < sealedStart);
    const sealedDates = allDates.filter((date) => date >= sealedStart && date <= end);
    if (developmentDates.length < 2 || sealedDates.length < 2) {
      throw new Error("开发期或封存期交易日不足");
    }
    return {
      start,
      end,
      development: [developmentDates[0], last(developmentDates)],
      sealed: [sealedDates[0], last(sealedDates)],
      sealedStart: sealedDates[0],
      developmentDays: developmentDates.length,
      sealedDays: sealedDates.length,
    };
  }

  function exposureTemplates(targetDrawdown) {
    return targetDrawdown <= 0.2
      ? [
          { base: 0, mid: 0.35, max: 0.75 },
          { base: 0.15, mid: 0.45, max: 0.85 },
          { base: 0.25, mid: 0.55, max: 1 },
        ]
      : [
          { base: 0, mid: 0.5, max: 1 },
          { base: 0.2, mid: 0.6, max: 1 },
          { base: 0.35, mid: 0.7, max: 1 },
        ];
  }

  function candidate(family, params) {
    const normalized = { family, params: canonicalize(params) };
    return {
      id: `${family}-${stableHash(normalized)}`,
      family,
      familyName: FAMILIES.find((row) => row.id === family)?.name || family,
      params: normalized.params,
    };
  }

  function generateCandidates(targetDrawdown) {
    const cap = normalizeTargetDrawdown(targetDrawdown);
    const templates = exposureTemplates(cap);
    const rows = [];
    for (const fast of [10, 20]) {
      for (const slow of [60, 120]) {
        if (fast >= slow) continue;
        templates.forEach((weights) => rows.push(candidate("trend_ladder", { fast, slow, ...weights })));
        for (const weights of templates.slice(0, 2)) {
          for (const protection of cap <= 0.2
            ? [
                { trail: 0.09, rebound: 0.04 },
                { trail: 0.12, rebound: 0.06 },
              ]
            : [
                { trail: 0.12, rebound: 0.05 },
                { trail: 0.16, rebound: 0.08 },
              ]) {
            rows.push(candidate("trend_reentry", { fast, slow, ...weights, ...protection }));
          }
        }
      }
    }
    for (const shortLookback of [20, 40]) {
      for (const longLookback of [60, 120]) {
        if (shortLookback >= longLookback) continue;
        templates.slice(0, 2).forEach((weights) =>
          rows.push(candidate("dual_momentum", { shortLookback, longLookback, ...weights })),
        );
      }
    }
    for (const trendPeriod of [60, 120]) {
      for (const volPeriod of [20, 60]) {
        for (const targetVol of cap <= 0.2 ? [0.1, 0.12, 0.15] : [0.15, 0.18, 0.22]) {
          rows.push(
            candidate("vol_target", {
              trendPeriod,
              volPeriod,
              targetVol,
              base: cap <= 0.2 ? 0 : 0.2,
              max: cap <= 0.2 ? 0.9 : 1,
            }),
          );
        }
      }
    }
    for (const [entry, exit] of [
      [20, 10],
      [55, 20],
      [90, 40],
    ]) {
      templates.slice(0, 2).forEach((weights) =>
        rows.push(candidate("donchian", { entry, exit, base: weights.base, max: weights.max })),
      );
    }
    for (const trendPeriod of [60, 120, 200]) {
      for (const pullback of [0.06, 0.1]) {
        const weights = templates[1];
        rows.push(
          candidate("pullback", {
            trendPeriod,
            pullback,
            rebound: cap <= 0.2 ? 0.04 : 0.06,
            ...weights,
          }),
        );
      }
    }
    return rows;
  }

  function normalizeCandidates(rows, targetDrawdown) {
    const source = rows?.length ? rows : generateCandidates(targetDrawdown);
    const seen = new Set();
    const output = [];
    source.forEach((row) => {
      if (!row || !FAMILY_IDS.has(row.family)) return;
      const normalized = candidate(row.family, row.params || {});
      normalized.id = String(row.id || normalized.id);
      normalized.familyName = String(
        row.familyName || FAMILIES.find((family) => family.id === row.family)?.name || row.family,
      );
      if (!seen.has(normalized.id)) {
        seen.add(normalized.id);
        output.push(normalized);
      }
    });
    if (!output.length) throw new Error("没有有效候选策略");
    return output;
  }

  function defaultStrategyState() {
    return {
      on: false,
      stopped: false,
      recovering: false,
      peak: 0,
      low: Infinity,
      active: false,
    };
  }

  function targetExposure(bars, index, strategy, state, indicators) {
    if (index < 0 || !bars[index]) return 0;
    state = state || defaultStrategyState();
    const params = strategy.params || {};
    const base = clamp(params.base == null ? 0 : params.base, 0, 1);
    const mid = clamp(params.mid == null ? (base + 1) / 2 : params.mid, base, 1);
    const maximum = clamp(params.max == null ? 1 : params.max, mid, 1);
    const close = Number(bars[index].close);

    if (strategy.family === "trend_ladder") {
      const fast = cachedMovingAverage(bars, index, Number(params.fast || 20), indicators);
      const slow = cachedMovingAverage(bars, index, Number(params.slow || 120), indicators);
      if (!Number.isFinite(fast) || !Number.isFinite(slow)) return base;
      if (close > fast && fast > slow) return maximum;
      if (close > slow) return mid;
      return base;
    }

    if (strategy.family === "trend_reentry") {
      const fast = cachedMovingAverage(bars, index, Number(params.fast || 20), indicators);
      const slow = cachedMovingAverage(bars, index, Number(params.slow || 120), indicators);
      if (!Number.isFinite(fast) || !Number.isFinite(slow)) return base;
      if (!state.peak) state.peak = close;
      state.peak = Math.max(state.peak, close);
      const trail = clamp(params.trail == null ? 0.12 : params.trail, 0.01, 0.5);
      const rebound = clamp(params.rebound == null ? 0.05 : params.rebound, 0.005, 0.5);
      if (!state.stopped && (close < slow || safeReturn(close, state.peak) <= -trail)) {
        state.stopped = true;
        state.recovering = false;
        state.low = close;
      }
      if (state.stopped) {
        state.low = Math.min(state.low, close);
        if (close > fast && safeReturn(close, state.low) >= rebound) {
          state.stopped = false;
          state.recovering = true;
          state.peak = close;
        } else {
          return base;
        }
      }
      if (state.recovering) {
        if (close > fast && fast > slow) {
          state.recovering = false;
          return maximum;
        }
        return mid;
      }
      return close > fast && fast > slow ? maximum : close > slow ? mid : base;
    }

    if (strategy.family === "dual_momentum") {
      const shortMomentum = momentum(bars, index, Number(params.shortLookback || 20));
      const longMomentum = momentum(bars, index, Number(params.longLookback || 120));
      if (!Number.isFinite(shortMomentum) || !Number.isFinite(longMomentum)) return base;
      if (shortMomentum > 0 && longMomentum > 0) return maximum;
      if (shortMomentum > 0 || longMomentum > 0) return mid;
      return base;
    }

    if (strategy.family === "vol_target") {
      const trendPeriod = Number(params.trendPeriod || 120);
      const trend = cachedMovingAverage(bars, index, trendPeriod, indicators);
      const volatility = cachedVolatility(
        bars,
        index,
        Number(params.volPeriod || 20),
        indicators,
      );
      if (!Number.isFinite(trend) || !Number.isFinite(volatility) || close <= trend) return base;
      const target = clamp(Number(params.targetVol || 0.15) / Math.max(0.04, volatility), base, maximum);
      return clamp(Math.round(target * 20) / 20, base, maximum);
    }

    if (strategy.family === "donchian") {
      const upper = rollingExtreme(
        bars,
        index,
        Number(params.entry || 55),
        "high",
        "max",
        false,
      );
      const lower = rollingExtreme(
        bars,
        index,
        Number(params.exit || 20),
        "low",
        "min",
        false,
      );
      if (Number.isFinite(upper) && close > upper) state.on = true;
      if (state.on && Number.isFinite(lower) && close < lower) state.on = false;
      return state.on ? maximum : base;
    }

    if (strategy.family === "pullback") {
      const trend = cachedMovingAverage(
        bars,
        index,
        Number(params.trendPeriod || 120),
        indicators,
      );
      const peak = rollingExtreme(bars, index, 20, "close", "max", true);
      if (!Number.isFinite(trend) || !Number.isFinite(peak)) return base;
      if (close < trend) state.active = false;
      if (close > trend && safeReturn(close, peak) <= -Math.abs(Number(params.pullback || 0.08))) {
        state.active = true;
        state.low = close;
      }
      if (state.active) {
        state.low = Math.min(state.low, close);
        if (safeReturn(close, state.low) >= Number(params.rebound || 0.04)) return maximum;
        return mid;
      }
      return close > trend ? mid : base;
    }

    return base;
  }

  function curveStats(rows, key, initialValue) {
    if (!rows.length) {
      return {
        return: 0,
        cagr: 0,
        drawdown: 0,
        volatility: 0,
        sharpe: 0,
        calmar: 0,
      };
    }
    let previous = initialValue;
    let peak = initialValue;
    let maxDrawdown = 0;
    const returns = [];
    for (const row of rows) {
      const value = Number(row[key]);
      if (!(value >= 0)) continue;
      returns.push(safeReturn(value, previous));
      previous = value;
      peak = Math.max(peak, value);
      maxDrawdown = Math.min(maxDrawdown, safeReturn(value, peak));
    }
    const ending = Number(last(rows)?.[key]) || initialValue;
    const totalReturn = safeReturn(ending, initialValue);
    const days = Math.max(
      1,
      (new Date(`${last(rows).date}T00:00:00Z`) - new Date(`${rows[0].date}T00:00:00Z`)) /
        864e5 +
        1,
    );
    const years = Math.max(days / 365.25, 1 / 252);
    const cagr = Math.pow(Math.max(EPSILON, 1 + totalReturn), 1 / years) - 1;
    const volatility = standardDeviation(returns) * Math.sqrt(252);
    const sharpe = volatility ? (mean(returns) * 252) / volatility : 0;
    const calmar = maxDrawdown < 0 ? cagr / Math.abs(maxDrawdown) : cagr > 0 ? 9.99 : 0;
    return { return: totalReturn, cagr, drawdown: maxDrawdown, volatility, sharpe, calmar };
  }

  function stampRate(date, options) {
    const multiplier = Number(options.costMultiplier || 1);
    if (Number.isFinite(Number(options.stampBp))) return (Number(options.stampBp) / 10000) * multiplier;
    return (date < "2023-08-28" ? 10 : 5) / 10000 * multiplier;
  }

  function simulateAsset(assetInput, strategyInput, optionsInput) {
    const inputBars =
      optionsInput?._barsNormalized === true
        ? assetInput.bars || []
        : normalizeBars(assetInput.bars || []);
    const asset = {
      code: String(assetInput.code || "asset"),
      name: String(assetInput.name || assetInput.code || "asset"),
      type: String(assetInput.type || "stock"),
      bars: inputBars,
    };
    const strategy = normalizeCandidates([strategyInput], 0.3)[0];
    const options = {
      capital: 1_000_000,
      feeBp: 3,
      slipBp: 5,
      costMultiplier: 1,
      executionDelayDays: 0,
      minTargetChange: 0.005,
      ...optionsInput,
    };
    const capital = Number(options.capital) > 0 ? Number(options.capital) : 1_000_000;
    const feeRate = (Number(options.feeBp || 0) / 10000) * Number(options.costMultiplier || 1);
    const slipRate = (Number(options.slipBp || 0) / 10000) * Number(options.costMultiplier || 1);
    const executionDelayDays = Math.max(0, Math.floor(Number(options.executionDelayDays || 0)));
    const first = asset.bars.findIndex((bar) => !options.start || bar.date >= options.start);
    const startIndex = first < 0 ? asset.bars.length : first;
    let endIndex = startIndex - 1;
    for (let index = startIndex; index < asset.bars.length; index += 1) {
      if (options.end && asset.bars[index].date > options.end) break;
      endIndex = index;
    }

    let cash = capital;
    let shares = 0;
    let benchmarkCash = capital;
    let benchmarkShares = 0;
    let benchmarkEntered = false;
    let lastProcessedSignal = -1;
    let lastExecutedTarget = null;
    let currentTarget = 0;
    let turnoverValue = 0;
    let exposureSum = 0;
    const state = defaultStrategyState();
    const trades = [];
    const equity = [];
    const costs = { commission: 0, stamp: 0, slippage: 0, total: 0 };
    const benchmarkCosts = { commission: 0, slippage: 0, total: 0 };
    const indicators = buildIndicatorCache(asset.bars);

    function processSignalsThrough(signalIndex) {
      if (!STATEFUL_FAMILIES.has(strategy.family)) {
        currentTarget = targetExposure(asset.bars, signalIndex, strategy, state, indicators);
        lastProcessedSignal = signalIndex;
        return currentTarget;
      }
      for (let index = lastProcessedSignal + 1; index <= signalIndex; index += 1) {
        currentTarget = targetExposure(asset.bars, index, strategy, state, indicators);
      }
      lastProcessedSignal = Math.max(lastProcessedSignal, signalIndex);
      return currentTarget;
    }

    for (let index = startIndex; index <= endIndex; index += 1) {
      const bar = asset.bars[index];
      const signalIndex = index - 1 - executionDelayDays;
      const target = signalIndex >= 0 ? processSignalsThrough(signalIndex) : 0;

      if (!benchmarkEntered) {
        const price = bar.open * (1 + slipRate);
        const quantity = benchmarkCash / (price * (1 + feeRate));
        if (quantity > 0) {
          const value = quantity * price;
          const commission = value * feeRate;
          benchmarkCash -= value + commission;
          benchmarkShares += quantity;
          benchmarkCosts.commission += commission;
          benchmarkCosts.slippage += quantity * bar.open * slipRate;
          benchmarkCosts.total = benchmarkCosts.commission + benchmarkCosts.slippage;
          benchmarkEntered = true;
        }
      }

      if (
        lastExecutedTarget === null ||
        Math.abs(target - lastExecutedTarget) >= Number(options.minTargetChange || 0)
      ) {
        const openEquity = cash + shares * bar.open;
        const desiredShares = (openEquity * clamp(target, 0, 1)) / bar.open;
        const difference = desiredShares - shares;
        if (difference < -EPSILON) {
          const quantity = Math.min(shares, -difference);
          const price = bar.open * (1 - slipRate);
          const value = quantity * price;
          const commission = value * feeRate;
          const stamp = value * stampRate(bar.date, options);
          cash += value - commission - stamp;
          shares -= quantity;
          turnoverValue += value;
          costs.commission += commission;
          costs.stamp += stamp;
          costs.slippage += quantity * bar.open * slipRate;
          trades.push({
            signalDate: asset.bars[signalIndex]?.date || null,
            tradeDate: bar.date,
            action: "sell",
            quantity,
            price,
            target,
            fee: commission,
            stamp,
          });
        } else if (difference > EPSILON) {
          const price = bar.open * (1 + slipRate);
          const affordable = cash / (price * (1 + feeRate));
          const quantity = Math.min(difference, Math.max(0, affordable));
          if (quantity > EPSILON) {
            const value = quantity * price;
            const commission = value * feeRate;
            cash -= value + commission;
            shares += quantity;
            turnoverValue += value;
            costs.commission += commission;
            costs.slippage += quantity * bar.open * slipRate;
            trades.push({
              signalDate: asset.bars[signalIndex]?.date || null,
              tradeDate: bar.date,
              action: "buy",
              quantity,
              price,
              target,
              fee: commission,
              stamp: 0,
            });
          }
        }
        lastExecutedTarget = target;
      }

      const strategyValue = cash + shares * bar.close;
      const benchmarkValue = benchmarkCash + benchmarkShares * bar.close;
      const exposure = strategyValue > 0 ? (shares * bar.close) / strategyValue : 0;
      exposureSum += exposure;
      equity.push({
        date: bar.date,
        equity: strategyValue,
        benchmark: benchmarkValue,
        cash,
        shares,
        target,
        exposure,
      });
    }

    costs.total = costs.commission + costs.stamp + costs.slippage;
    const strategyStats = curveStats(equity, "equity", capital);
    const benchmarkStats = curveStats(equity, "benchmark", capital);
    return {
      asset: { code: asset.code, name: asset.name, type: asset.type },
      strategy,
      period: [options.start || asset.bars[startIndex]?.date || null, options.end || asset.bars[endIndex]?.date || null],
      equity,
      trades,
      costs,
      benchmarkCosts,
      strategyStats,
      benchmarkStats,
      excessReturn: strategyStats.return - benchmarkStats.return,
      excessCagr: strategyStats.cagr - benchmarkStats.cagr,
      drawdownImprovement:
        Math.abs(benchmarkStats.drawdown) - Math.abs(strategyStats.drawdown),
      turnover: turnoverValue / capital,
      averageExposure: equity.length ? exposureSum / equity.length : 0,
    };
  }

  function aggregateRuns(runs, capital) {
    if (!runs.length) {
      const empty = curveStats([], "equity", 1);
      return {
        equity: [],
        strategy: empty,
        benchmark: empty,
        excessReturn: 0,
        excessCagr: 0,
        drawdownImprovement: 0,
        assetWinRate: 0,
        averageTurnover: 0,
        averageExposure: 0,
        averageCostRate: 0,
      };
    }
    const dates = [...new Set(runs.flatMap((run) => run.equity.map((row) => row.date)))].sort();
    const maps = runs.map(
      (run) =>
        new Map(
          run.equity.map((row) => [
            row.date,
            { equity: row.equity / capital, benchmark: row.benchmark / capital },
          ]),
        ),
    );
    const carried = runs.map(() => ({ equity: 1, benchmark: 1 }));
    const equity = dates.map((date) => {
      maps.forEach((map, index) => {
        if (map.has(date)) carried[index] = map.get(date);
      });
      return {
        date,
        equity: mean(carried.map((row) => row.equity)),
        benchmark: mean(carried.map((row) => row.benchmark)),
      };
    });
    const strategy = curveStats(equity, "equity", 1);
    const benchmark = curveStats(equity, "benchmark", 1);
    return {
      equity,
      strategy,
      benchmark,
      excessReturn: strategy.return - benchmark.return,
      excessCagr: strategy.cagr - benchmark.cagr,
      drawdownImprovement: Math.abs(benchmark.drawdown) - Math.abs(strategy.drawdown),
      assetWinRate: runs.filter((run) => run.excessReturn > 0).length / runs.length,
      averageTurnover: mean(runs.map((run) => run.turnover)),
      averageExposure: mean(runs.map((run) => run.averageExposure)),
      averageCostRate: mean(runs.map((run) => run.costs.total / capital)),
    };
  }

  function panelOptions(options, start, end, overrides) {
    return {
      capital: options.capital,
      feeBp: options.feeBp,
      slipBp: options.slipBp,
      stampBp: options.stampBp,
      costMultiplier: options.costMultiplier,
      executionDelayDays: options.executionDelayDays,
      minTargetChange: options.minTargetChange,
      _barsNormalized: true,
      start,
      end,
      ...overrides,
    };
  }

  function evaluatePanel(assets, strategy, options, start, end, overrides) {
    const simulationOptions = panelOptions(options, start, end, overrides || {});
    const runs = assets.map((asset) => simulateAsset(asset, strategy, simulationOptions));
    return {
      aggregate: aggregateRuns(runs, options.capital),
      perStock: runs.map((run) => ({
        code: run.asset.code,
        name: run.asset.name,
        strategy: run.strategyStats,
        benchmark: run.benchmarkStats,
        excessReturn: run.excessReturn,
        excessCagr: run.excessCagr,
        drawdownImprovement: run.drawdownImprovement,
        turnover: run.turnover,
        averageExposure: run.averageExposure,
        costs: run.costs,
        tradeCount: run.trades.length,
      })),
      runs,
    };
  }

  function developmentScore(result) {
    const aggregate = result.aggregate;
    return (
      aggregate.excessCagr * 100 +
      aggregate.strategy.calmar * 2 +
      aggregate.drawdownImprovement * 25 +
      (aggregate.assetWinRate - 0.5) * 10 -
      aggregate.averageTurnover * 0.1
    );
  }

  function compactPanel(result) {
    return { aggregate: result.aggregate, perStock: result.perStock };
  }

  function worstStockDrawdown(result) {
    return result.perStock.length
      ? Math.max(...result.perStock.map((row) => Math.abs(row.strategy.drawdown)))
      : 0;
  }

  function candidateAuditRow(strategy, development, targetDrawdown) {
    const worstDevelopmentStockDrawdown = worstStockDrawdown(development);
    // The requested cap is a portfolio-level risk budget. Per-stock violations remain
    // visible as a stricter robustness gate, but do not make a diversified candidate
    // impossible to freeze before OOS is opened.
    const feasible =
      Math.abs(development.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON;
    return {
      id: strategy.id,
      family: strategy.family,
      familyName: strategy.familyName,
      params: strategy.params,
      hash: stableHash(strategy),
      development: {
        aggregate: development.aggregate,
        perStock: development.perStock,
      },
      developmentScore: developmentScore(development),
      developmentDrawdownCap: targetDrawdown,
      worstDevelopmentStockDrawdown,
      developmentFeasible: feasible,
    };
  }

  function rankDevelopment(left, right) {
    if (left.developmentFeasible !== right.developmentFeasible) {
      return left.developmentFeasible ? -1 : 1;
    }
    if (Math.abs(left.developmentScore - right.developmentScore) > EPSILON) {
      return right.developmentScore - left.developmentScore;
    }
    return left.id.localeCompare(right.id);
  }

  function validationGates(development, sealed, costStress, delayStress, targetDrawdown, options) {
    const gates = {
      developmentDrawdownCap:
        Math.abs(development.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON,
      developmentEveryStockDrawdownCap:
        worstStockDrawdown(development) <= targetDrawdown + EPSILON,
      sealedPositive: sealed.aggregate.strategy.return > 0,
      sealedExcessPositive: sealed.aggregate.excessReturn > 0,
      sealedDrawdownCap:
        Math.abs(sealed.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON,
      sealedEveryStockDrawdownCap: worstStockDrawdown(sealed) <= targetDrawdown + EPSILON,
      sealedDrawdownImproved:
        Math.abs(sealed.aggregate.strategy.drawdown) <
        Math.abs(sealed.aggregate.benchmark.drawdown),
      sealedCalmarImproved:
        sealed.aggregate.strategy.calmar >= sealed.aggregate.benchmark.calmar,
      crossStockWinRate:
        sealed.aggregate.assetWinRate >= Number(options.assetWinRateGate == null ? 0.5 : options.assetWinRateGate),
      doubleCostRobust:
        costStress.aggregate.excessReturn > 0 &&
        Math.abs(costStress.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON,
      oneDayDelayRobust:
        delayStress.aggregate.excessReturn > 0 &&
        Math.abs(delayStress.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON,
    };
    return { ...gates, passed: Object.values(gates).every(Boolean) };
  }

  function runResearch(input) {
    const assets = normalizeAssets(input || {});
    const rawOptions = input.options || {};
    const targetDrawdown = normalizeTargetDrawdown(
      rawOptions.targetDrawdown ?? rawOptions.maxDrawdown ?? 0.3,
    );
    const options = {
      capital: Number(rawOptions.capital) > 0 ? Number(rawOptions.capital) : 1_000_000,
      feeBp: Number(rawOptions.feeBp == null ? 3 : rawOptions.feeBp),
      slipBp: Number(rawOptions.slipBp == null ? 5 : rawOptions.slipBp),
      stampBp: rawOptions.stampBp,
      costMultiplier: Number(rawOptions.costMultiplier || 1),
      executionDelayDays: Math.max(0, Math.floor(Number(rawOptions.executionDelayDays || 0))),
      minTargetChange: Number(rawOptions.minTargetChange == null ? 0.005 : rawOptions.minTargetChange),
      assetWinRateGate: Number(rawOptions.assetWinRateGate == null ? 0.5 : rawOptions.assetWinRateGate),
      start: rawOptions.start,
      end: rawOptions.end,
      sealedStart: rawOptions.sealedStart,
      developmentRatio: rawOptions.developmentRatio,
      targetDrawdown,
    };
    const split = buildSplit(assets, options);
    const candidates = normalizeCandidates(input.candidates, targetDrawdown);

    // Selection is completed using development data only. No sealed result exists yet.
    const candidateAudit = candidates
      .map((strategy) => {
        const development = evaluatePanel(
          assets,
          strategy,
          options,
          split.development[0],
          split.development[1],
        );
        return candidateAuditRow(strategy, development, targetDrawdown);
      })
      .sort(rankDevelopment);

    const familyWinners = [];
    const usedFamilies = new Set();
    for (const row of candidateAudit) {
      if (!row.developmentFeasible || usedFamilies.has(row.family)) continue;
      usedFamilies.add(row.family);
      familyWinners.push(row);
    }
    familyWinners.sort(rankDevelopment);
    const frozen = familyWinners.slice(0, 3).map((row, index) => ({
      rank: index + 1,
      id: row.id,
      family: row.family,
      familyName: row.familyName,
      params: row.params,
      hash: row.hash,
      development: row.development,
      developmentScore: row.developmentScore,
    }));
    const developmentDataHash = stableHash(
      assets.map((asset) => ({
        code: asset.code,
        bars: asset.bars
          // Warm-up history also affects the first tradable signal and belongs in the freeze hash.
          .filter((bar) => bar.date <= split.development[1])
          .map((bar) => [bar.date, bar.open, bar.close, bar.high, bar.low, bar.volume]),
      })),
    );
    const selectionHash = stableHash({
      version: VERSION,
      targetDrawdown,
      split,
      developmentDataHash,
      candidates: candidateAudit.map((row) => ({ id: row.id, hash: row.hash })),
      frozen: frozen.map((row) => ({ rank: row.rank, id: row.id, hash: row.hash })),
    });

    // Only the already-frozen, development-ranked strategies are evaluated on sealed data.
    const top3 = frozen.map((selection) => {
      const strategy = {
        id: selection.id,
        family: selection.family,
        familyName: selection.familyName,
        params: selection.params,
      };
      const sealed = evaluatePanel(
        assets,
        strategy,
        options,
        split.sealed[0],
        split.sealed[1],
      );
      const doubleCost = evaluatePanel(
        assets,
        strategy,
        options,
        split.sealed[0],
        split.sealed[1],
        { costMultiplier: options.costMultiplier * 2 },
      );
      const oneDayDelay = evaluatePanel(
        assets,
        strategy,
        options,
        split.sealed[0],
        split.sealed[1],
        { executionDelayDays: options.executionDelayDays + 1 },
      );
      return {
        ...selection,
        sealed: compactPanel(sealed),
        sensitivity: {
          doubleCost: compactPanel(doubleCost),
          oneDayDelay: compactPanel(oneDayDelay),
        },
        gates: validationGates(
          selection.development,
          sealed,
          doubleCost,
          oneDayDelay,
          targetDrawdown,
          options,
        ),
      };
    });

    return {
      version: VERSION,
      protocol: {
        objective: `跨股票统一参数，在开发期等权研究组合最大回撤不超过${Math.round(targetDrawdown * 100)}%的约束下选择策略`,
        selectionData: "development-only",
        sealedPolicy: "封存OOS只验收，不参与参数、家族或Top3排序",
        execution: "T日收盘信号，T+1开盘成交；延迟敏感性为再延后1个交易日",
        universe: "输入股票池须在打开封存期前冻结",
      },
      runId: `adv-v9-${selectionHash}`,
      selectionHash,
      developmentDataHash,
      targetDrawdown,
      split,
      assetCount: assets.length,
      candidateCount: candidates.length,
      feasibleCandidateCount: candidateAudit.filter((row) => row.developmentFeasible).length,
      candidateAudit,
      frozenTop3: frozen.map((row) => ({
        rank: row.rank,
        id: row.id,
        family: row.family,
        familyName: row.familyName,
        params: row.params,
        hash: row.hash,
        developmentScore: row.developmentScore,
      })),
      top3,
      primary: top3[0] || null,
      gates: {
        hasThreeDistinctFamilies: top3.length === 3 && new Set(top3.map((row) => row.family)).size === 3,
        allDevelopmentDrawdownsWithinCap: top3.every(
          (row) =>
            Math.abs(row.development.aggregate.strategy.drawdown) <= targetDrawdown + EPSILON,
        ),
        anySealedQualified: top3.some((row) => row.gates.passed),
      },
    };
  }

  function runDrawdownTiers(input, tiers) {
    const requested = tiers?.length ? tiers : [20, 30];
    return Object.fromEntries(
      requested.map((tier) => {
        const normalized = normalizeTargetDrawdown(tier);
        const key = `${Math.round(normalized * 100)}`;
        return [
          key,
          runResearch({
            ...(input || {}),
            options: { ...(input?.options || {}), targetDrawdown: normalized },
          }),
        ];
      }),
    );
  }

  return {
    VERSION,
    FAMILIES,
    stableHash,
    normalizeTargetDrawdown,
    buildSplit,
    generateCandidates,
    targetExposure,
    curveStats,
    simulateAsset,
    aggregateRuns,
    evaluatePanel,
    runResearch,
    runDrawdownTiers,
  };
});
