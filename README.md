# A股短线热点筛选器 (A-Share Hot Screener)

> **Stage 1 — 结构化、客观、可复现的短线热点股票筛选与打分系统**
>
> v2.1 — 四轴评分 + HT8/9/10 市场确认 + 观察时机 + 动态 Universe + 评估框架 + 每日自动运行

## 项目概述

本项目实现了一套 A 股短线热点股票的第一阶段（Stage 1）客观筛选系统。系统通过 **动态候选池构建 → 硬性过滤 → 多维度特征计算 → 四轴评分 → 阈值判定 → 观察时机评估 → Discord 自动推送** 的完整流水线，输出结构化的筛选结果。

**核心原则：**
- 仅使用 **Tushare Pro** 作为数据源（6000 积分 / 600元/年），保证数据口径统一
- 严格遵守 **as-of / point-in-time** 口径：仅使用 `run_date` 当天或之前已知数据
- 第一阶段只做结构化、客观、可脚本化、可复现的筛选与打分
- 禁止将热点催化语义判断、题材阶段主观判断等第二阶段内容混入

**项目规模：**
- 生产代码：~15,800 行 / 46 个模块
- 测试代码：~14,150 行 / 34 个测试文件 / 1010 个测试用例
- 测试通过率：100%（~7s）

---

## 目录结构

```
a_share_hot_screener/                    # ~15,800 行生产代码
├── clients/
│   └── tushare_client.py                # 1,078行 — 18个 API + TokenBucket 限流 + 重试 + prefetch_risk/flow
├── scorers/
│   ├── hot_theme.py                     # 417行 — HT1~HT7 热点题材评分 (+ HT8/9/10 条件集成)
│   ├── trend_flow.py                    # 207行 — TF1~TF7 趋势资金流评分
│   ├── liquidity_execution.py           # 229行 — LE1~LE5 流动性执行评分
│   └── risk_control.py                  # 281行 — RC1~RC10 风险控制评分
├── evaluation/                          # Phase 1 评估框架
│   ├── label_generator.py               # 398行 — 未来收益标签 (T+1/MFE/MAE/涨停/超指数)
│   ├── harness.py                       # 530行 — 评估引擎 (分组+单调性+分离度+消融)
│   └── report.py                        # 216行 — 文本/CSV 报告
├── pipeline.py                          # 718行 — 薄编排层（Step 0~11）
├── stock_processor.py                   # 386行 — 单股处理（Step 3~6.11）
├── scoring_aggregator.py                # 199行 — 四轴评分聚合 + total_score
├── stage1_judge.py                      # 299行 — Stage1 通过判定 + crowding cap
├── scoring.py                           # 623行 — ScoreItem/AxisScore/ScoringPool 框架
├── event_layer.py                       # 908行 — 涨停池(并发)/强势股/龙虎榜/行业热度(三策略级联)/概念热度
├── context_scores.py                    # 511行 — HT8/HT9/HT10 市场确认度 (experimental)
├── setup_timing.py                      # 1,239行 — 观察时机评估 5维评分 (experimental)
├── universe_builder.py                  # 579行 — 动态候选池构建 (CSI指数+涨停池+龙虎榜+成交额Top+热门板块)
├── daily_runner.py                      # 797行 — 每日自动运行器 v2 (17:30策略+Universe+Discord v2)
├── discord_notifier.py                  # 232行 — Discord Bot 通知 (Embed+附件+DM fallback)
├── models.py                            # 565行 — HotStockDetail(88+字段)/Summary/Rejected/Metadata
├── price_features.py                    # 526行 — 日线派生特征 (含上影线双指标)
├── config.py                            # 123行 — HotScreenerConfig
├── cli.py                               # 430行 — CLI 入口 (dotenv + batch + purge-cache)
├── output.py                            # 320行 — CSV/JSON 输出
├── validation.py                        # 306行 — SpotUniverse + StockValidator
├── trade_calendar.py                    # 364行 — 交易日历 + last_n_trade_dates
├── hard_filters.py                      # 347行 — H1~H9 硬筛
├── flags.py                             # 226行 — structured flags（26+ keys）
├── trend_compare.py                     # 396行 — 时序连续性对比
├── cache.py                             # 254行 — 文件 JSON 缓存（版本化 + purge_stale + stats）
├── batch_runner.py                      # 625行 — 批量运行编排（自动分批/合并/断点续跑/全局pool）
├── sector_rotation.py                   # 296行 — 板块轮动信号（A4，集成到 pipeline + HT7）
├── indicators.py                        # 40行 — upper_wick_ratio + upper_reversal_ratio
├── moneyflow.py                         # 101行 — 个股资金流向
├── holdertrade.py                       # 99行 — 股东增减持
├── margin.py                            # 123行 — 融资融券
├── shareholder_reduction.py             # 146行 — 股东减持检测
├── pledge_ratio.py                      # 86行 — 质押比例
├── restricted_unlock.py                 # 127行 — 限售解禁
├── ticker_mapping.py                    # 89行 — 代码格式转换
├── limit_rules.py                       # 54行 — 涨跌停幅度推断
├── date_utils.py                        # 90行
├── stock_codes.py                       # 90行
├── logger.py                            # 84行
├── __main__.py                          # CLI 入口
└── tests/                               # ~14,150 行 / 34 个测试文件 / 1010 tests
```

---

## 核心架构

### Pipeline 流程

```
Step 0:     stock_codes 解析（支持逗号分隔、空格分隔、文件输入）
Step 0.5:   交易日历加载 + trade_date_used 确定
Step 1:     Tushare daily_basic 全市场表加载（SpotUniverse）
Step 2:     股票池校验（StockValidator）→ 同步收集 spot 字段
Step 2.5:   事件层批量加载（EventLayerLoader）→ 涨停池/强势股/龙虎榜/行业热度/概念热度
Step 2.7:   风控数据预加载（pledge/float/holdnum，并发 5 线程）
Step 2.8:   资金流向数据预加载（moneyflow/holdertrade/margin，并发 5 线程）
Step 2.9:   板块轮动分析 → sector_heat.csv + industry→momentum 查找表
Step 3~6:   并发个股处理 → stock_processor.process_single_stock()
            ├─ 日线行情获取（近 60+10 日）
            ├─ price_features 特征计算（22+ 项）
            ├─ 硬性过滤（H1~H9）
            ├─ 事件层特征填充
            ├─ 风控指标计算（质押/解禁/股东人数/减持/资金流向/融资融券）
            └─ 板块轮动信号填充
Step 6.15:  Context Scores 计算（HT8/HT9/HT10，experimental）
Step 7:     四轴评分 → scoring_aggregator.apply_four_axis_scores()
Step 7.5:   structured flags 生成
Step 8:     pass_stage1 判定 → stage1_judge（含 crowding cap）
            ├─ tradeable: 可交易候选
            └─ watch_only: 一字板/高位不可交易，仅观察
Step 8.5:   Setup Timing 观察时机评估（仅 tradeable，experimental）
Step 9:     时序连续性对比（与前次运行结果比较趋势变化）
Step 10:    输出（summary/detail/rejected/metadata + sector_heat + setup_timing CSV）
Step 11:    保存基准 pool（可选）
```

### 四轴评分体系

四维度加权评分，每个轴独立计算后汇总为 `total_score`：

#### HT 轴 — hot_theme（权重 40%）

| 子项 | 指标 | 权重 | 类型 | 说明 |
|------|------|------|------|------|
| HT1 | 近5日收益百分位 | 8 | 下限型 | L=60/T=80/H=95 |
| HT2 | 近10日收益百分位 | 6 | 下限型 | L=60/T=80/H=95 |
| HT3 | 近10日大涨天数 | 8 | 离散 | 0→0 / 1→0.40 / 2→0.70 / 3→0.90 / ≥4→1.0 |
| HT4 | 连续上涨天数 | 6 | 离散 | 0→0 ~ ≥5→1.0 |
| HT5 | 行业热度百分位 | 7 | 下限型 | 三策略级联（L3→L2→L1），覆盖率 100% |
| HT6 | 概念热度百分位 | 7 | 下限型 | 取所属概念中最强者 |
| HT7 | 板块轮动动量 | 5 | 离散 | rotate_in→1.0 / rotate_out→0.15 |
| *HT8* | *市场确认度* | *6* | *5级* | *experimental — 涨停+龙虎榜+概念+放量组合* |
| *HT9* | *板块扩散度* | *5* | *下限型* | *experimental — 板块内涨幅>5%占比* |
| *HT10* | *板块内辨识度* | *6* | *加权* | *experimental — 排名+成交额份额+首板* |

> HT8/9/10 默认仅计算输出，不纳入总分（`use_context_scores_in_total=False`），回测验证后正式纳入。

#### TF 轴 — trend_flow（权重 30%）

| 子项 | 指标 | 权重 | 类型 |
|------|------|------|------|
| TF1 | 20日区间收盘位置 | 7 | 下限型 |
| TF2 | 均线多头排列 | 6 | 离散 |
| TF3 | 量比 | 6 | 下限型 |
| TF4 | CLV | 5 | 下限型 |
| TF5 | 量能比率 | 6 | 下限型 |
| TF6 | 主力净流入占比 | 6 | 下限型 |
| TF7 | 融资净买入占比 | 4 | 下限型 (两融标的) |

#### LE 轴 — liquidity_execution（权重 20%）

| 子项 | 指标 | 权重 |
|------|------|------|
| LE1 | 成交额分档 | 9 |
| LE2 | 换手率 | 6 |
| LE3 | 流通市值分档 | 4 |
| LE4 | 龙虎榜次数 | 1 |

> LE2 优先使用 `turnover_rate_f`/`turnover_rate`，proxy 推算值上限 0.85。

#### RC 轴 — risk_control（权重 10%）

| 子项 | 指标 | 权重 | 类型 |
|------|------|------|------|
| RC1 | 一字板次数 | 4 | 离散 |
| RC2 | 振幅/涨跌停 | 3 | 上限型 |
| RC3 | MA10偏离 | 4 | 上限型 |
| RC4 | 上影线次数 | 2 | 离散 |
| RC5 | 3日涨幅/涨跌停 | 2 | 上限型 |
| RC6 | 质押比例 | 2 | 上限型 |
| RC7 | 限售解禁 | 2 | 上限型 |
| RC8 | 股东人数增幅 | 1 | 上限型 |
| RC9 | 股东净减持 | 3 | 上限型 |
| RC10 | 融券余额变化率 | 1 | 上限型 (两融标的) |

#### pass_stage1 阈值（AND 关系）

```
total_score        ≥ 0.68
hot_theme          ≥ 0.65
trend_flow         ≥ 0.60
liquidity_execution ≥ 0.55
risk_control       ≥ 0.40
```

#### 高位拥挤 Cap

在阈值判定前，对高位拥挤股施加 total_score 上限：
- 一字涨停 → cap 0.67
- `risk_control_score < 0.30` → cap 0.66
- MA10偏离>25% 且上影线≥2 → cap 0.65

#### Tradeable vs Watch-Only

- **tradeable**: 正常通过 pass_stage1 → 可交易候选
- **watch_only**: 一字板/高位不可交易 → 仅观察（`pass_stage1_watch=True`）

---

## 动态 Universe 构建

`universe_builder.py` 自动构建候选股票池（`--use-universe-builder`）：

| 来源 | 说明 |
|------|------|
| CSI 300 + 500 + 1000 成分 | 底仓（权重指数成分） |
| 涨停池 | 当日涨停 + 连板股 (`limit_list_d`) |
| 龙虎榜 | 当日上榜股 (`top_list`) |
| 成交额 Top 200 | 当日成交额前 200 (`daily_basic`) |
| 热门板块成分 | 近5日涨幅 Top 行业/概念的成分股 |

去重后通常产出 800~1200 只候选。

---

## 观察时机评估 (Setup Timing)

`setup_timing.py` 对 tradeable 候选进行 5 维时机评估（`--enable-setup-timing`，experimental）：

| 维度 | 权重 | 核心指标 |
|------|------|----------|
| 趋势状态 | 0.22 | MA排列 + 价格位置 + 方向 + 斜率 |
| 回踩位置 | 0.28 | dist_ma10/ma20 + range_pos + pullback_depth (钟形曲线) |
| 量价确认 | 0.22 | 回调缩量 + 量比 + 下跌缩量比 + 底部放量 |
| 分歧后修复 | 0.16 | 分歧信号 + 守住支撑 + 缩量 + 阳线修复 |
| 风险度量 | 0.12 | ATR% + 回撤 + 上影线密度 + 一字板风险 |

**大盘环境乘数**：bull=1.10 / neutral=1.00 / bear=0.75

**动作映射**：
- `setup_ready` (≥80): 时机较好
- `watch` (65~79): 等待确认
- `wait` (45~64): 暂不介入
- `avoid_chase` (<45): 避免追高

**参考价位输出**：支撑区间 / 失效位 / 压力位 / 盈亏比 + 价位置信度 (high/medium/low)

---

## 评估框架

`evaluation/` 目录提供事后评估工具：

- **label_generator.py** — 为已筛选股票生成未来收益标签（T+1/T+3/T+5 收益率、MFE/MAE、是否涨停、是否超指数）
- **harness.py** — 分组对比（pass vs reject、高分 vs 低分）、单调性检验、分离度指标、消融实验
- **report.py** — 生成文本/CSV 评估报告

---

## 数据源 — Tushare Pro (18 个 API)

| 接口 | 积分 | 用途 |
|------|------|------|
| `trade_cal` | 120 | 交易日历 |
| `stock_basic` | 120 | 股票列表/行业/上市日期 |
| `daily` | 120 | 日线行情（HT/TF/RC 多项） |
| `daily_basic` | 2000 | PE/PB/换手率/市值/量比 |
| `limit_list_d` | 5000 | 涨停池 + Universe Builder |
| `top_list` | 2000 | 龙虎榜 + Universe Builder |
| `pledge_stat` | 2000 | RC6 质押统计 |
| `share_float` | 3000 | RC7 限售解禁 |
| `stk_holdernumber` | 2000 | RC8 股东人数变动 |
| `stk_holdertrade` | 2000 | RC9 股东增减持 |
| `index_classify` | 2000 | 申万行业分类 |
| `index_member_all` | 2000 | 指数成分→行业映射 |
| `dividend` | 2000 | 分红记录 |
| `moneyflow` | 2000 | TF6 个股资金流向 |
| `margin_detail` | 2000 | TF7+RC10 融资融券 |
| `ths_index` | 6000 | HT5/HT6/A4 同花顺指数列表 |
| `ths_daily` | 6000 | HT5/HT6/A4 板块指数行情 |
| `ths_member` | 6000 | HT6 概念板块成分 |

> **自动降级：** 系统运行时自动检测积分权限。有 6000 积分走完整版，否则自动降级（HT5 → `tushare_200_degraded`，HT6 → `tushare_200_unavailable`）。

---

## 安装与运行

### 环境要求

- Python 3.9+
- Tushare Pro 账号（推荐 600元/年 6000 积分以获取完整功能）

### 安装

```bash
git clone https://github.com/QifangCao1999/a-share-hot-screener.git
cd a-share-hot-screener

pip install -r requirements.txt

# 配置环境变量（创建 .env 文件）
cat > .env << EOF
TUSHARE_TOKEN=your_tushare_token
DISCORD_BOT_TOKEN=your_discord_bot_token    # 可选，用于自动推送
DISCORD_CHANNEL_ID=your_channel_id          # 可选
DISCORD_USER_ID=your_user_id               # 可选，DM fallback
EOF
```

### 基本运行

```bash
# 单只/多只股票筛选（token 从 .env 自动读取）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,300750" \
  --output-dir ./output \
  --max-workers 3

# 指定日期运行
python3 -m a_share_hot_screener \
  --run-date 2026-04-22 \
  --stock-codes "600519" \
  --output-dir ./output

# 宽松模式（降低通过阈值，扩大候选集）
python3 -m a_share_hot_screener --preset relaxed \
  --run-date today \
  --stock-codes "600519,000858" \
  --output-dir ./output
```

### 批量运行

```bash
# 自动分批（每批 100 只，全局 scoring pool，自动合并结果）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,...(大量代码)..." \
  --output-dir ./output \
  --batch-size 100 \
  --max-workers 3

# 断点续跑（跳过已完成批次）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,..." \
  --output-dir ./output \
  --batch-size 100 --resume
```

### 启用实验性功能

```bash
# 启用观察时机评估
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,300750" \
  --output-dir ./output \
  --enable-setup-timing
```

### 每日自动运行器

```bash
# 标准运行（自动检测交易日 + 17:30 CST 数据就绪策略）
python3 -m a_share_hot_screener.daily_runner

# 指定日期
python3 -m a_share_hot_screener.daily_runner --date 2026-04-22

# 启用动态 Universe + 观察时机 + Discord 推送
python3 -m a_share_hot_screener.daily_runner \
  --use-universe-builder \
  --enable-setup-timing

# 15:00~17:30 CST 之间强制使用当天数据（可能不完整）
python3 -m a_share_hot_screener.daily_runner --allow-partial-current-day

# 仅检查，不实际运行
python3 -m a_share_hot_screener.daily_runner --dry-run
```

#### 17:30 CST 数据就绪策略

| 时间 (CST) | 行为 |
|------------|------|
| 17:30 后 | 使用当天完整数据 |
| 15:00~17:30 | 默认用前一交易日（`--allow-partial-current-day` 允许当天） |
| 盘中/周末 | 自动回溯最近交易日 |

#### Cron 定时运行

```bash
# 每个工作日 01:45 PDT (16:45 CST) 自动运行
45 1 * * 1-5 cd /path/to/project && python3 -m a_share_hot_screener.daily_runner \
  --use-universe-builder --enable-setup-timing >> screener_output/cron.log 2>&1
```

### 缓存管理

```bash
# 清理旧版本缓存
python3 -m a_share_hot_screener --purge-cache

# 查看缓存统计
python3 -m a_share_hot_screener --cache-stats

# 清理已过期缓存
python3 -m a_share_hot_screener --purge-expired
```

---

## 输出文件

### 核心筛选输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `{date}_stage1_hot_summary.csv` | CSV 宽表 | 每只股票一行，含四轴分数、total_score、pass_stage1、watch_only |
| `{date}_stage1_hot_detail.csv` | CSV 长表 | 每只股票×每个指标，完整溯源（raw_value → subscore） |
| `{date}_stage1_hot_rejected.csv` | CSV | 被淘汰股票，含 reject_stage / reject_reason |
| `{date}_stage1_hot_metadata.json` | JSON | 运行元数据（耗时、参数、统计） |
| `{date}_sector_heat.csv` | CSV | 板块轮动信号 |
| `{date}_setup_timing.csv` | CSV | 观察时机评估（if enabled） |

### 输出示例 (summary)

| code | name | total_score | hot_theme | trend_flow | le | rc | pass_stage1 | watch_only |
|------|------|-------------|-----------|------------|----|----|-------------|------------|
| 300476 | 胜宏科技 | 0.790 | 0.907 | 0.790 | 0.601 | 0.703 | True | False |
| 002475 | 立讯精密 | 0.772 | 0.837 | 0.852 | 0.579 | 0.658 | True | False |
| 688981 | 中芯国际 | 0.735 | 0.755 | 0.763 | 0.579 | 0.885 | True | False |

### Discord 推送内容

- **概览 Embed**：数据策略 / partial 风险 / universe 来源 / 统计
- **Tradeable 候选**：分开展示，含 timing 信息（if enabled）
- **Watch-Only 候选**：高辨识度/仅观察
- **Setup Timing Embed**：setup_ready / watch / wait / avoid_chase 分组
- **CSV 附件**：summary + detail 文件

---

## 板块轮动信号

`sector_rotation.py` 提供板块轮动分析（需 6000 积分，已集成到 pipeline Step 2.9 + HT7）：

- 计算行业 + 概念指数近 5/10/20 日涨幅、横截面排名百分位
- 动量切换信号分类：
  - `rotate_in`：20d 排名 < 50% 且 5d 排名 > 70%（低位板块突然加速）
  - `rotate_out`：20d 排名 > 70% 且 5d 排名 < 40%（强势板块动量衰减）
  - `steady_strong` / `steady_weak` / `neutral`
- 输出 `sector_heat.csv`

---

## 硬性过滤（H1~H9）

| 编号 | 条件 | 默认值 |
|------|------|--------|
| H1 | 数据覆盖率 | ≥ 75% |
| H2 | 非 ST/退市 | — |
| H3 | 最少上市交易日 | ≥ 20 日（可配置） |
| H4 | 最低价格 | ≥ 3.0 元 |
| H5 | 排除北交所（可选） | 默认排除 |
| H6 | 5日均成交额 | ≥ 1 亿元 |
| H7 | 流通市值 | ≥ 15 亿元 |
| H8 | 排除金融行业（可配置） | 默认排除 |
| H9 | 近5日最大单日跌幅 | ≤ 涨跌停幅度 |

> 核心数据（price/amount/market_cap）缺失 → hard_fail；非核心数据缺失 → warn（放行）。

---

## 缓存机制

- **缓存位置：** `~/.a_share_hot_screener/cache/`
- **版本化：** `CACHE_SCHEMA_VERSION = "v2"` 嵌入缓存键哈希，升级自动失效旧缓存
- **TTL：** `daily_basic` 4 小时，其他 API 按天或更长
- **性能：**
  - 100 只冷启动：~180s（含 prefetch_risk + prefetch_flow）
  - 100 只热缓存：~0.6s
  - 500 只（5批）：~38min

---

## 评分框架 (scoring.py)

统一的评分函数层，所有指标通过标准化的评分函数映射到 `[0, 1]` 子分：

| 函数 | 说明 |
|------|------|
| `score_lower_bound` | 下限型三段折线（越高越好） |
| `score_upper_bound` | 上限型三段折线（越低越好） |
| `score_clamp_linear` | 区间线性映射 |
| `score_discrete` | 离散值查表 |
| `score_percentile` | 横截面百分位（预排序 + bisect，26x 加速） |
| `score_bool` | 布尔型 True→1.0 / False→0.0 |

每个指标产出一个 `ScoreItem`，包含完整溯源（`raw_value` / `derived_value` / `subscore` / `weight` / `note`），便于审计。

---

## 测试

```bash
# 运行全部测试
python3 -m pytest tests/ -v

# 快速运行
python3 -m pytest tests/ -q
# → 1010 passed, 2 skipped in ~7s

# 运行特定模块测试
python3 -m pytest tests/test_context_scores.py -v
python3 -m pytest tests/test_setup_timing.py -v
```

**测试覆盖：**
- 1010 个测试用例，全部通过
- 所有 Tushare API 调用均已 mock，测试无需网络/token
- 覆盖：单元测试、集成测试、边界条件、降级路径、评分校准、消融实验

---

## 关键设计决策

1. **权限自动检测** — 运行时先尝试高积分 API，失败自动降级，无需手动配置
2. **HT5 行业热度三策略级联** — L3→L2→L1 精确匹配 + 投票桥接 + 名称直匹配，覆盖率 100%
3. **概念热度取最强** — 一只股票可能属于多个概念板块，取百分位最高者
4. **HT8/9/10 先 experimental** — 3-flag 独立控制（compute / use_in_total / show_in_discord）
5. **一字板双重验证** — OHLC 形态 + 涨跌幅方向，两个条件同时满足才计为一字板
6. **小池百分位修正** — 大池正常 / 小池(2~29) cap 0.90 / 极小池(<2) 绝对涨幅 fallback
7. **基准 pool 补充** — 当 scoring_pool < 30 只时自动合并 baseline_pool.json
8. **全局 scoring pool** — 批量模式下跨批次共享 pool，避免分批百分位不一致
9. **Pipeline 薄编排** — `pipeline.py` 仅做流程编排，业务逻辑分散到各模块
10. **17:30 CST 数据就绪** — 避免 Tushare 收盘后数据不完整的窗口期
11. **pass_stage1 = tradeable only** — watch_only 显式 opt-in，安全默认

---

## 踩坑记录

| 问题 | 说明 |
|------|------|
| Tushare `amount` 单位 | `daily.amount` 为**千元**，需 ×1000 转元 |
| Tushare `total_mv/circ_mv` 单位 | `daily_basic` 中为**万元** |
| Ticker 格式 | Tushare 使用 `600519.SH` 格式，通过 `ticker_mapping` 转换 |
| 风控接口超时 | `pledge_stat`/`share_float`/`stk_holdernumber` 不稳定，已有预加载+重试 |
| 涨停池 `limit_times` | 表示连板天数，>1 即为连板强势股 |
| 非交易日数据 | `daily_basic` 返回上一交易日数据，建议工作日运行 |
| flags 重建 | Step 7.5 `compute_flags()` 会重建 flags，stock_processor 写入的必须显式保留 |
| TF7/RC10 适用性 | 非两融标的 `is_applicable=False`，不影响轴 score |
| HT9/HT10 板块数据 | 从当前 run 的 details 分组构建，单只查询可能 is_applicable=False |
| setup_timing 数据 | 需 120 日数据，单独获取（Tushare LocalCache 命中，API 调用少） |

---

## 配置参数

完整参数列表：`python3 -m a_share_hot_screener --help`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run-date` | *必填* | 运行日期（`YYYY-MM-DD` 或 `today`） |
| `--stock-codes` | *必填* | 股票代码（逗号分隔） |
| `--output-dir` | *必填* | 输出目录 |
| `--tushare-token` | .env | Tushare API token |
| `--max-workers` | 3 | 并发线程数 |
| `--preset` | default | 评分预设（default / relaxed） |
| `--batch-size` | 0 | 每批大小（0=不分批，建议 100） |
| `--resume` | false | 断点续跑 |
| `--enable-setup-timing` | false | 启用观察时机评估 (experimental) |
| `--enable-concept-heat-module` | true | 概念热度（无权限自动降级） |
| `--include-beijing` | false | 包含北交所 |
| `--include-finance` | false | 包含金融行业 |
| `--min-price` | 3.0 | H4 最低价格 |
| `--min-amount-avg-5d` | 1亿 | H6 五日均成交额下限 |
| `--min-float-market-cap` | 15亿 | H7 流通市值下限 |
| `--min-trading-days` | 20 | H3 最少上市天数 |
| `--no-global-pool` | false | 禁用跨批全局 pool |
| `--purge-cache` | false | 清理旧版本缓存 |
| `--cache-stats` | false | 显示缓存统计 |

---

## 技术栈

- **语言：** Python 3.9+
- **数据源：** [Tushare Pro](https://tushare.pro/)（6000 积分推荐）
- **数据处理：** pandas
- **测试：** pytest（全部 mock，无需网络）
- **缓存：** 文件 JSON 缓存（版本化 + purge + stats）
- **并发：** `concurrent.futures.ThreadPoolExecutor`
- **通知：** Discord Bot REST API（Embed + 附件 + DM fallback）

---

## License

Private — 仅供个人使用。
