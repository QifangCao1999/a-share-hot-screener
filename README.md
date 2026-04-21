# A股短线热点筛选器 (A-Share Hot Screener)

> **Stage 1 — 结构化、客观、可复现的短线热点股票筛选与打分系统**

## 项目概述

本项目实现了一套 A 股短线热点股票的第一阶段（Stage 1）客观筛选系统。系统接收一组股票代码，通过 **硬性过滤 → 多维度特征计算 → 四轴评分 → 阈值判定** 的流水线，输出结构化的筛选结果。

**核心原则：**
- 仅使用 **Tushare Pro** 作为数据源，保证数据口径统一
- 严格遵守 **as-of / point-in-time** 口径：仅使用 `run_date` 当天或之前已知数据
- 第一阶段只做结构化、客观、可脚本化、可复现的筛选与打分
- 禁止将热点催化语义判断、题材阶段主观判断等第二阶段内容混入

**项目规模：**
- 生产代码：~9,000 行 / 29 个模块
- 测试代码：~9,000 行 / 26 个测试文件 / 670 个测试用例
- 测试通过率：100%（~5s）

---

## 目录结构

```
a_share_hot_screener/
├── clients/
│   └── tushare_client.py        # 804行 — Tushare API 客户端（TokenBucket 限流 + 重试 + 预加载）
├── scorers/
│   ├── hot_theme.py             # 194行 — HT1~HT6 热点题材评分
│   ├── trend_flow.py            # 168行 — TF1~TF5 趋势资金流评分
│   ├── liquidity_execution.py   # 187行 — LE1~LE4 流动性执行评分
│   └── risk_control.py          # 184行 — RC1~RC5 风险控制评分
├── tests/                       # 26 个测试文件，670 tests
│   ├── test_models.py
│   ├── test_scoring.py
│   ├── test_scoring_calibration.py
│   ├── test_event_layer.py
│   ├── test_event_layer_ths.py
│   ├── test_hard_filters.py
│   ├── test_price_features.py
│   ├── test_sector_rotation.py
│   ├── test_ths_client.py
│   ├── test_batch_runner.py
│   └── ... (16 more test files)
├── pipeline.py                  # 412行 — 薄编排层（orchestrator）
├── stock_processor.py           # 328行 — 单股处理（Step 3~6.8）
├── scoring_aggregator.py        # 145行 — 四轴评分聚合 + total_score
├── stage1_judge.py              # 124行 — Stage1 通过判定
├── scoring.py                   # 623行 — ScoreItem/AxisScore/ScoringPool 框架（预排序 + bisect）
├── event_layer.py               # 816行 — 涨停池(并发)/强势股/龙虎榜/行业热度/概念热度
├── models.py                    # 505行 — HotStockDetail(88字段)/Summary(自动投影)/Rejected/Metadata
├── price_features.py            # 463行 — 日线派生特征
├── config.py                    # 112行 — HotScreenerConfig
├── cli.py                       # 379行 — CLI 入口（dotenv + batch + purge-cache）
├── output.py                    # 320行 — CSV/JSON 输出
├── validation.py                # 304行 — SpotUniverse + StockValidator
├── trade_calendar.py            # 364行 — 交易日历 + last_n_trade_dates
├── hard_filters.py              # 324行 — H1~H9 硬筛
├── flags.py                     # 196行 — structured flags
├── trend_compare.py             # 396行 — 时序连续性对比
├── cache.py                     # 174行 — 文件 JSON 缓存（带版本化 + purge_stale）
├── batch_runner.py              # 341行 — 批量运行编排（自动分批/合并/断点续跑）
├── sector_rotation.py           # 296行 — 板块轮动信号（独立模块）
├── shareholder_reduction.py     # 146行 — 股东减持检测
├── pledge_ratio.py              # 86行 — 质押比例
├── restricted_unlock.py         # 127行 — 限售解禁
├── ticker_mapping.py            # 89行 — 代码格式转换
├── limit_rules.py               # 54行 — 涨跌停幅度推断
├── date_utils.py                # 90行
├── stock_codes.py               # 90行
├── logger.py                    # 84行
├── __main__.py                  # CLI 入口
└── __init__.py
```

---

## 核心架构

### Pipeline 流程

```
Step 0:    stock_codes 解析（支持逗号分隔、空格分隔、文件输入）
Step 0.5:  交易日历加载 + trade_date_used 确定
Step 1:    Tushare daily_basic 全市场表加载（SpotUniverse）
Step 2:    股票池校验（StockValidator）→ 同步收集 spot 字段
Step 2.5:  事件层批量加载（EventLayerLoader）→ 涨停池/强势股/龙虎榜/行业热度
Step 2.7:  风控数据预加载（pledge/float/holdnum，并发 5 线程）
Step 3~6:  并发个股处理 → stock_processor.process_single_stock()
           ├─ 日线行情获取（近60日）
           ├─ price_features 特征计算（22项）
           ├─ 硬性过滤（H1~H9）
           ├─ 事件层特征填充
           └─ 风控指标计算
Step 7:    四轴评分 → scoring_aggregator.apply_four_axis_scores()
Step 7.5:  structured flags 生成
Step 8:    pass_stage1 判定 → stage1_judge.judge_pass_stage1()
Step 9:    时序连续性对比（与前次运行结果比较趋势变化）
Step 10:   输出四类文件（summary/detail/rejected/metadata）
Step 11:   保存基准 pool（可选，用于单股查询时补充百分位分母）
```

### 四轴评分体系

本系统使用四维度加权评分，每个轴独立计算后汇总为 `total_score`：

| 评分轴 | 权重 | 指标数 | 说明 |
|--------|------|--------|------|
| **hot_theme** (热点题材) | 40% | HT1~HT6 | 收益率百分位 / 大涨天数 / 连续上涨 / 行业热度 / 概念热度 |
| **trend_flow** (趋势资金) | 30% | TF1~TF5 | 量比 / 成交额比 / 收盘位置 / 均线排列 / CLV |
| **liquidity_execution** (流动性) | 20% | LE1~LE4 | 成交额 / 换手率 / 流通市值 / 龙虎榜 |
| **risk_control** (风控) | 10% | RC1~RC5 | 一字板 / 振幅 / 均线偏离 / 上影线 / 累计涨幅 |

**pass_stage1 阈值（AND 关系）：**
- `total_score ≥ 0.68`
- `hot_theme ≥ 0.65`
- `trend_flow ≥ 0.60`
- `liquidity_execution ≥ 0.55`
- `risk_control ≥ 0.40`

### 硬性过滤（H1~H9）

在评分之前，股票必须通过以下硬性过滤条件：

| 编号 | 条件 | 默认值 |
|------|------|--------|
| H1 | 数据覆盖率 | ≥ 75% |
| H2 | 非 ST/退市 | — |
| H3 | 最少上市交易日 | ≥ 20 日 |
| H4 | 最低价格 | ≥ 3.0 元 |
| H5 | 排除北交所（可选） | 默认排除 |
| H6 | 5日均成交额 | ≥ 2 亿元 |
| H7 | 流通市值 | ≥ 15 亿元 |
| H8 | 排除金融行业（可选） | 默认排除 |
| H9 | 近5日最大单日跌幅 | ≤ 涨跌停幅度 |

### 评分框架 (scoring.py)

统一的评分函数层，所有指标通过标准化的评分函数映射到 `[0, 1]` 子分：

| 函数 | 说明 |
|------|------|
| `score_lower_bound` | 下限型三段折线（越高越好） |
| `score_upper_bound` | 上限型三段折线（越低越好） |
| `score_clamp_linear` | 区间线性映射 |
| `score_discrete` | 离散值查表 |
| `score_percentile` | 横截面百分位（支持升/降序） |
| `score_bool` | 布尔型 True→1.0 / False→0.0 |

每个指标产出一个 `ScoreItem`，包含完整溯源信息（`raw_value` / `derived_value` / `subscore` / `weight` / `note`），便于审计。

---

## 数据源

### Tushare Pro（200元/年，~5000 积分）

| 接口 | 积分 | 用途 |
|------|------|------|
| `daily` | 120 | 个股日线行情 |
| `daily_basic` | 2000 | 全市场 PE/PB/换手率/市值/量比 |
| `trade_cal` | 120 | 交易日历 |
| `stock_basic` | 120 | 股票列表（行业/上市日期） |
| `limit_list_d` | 5000 | 涨停池（含连板数） |
| `top_list` | 2000 | 龙虎榜 |
| `pledge_stat` | 2000 | 质押统计 |
| `share_float` | 3000 | 限售解禁 |
| `stk_holdernumber` | 2000 | 股东人数变动 |

### 可选增强（需 600元/年，~6000 积分）

| 接口 | 用途 | 降级行为 |
|------|------|----------|
| `ths_index` | 同花顺行业/概念指数列表 | HT5 降级为 `tushare_200_degraded` |
| `ths_daily` | 板块指数行情 | HT6 降级为 `tushare_200_unavailable` |
| `ths_member` | 概念板块成分 | — |

> **自动降级：** 系统运行时自动检测积分权限，无需手动配置。有 6000 积分自动走完整版，否则自动降级。

---

## 安装与运行

### 环境要求

- Python 3.9+
- Tushare Pro 账号（200元/年起）

### 安装

```bash
git clone https://github.com/QifangCao1999/a-share-hot-screener.git
cd a-share-hot-screener

# 安装依赖
pip install -r requirements.txt

# 配置 Tushare Token（二选一）
# 方式一：创建 .env 文件
echo "TUSHARE_TOKEN=your_token_here" > .env

# 方式二：通过 CLI 参数传入（见下方运行命令）
```

### 基本运行

```bash
# 单只/多只股票筛选（token 从 .env 自动读取）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,300750" \
  --output-dir ./output \
  --max-workers 3

# 显式传入 token
python3 -m a_share_hot_screener \
  --tushare-token "your_token" \
  --run-date 2026-04-20 \
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
# 自动分批（每批 100 只，自动合并结果）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,...(大量代码)..." \
  --output-dir ./output \
  --batch-size 100 \
  --max-workers 3

# 断点续跑（上次中断后跳过已完成批次）
python3 -m a_share_hot_screener \
  --run-date today \
  --stock-codes "600519,000858,..." \
  --output-dir ./output \
  --batch-size 100 --resume
```

### 其他命令

```bash
# 清理旧版本缓存
python3 -m a_share_hot_screener --purge-cache

# 保存基准 pool（大规模运行后，用于单只查询时补充百分位分母）
python3 -m a_share_hot_screener --save-baseline-pool \
  --run-date today \
  --stock-codes "..." \
  --output-dir ./output
```

---

## 输出文件

每次运行产出四类文件（文件名含 `run_date` 前缀）：

| 文件 | 格式 | 说明 |
|------|------|------|
| `{date}_stage1_hot_summary.csv` | CSV 宽表 | 每只股票一行，含四轴分数、total_score、pass_stage1 |
| `{date}_stage1_hot_detail.csv` | CSV 长表 | 每只股票×每个指标一行，完整溯源（raw_value → subscore） |
| `{date}_stage1_hot_rejected.csv` | CSV | 被淘汰股票，含 reject_stage / reject_reason |
| `{date}_stage1_hot_metadata.json` | JSON | 运行元数据（耗时、参数、统计信息） |

### 输出示例 (summary)

| code | name | total_score | hot_theme | trend_flow | le | rc | pass_stage1 |
|------|------|-------------|-----------|------------|----|----|-------------|
| 300476 | 胜宏科技 | 0.790 | 0.907 | 0.790 | 0.601 | 0.703 | True |
| 002475 | 立讯精密 | 0.772 | 0.837 | 0.852 | 0.579 | 0.658 | True |
| 688981 | 中芯国际 | 0.735 | 0.755 | 0.763 | 0.579 | 0.885 | True |

---

## 板块轮动信号（独立模块）

`sector_rotation.py` 提供独立的板块轮动分析功能（需 6000 积分）：

- 计算行业 + 概念指数近 5/10/20 日涨幅
- 横截面排名百分位
- 动量切换信号分类：
  - `rotate_in`：20d 排名 < 50% 且 5d 排名 > 70%（低位板块突然加速）
  - `rotate_out`：20d 排名 > 70% 且 5d 排名 < 40%（强势板块动量衰减）
  - `steady_strong` / `steady_weak` / `neutral`
- 输出 `sector_heat.csv`

---

## 缓存机制

- **缓存位置：** `~/.a_share_hot_screener/cache/`
- **版本化：** `CACHE_SCHEMA_VERSION = "v2"` 嵌入缓存键哈希，升级自动失效旧缓存
- **TTL：** `daily_basic` 4 小时，其他 API 按天或更长
- **清理：** `--purge-cache` 命令清理旧版本缓存文件
- **性能：**
  - 100 只冷启动：~106s
  - 100 只热缓存：~0.6s

---

## 测试

```bash
# 运行全部测试
python3 -m pytest tests/ -v

# 快速运行
python3 -m pytest tests/ -q

# 运行特定测试文件
python3 -m pytest tests/test_scoring.py -v
```

**测试覆盖：**
- 670 个测试用例，全部通过
- 所有 Tushare API 调用均已 mock，测试无需网络/token
- 包含：单元测试、集成测试、边界条件、降级路径、评分校准

---

## 关键设计决策

1. **权限自动检测** — 运行时先尝试高积分 API，失败自动降级，无需手动配置积分等级
2. **概念热度取最强** — 一只股票可能属于多个概念板块，取百分位最高者
3. **行业映射优先级** — `index_member_all` 成分映射优先于 `stock_basic.industry`
4. **ths_daily 按天拉取** — 行业 + 概念可共享同一天缓存，减少 API 调用
5. **预排序优化** — `ScoringPool` 预排序 + `bisect_right`，百分位计算加速 26x
6. **基准 pool 补充** — 当 scoring_pool < 5 只时自动合并 baseline_pool.json
7. **Pipeline 薄编排** — `pipeline.py` 仅做流程编排，业务逻辑分散到各模块

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

---

## 配置参数一览

所有参数均可通过 CLI 覆盖，详见 `python3 -m a_share_hot_screener --help`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run-date` | *必填* | 运行日期（`YYYY-MM-DD` 或 `today`） |
| `--stock-codes` | *必填* | 股票代码（逗号分隔） |
| `--output-dir` | *必填* | 输出目录 |
| `--tushare-token` | 从 .env 读取 | Tushare API token |
| `--max-workers` | 3 | 并发线程数 |
| `--preset` | default | 评分预设（default / relaxed） |
| `--batch-size` | 0 | 批量运行每批大小（0=不分批） |
| `--resume` | false | 断点续跑 |
| `--save-baseline-pool` | false | 保存基准 pool |
| `--purge-cache` | false | 清理旧版本缓存 |
| `--enable-concept-heat` | false | 启用概念热度模块 |
| `--min-price` | 3.0 | H4 最低价格 |
| `--min-amount-avg-5d` | 200000000 | H6 五日均成交额下限 |
| `--min-float-market-cap` | 1500000000 | H7 流通市值下限 |

---

## 技术栈

- **语言：** Python 3.9+
- **数据源：** [Tushare Pro](https://tushare.pro/)
- **数据处理：** pandas
- **测试：** pytest（全部 mock，无需网络）
- **缓存：** 文件 JSON 缓存（版本化 + purge）
- **并发：** `concurrent.futures.ThreadPoolExecutor`

---

## License

Private — 仅供个人使用。
