# A股短线热点筛选系统 v2 — 设计文档（v2.1 修订版）

> 日期: 2026-04-22
> 状态: **v2.1 修订版，可进入分 Phase 开发**
> 基线: v1（785 tests / 10,100 行生产代码 / 28 评分子项 / 18 个 Tushare API）
> 审阅: 经 ChatGPT Pro 两轮 review → v2 终版候选 → 第三轮 review → v2.1 修订
> 变更记录:
>   - v2.0: 初始设计 + ChatGPT Pro 两轮 review 修订
>   - v2.1: 第三轮 review 修订（pass_stage1 语义/评估样本范围/权重归一/数据就绪时间/降级策略等 18 项）

---

## 一、改进目标

将当前「静态 Universe → Stage 1 量化筛选 → 手工 Stage 2 → 无 Stage 3」流程，
升级为「动态 Universe → 增强 Stage 1 → 观察时机评估 → 一键推送」的全自动日频系统。

### 改进前（v1）

```
静态 universe (797只 CSI300+500)
  → hot_screener Stage 1 (~38min)
    → Discord 推送 pass 列表
      → 手动粘贴到 ChatGPT Pro 做 Stage 2 (~30-60min 人工)
        → 人肉判断买入时机
```

### 改进后（v2）

```
Phase 0 准确度修复 (一次性)
  → universe_builder 动态构建 (~1500只, 2min)
    → hot_screener Stage 1 增强版 (~55min)
      → setup_timing 观察时机评估 (~2min)
        → Discord 一次性推送完整报告
          → 人工 5min 复盘确认
```

**人工参与从 ~60min 降到 ~5min，覆盖从 ~800 只扩展到 ~1500 只。**

---

## 二、系统架构总览

```
daily_runner.py (编排层)
 │
 ├── Step 0: universe_builder.py ──────── 动态构建当日候选池
 │     ├── 静态底仓 (CSI300+500+1000, 月更)
 │     ├── 近5日涨停池 (limit_list_d)
 │     ├── 近5日龙虎榜 (top_list)
 │     ├── 当日成交额 Top 200 (daily — 独立加载)
 │     └── 热门板块成分 (ths_daily + ths_member)
 │
 ├── Step 1: hot_screener Stage 1 ────── 量化筛选 (现有 + 增强)
 │     ├── 现有 28 子项评分
 │     ├── [新] 市场确认度 (HT8) ← experimental
 │     ├── [新] 板块扩散度 (HT9) ← experimental
 │     └── [新] 板块内辨识度 (HT10) ← experimental
 │
 ├── Step 2: setup_timing.py ─────────── 观察时机评估
 │     ├── 趋势状态
 │     ├── 回踩/支撑位置
 │     ├── 量价确认
 │     ├── 分歧后修复结构
 │     ├── 风险度量
 │     └── 大盘环境 (乘数调节)
 │
 └── Step 3: discord_notifier ─────────── 推送
       ├── 概览 + 模块状态
       ├── Stage 1 pass 列表 (tradeable / watch_only 分开展示)
       ├── 结构化上下文摘要
       ├── 观察时机信号 (experimental 阶段仅输出不主推)
       └── CSV 附件 (失败不阻塞)
```

---

## 三、Phase 0: v1 准确度修复（前置，必须先完成）

> 在任何 v2 新模块开发之前，先修好现有系统的已知问题。

### 0.1 P0-A: 长上影/冲高回落 proxy 统一 (~0.75h)

**问题**: `price_features.py` 和 `risk_control.py` 的上影线判断口径不一致，且语义未区分。

**修改**:

定义两个指标，各有明确语义：

```python
# 1. K线上影线（形态描述，用于 price_features 特征输出）
def upper_wick_ratio(open, high, low, close):
    """经典 K 线上影线比率。"""
    if high <= low:
        return 0.0
    return (high - max(open, close)) / (high - low)

# 2. 冲高回落（交易风险信号，用于 RC4 评分和 crowding cap）
def upper_reversal_ratio(high, low, close):
    """当日冲高回落幅度。close 越远离 high，风险越高。"""
    if high <= low:
        return 0.0
    return (high - close) / (high - low)

# RC4 使用 upper_reversal_ratio（风险导向）
amplitude_pct = (high - low) / prev_close
is_reversal_risk = upper_reversal_ratio(high, low, close) >= 0.45 and amplitude_pct >= 0.05
```

- `upper_wick_ratio` → price_features 形态描述、CSV 输出
- `upper_reversal_ratio` → RC4 评分、crowding cap 判定、Setup Timing 风险度量
- 两个函数放在共用模块 `indicators.py`，消除 price_features / risk_control 各自实现的不一致
- RC4 的 `threshold_text`、`note`、字段名全部统一为 `upper_reversal_ratio`

### 0.2 P0-B: LE2 真实换手率优先 (~0.5h)

**问题**: LE2 始终使用 `amount/float_market_cap` proxy，但 `daily_basic.turnover_rate_f` 已在 SpotUniverse 中加载。

**修改**:

数据优先级：
1. `daily_basic.turnover_rate_f`（自由流通换手率）— 最准确
2. `daily_basic.turnover_rate`（总股本换手率）— 次选
3. `amount_avg_5d / float_market_cap * 100`（proxy，子分 cap 0.85）— 兜底

- 在 `stock_processor.py` 中传入 `turnover_rate_f`
- `liquidity_execution.py` LE2 判断优先级
- 新增输出字段 `turnover_method: turnover_rate_f | turnover_rate | amount_proxy`
- **不需要额外 API 调用**

### 0.3 P0-C: watch_only_candidate 分池 (~1.5h)

**问题**: `pass_stage1` 是二元的，无法区分「强但不可交易」（一字涨停、高位拥挤）。

**修改**:

新增字段与语义变更：

```python
# 语义变更（v2.1 关键修订）
pass_stage1: bool           # = tradeable only（安全默认，旧消费方自动排除 watch_only）
pass_stage1_watch: bool     # = 观察池（强但当前不可交易）
pass_stage1_any: bool       # = tradeable ∪ watch_only（显式 opt-in）

# 分池字段
candidate_pool_type: str    # tradeable / watch_only / failed_score / insufficient_data / rejected_hard
candidate_pool_reason: str  # 具体原因描述
```

`candidate_pool_type` 枚举：

| 类型 | 条件 |
|------|------|
| `tradeable` | 分数达标 且 非一字涨停 且 无 very_high 风险 |
| `watch_only` | 分数达标 但 最新日一字涨停 / crowding_cap_applied / high 风险 flag |
| `failed_score` | 通过硬筛但分数不足 |
| `insufficient_data` | `core_data_coverage` < 阈值 |
| `rejected_hard` | 硬筛失败 |

> **v2.1 关键变更**: `pass_stage1` 不再包含 watch_only。
> 旧代码读 `pass_stage1=True` 时只会拿到 tradeable 股票，消除了误推送/误回测风险。
> 需要包含 watch_only 的场景必须显式读 `pass_stage1_any`。

### 0.4 P0-D: core/overall coverage 拆分 (~1h)

**问题**: 可选模块（质押/解禁/龙虎榜等）不可用时拉低 coverage。

**修改**:

新增：
- `core_data_coverage`: 价格/成交量/成交额/流通市值/收益率/均线/CLV/振幅
- `overall_data_coverage`: core + 事件层 + 风控增强 + 行业概念热度

`pass_stage1` (= tradeable) 使用 `core_data_coverage` 判断。

### 0.5 P0-E: HT5 行业热度 degraded cap (~0.25h)

**问题**: 6000pt 正常模式下 HT5 走完整三策略级联（覆盖 100%），但降级模式（`tushare_200_degraded`）使用简化计算，精度较低。当前降级模式无 cap，可能给出过高分数。

**修改**:

```python
# hot_theme.py HT5 评分
if industry_heat_source == "tushare_200_degraded":
    ht5_sub_score = min(ht5_sub_score, 0.80)  # degraded cap
```

- 降级模式子分上限 0.80（正常模式无 cap）
- 输出字段 `ht5_degraded_cap_applied: bool`
- 成本极低（3 行代码 + 1 个测试），但为降级场景提供安全阈值

### Phase 0 总计: ~4h

---

## 四、Phase 1: 评估框架（Evaluation Harness）

> HT8/9/10 和 Stage 3 的阈值与权重不能拍脑袋定——必须先建评估框架，再用数据校准。

### 1.1 文件

```
evaluation/
  harness.py          (~300行)
  label_generator.py  (~200行)
  report.py           (~120行)
```

### 1.2 标签生成

> **v2.1 关键变更**: 标签生成覆盖**全量 scored stocks**（所有通过硬筛的股票），不仅仅是 pass_stage1。

对历史每个 `run_date` 的**全量通过硬筛股票**，用 Tushare `daily` 生成未来收益标签：

| 标签 | 定义 |
|------|------|
| `return_t1` | T+1 日收益 |
| `mfe_3d` | 未来 3 日最大有利偏移 (Maximum Favorable Excursion) |
| `mfe_5d` | 未来 5 日 MFE |
| `mae_5d` | 未来 5 日最大不利偏移 (Maximum Adverse Excursion) |
| `hit_limit_up_3d` | 未来 3 日是否涨停 |
| `beat_index_5d` | 未来 5 日是否跑赢沪深 300 |

**样本分组**（用于对比分析）：

| 分组 | 定义 | 用途 |
|------|------|------|
| `pass_tradeable` | `candidate_pool_type == "tradeable"` | 主要评估对象 |
| `pass_watch_only` | `candidate_pool_type == "watch_only"` | 验证 watch_only 是否包含题材锚点 |
| `failed_score` | `candidate_pool_type == "failed_score"` | 对照组：分数不足 |
| `top_N` | total_score 排名前 N | 排序有效性 |
| `bottom_N` | total_score 排名后 N（通过硬筛内） | 排序有效性对照 |

**数据保存要求**:
- 每次 run 必须保存**全量 summary CSV**（当前已有此机制）
- evaluation harness 读取全量 summary + 对应标签
- 不允许只评估 pass 子集

### 1.3 评估指标

| 指标 | 用途 |
|------|------|
| Top N 命中率 | total_score 排名前 N 的股票，未来 3/5 日 MFE > X% 的比例 |
| Top N 平均 MFE | 前 N 名的平均最大涨幅 |
| Top N 平均 MAE | 前 N 名的平均最大回撤 |
| **pass vs fail 分离度** | pass_tradeable 组 vs failed_score 组的 MFE 分布差异（KS 检验或简单中位数对比） |
| **tradeable vs watch_only** | 两组的未来表现差异 |
| **分数排序单调性** | 按 total_score 十分位分组，MFE 是否递减 |
| 信号分层单调性 | timing_score 分组（80+/65~79/45~64/<45）是否收益递减 |

### 1.4 消融实验

| 实验 | 目的 |
|------|------|
| 去掉 crowding cap | cap 是否真的在防止追高亏损 |
| HT3 limit_up vs big_up | 哪种热度信号预测力更强 |
| LE2 真实换手率 vs proxy | 是否真的改善了 pass 组质量 |
| HT8/9/10 加入 vs 不加入 | 新子项是否提升 Top N 命中率 |
| Stage 3 分层 | timing_score 分组是否有信息量 |

### 1.5 工作量: ~4.5h（含全量样本处理增量）

---

## 五、Phase 2: Universe 动态构建

### 5.1 文件

```
universe_builder.py   (新建, ~380行)
universe/
  static_csi300.txt   (月更)
  static_csi500.txt   (月更)
  static_csi1000.txt  (月更)
  daily_YYYYMMDD.txt  (每日生成)
```

### 5.2 数据源与权限分级

| 来源 | API | 积分 | 权限级别 | 更新频率 |
|------|-----|------|---------|---------|
| 沪深300 成分 | `index_weight` | 2000 | Level 1 | 月更 |
| 中证500 成分 | `index_weight` | 2000 | Level 1 | 月更 |
| 中证1000 成分 | `index_weight` | 2000 | Level 1 | 月更 |
| 当日成交额 Top N | `daily` | 120 | Level 1 | 每日 |
| 近5日龙虎榜 | `top_list` | 2000 | Level 1 | 每日 |
| 近5日涨停池 | `limit_list_d` | 5000 | Level 2 | 每日 |
| 热门板块成分 | `ths_daily` + `ths_member` | 6000 | Level 3 | 每日 |

**降级规则**：
- **Level 1** 为核心数据源，原则上必须可用。但仍可能因 token 失效、限流、接口故障、网络异常而失败：
  - `trade_cal` / `stock_basic` 失败 → **终止运行**，这些是基础设施级依赖
  - `index_weight` 失败 → 使用最近本地快照，标记 `stale_static_universe=True`
  - `daily` 失败 → **终止运行**（成交额 Top N 和后续 Stage 1 都依赖此数据）
  - `top_list` 失败 → 跳过龙虎榜扩展，标记 `top_list_available=False`
- **Level 2** 不可用 → 跳过涨停池扩展，标记 `zt_pool_available=False`
- **Level 3** 不可用 → 跳过板块成分扩展，标记 `ths_hot_available=False`

### 5.3 构建流程

```python
class UniverseBuilder:
    def build(self, run_date: str) -> UniverseResult:
        """构建当日候选池."""
```

1. **加载静态底仓**
   - 从 `universe/static_csi*.txt` 读取
   - 文件不存在或超过 30 天未更新 → 自动拉取 `index_weight` 更新
   - 过期但拉取失败 → 使用旧文件，标记 `stale_static_universe=True`

2. **涨停池扩展**（Level 2）
   - 拉取近 5 个交易日 `limit_list_d(limit_type='U')`
   - 记录每只股票的涨停天数，供 HT8 使用

3. **龙虎榜扩展**（Level 1）
   - 拉取近 5 个交易日 `top_list`

4. **成交额 Top N 扩展**（Level 1）
   - **Universe Builder 自行加载** `daily(trade_date=run_date)`，不依赖 Step 1
   - 取 amount 排名 Top 200

5. **热门板块成分扩展**（Level 3）
   - `ths_daily` 取近 5 日涨幅 Top 10 板块
   - `ths_member` 获取成分股

6. **去重 + 基础过滤**
   - 合并所有来源，去重
   - 移除 ST / *ST / 退市 / 停牌
   - 上市天数过滤：通过配置项 `universe_min_listing_days`（默认 20）控制
     - 也可设为 0（不过滤），完全交给 Stage 1 的 hard_filter `min_trading_days` 处理
     - **不硬编码 60 天**——Universe Builder 广撒网，精确过滤交给 Stage 1

7. **行业过滤**（可选）
   - 配置项 `universe_excluded_industries: List[str]`（默认空列表）
   - 如需排除金融行业：`["银行", "保险", "证券", "多元金融"]`
   - 金融股（银行/保险/券商）不适合短线热点框架，但作为策略决策而非硬编码

8. **输出**
   - `universe/daily_YYYYMMDD.txt`
   - `UniverseResult` 含来源标记 + 可用性状态

### 5.4 来源标记

```python
source_tags: Dict[str, Set[str]]
# {"600519": {"csi300", "amount_top200"},
#  "002436": {"zt_pool", "ths_concept_hot"}}
```

写入 Stage 1 metadata，用于回测分组分析。

### 5.5 预估规模

| 层级 | 数量 | 说明 |
|------|------|------|
| CSI300+500+1000 原始 | ~1800 | 三个指数有重叠 |
| 去重后静态底仓 | ~1650 | |
| ST/停牌过滤后 | ~1550 | |
| + 涨停池 | +150~300 | 与底仓部分重叠 |
| + 龙虎榜 | +50~100 | |
| + 成交额 Top200 | +50~100 | 大部分已在底仓 |
| + 热门板块 | +100~200 | |
| **最终去重** | **~1500~2200** | 视行情而定 |

### 5.6 运行时间

| 行情 | Universe | 批次 | Stage 1 耗时 |
|------|----------|------|-------------|
| 冷清 | ~1200 | 12 | ~45 min |
| 普通 | ~1500 | 15 | ~55 min |
| 火热 | ~2000 | 20 | ~75 min |

Universe 构建自身 ~2 分钟。

### 5.7 工作量: ~3h（含行业过滤配置增量）

---

## 六、Phase 3: Stage 1 Context Scores（HT8/9/10 experimental）

### 6.1 设计原则

> **Stage 1 不能冒充 Stage 2。**
>
> - HT8/9/10 只表示「结构化市场确认度 / 板块扩散度 / 板块内辨识度」
> - 不能替代真正的公告/政策/订单/业绩催化分析
> - 不能替代对公司-题材真实受益的判断
> - HT6 + ths_member 仅提供 `concept_mapping_evidence`，不等于「公司-题材相关性已覆盖」
> - **初期 experimental 输出，不进入总分；回测验证后再决定是否纳入**

### 6.2 HT8: 市场确认度 (market_confirmation_score)

**定义**: 用结构化市场行为数据推断该股近期被市场关注的程度。

**评分逻辑**（离散型）：

| 信号组合 | 确认级别 | 子分 | Level 3 可用 |
|---------|---------|------|-------------|
| 龙虎榜 + 涨停 + 概念板块近5日涨幅 Top10 | 多重市场确认 | 1.0 | ✅ 需要 |
| 涨停 + 概念板块活跃（Top20） | 板块共振确认 | 0.80 | ✅ 需要 |
| 龙虎榜但未涨停 / 涨停但板块不活跃 | 单点异动确认 | 0.60 | ❌ 不需要 |
| 成交额 >2x 放大 + 涨幅>5% 但未涨停 | 资金异动 | 0.40 | ❌ 不需要 |
| 无以上信号 | 无明显结构确认 | 0.10 | ❌ 不需要 |

**数据来源**（全部已有 API）：
- 涨停: `limit_list_d`
- 龙虎榜: `top_list`
- 概念板块活跃度: `ths_daily`
- 成交额变化: `daily`

**Level 3 缺失降级**（v2.1 新增）：

```python
if not ths_daily_available:
    # 无法判断板块活跃度，禁止输出"板块共振确认"
    ht8_cap = 0.60  # 最高只能到"单点异动确认"
    confirmation_level 只在 [资金异动, 单点异动确认, 无明显结构确认] 中选择
```

**注意**: 多重市场确认只说明市场在交易某个叙事，**不能证明催化真实**。

### 6.3 HT9: 板块扩散度 (sector_breadth_score)

**定义**: 该股所属热门概念板块内，同步走强股票的比例。

**评分逻辑**（三段下限型）：

```
输入: sector_breadth_ratio = 板块内近5日涨幅>5%的股票占比

L=10% → 0.0    (只有零星个股涨)
T=30% → 0.70   (约1/3走强，正常扩散)
H=50% → 1.0    (半数以上走强)
```

**样本污染处理**：
| 板块成分数 | 规则 |
|-----------|------|
| ≥ 30 | 正常计算 |
| 10~29 | HT9 cap 0.80 |
| < 10 | HT9 不适用 (`is_applicable=False`) |

**辅助输出**（v2.1 新增，初期不进分，用于回测分析）：

| 字段 | 定义 | 用途 |
|------|------|------|
| `sector_amount_breadth_ratio` | 板块内近5日成交额放大>50%的股票占比 | 区分"小票普涨"vs"容量股参与" |
| `top5_amount_concentration` | 板块成交额 Top5 占板块总成交额比例 | 判断是否只有少数极端个股拉动 |

> 只有小票普涨但容量股不动，不应视为高质量扩散。辅助指标帮助回测验证是否需要将成交额扩散纳入评分。

### 6.4 HT10: 板块内辨识度 (sector_position_score)

**定义**: 该股在所属板块内的领涨/容量地位。

**完整公式**（Level 2 可用时）：

```python
score = (
    0.30 * rank_pctile        # 涨幅在板块内的百分位
  + 0.35 * amount_share_score # 成交额在板块 Top10 中的占比 (权重高，避免误伤大票)
  + 0.35 * first_zt_score     # 是否率先涨停
)
```

**Level 2 缺失降级**（v2.1 新增）：

```python
if not limit_list_d_available:
    # first_zt_score 无法可靠计算
    score = (
        0.46 * rank_pctile        # 重新归一: 0.30/0.65
      + 0.54 * amount_share_score # 重新归一: 0.35/0.65
    )
    ht10_cap = 0.80
    sector_position_confidence = "medium"
else:
    sector_position_confidence = "high"
```

**输出辅助字段**（不参与评分）：
- `sector_position_type`: `frontline_like` / `capacity_core_like` / `follower_like` / `unknown`
- `sector_position_confidence`: `high` / `medium`（v2.1 新增）

**设计决策**: `amount_share` 权重 0.35（高于 rank_pctile 的 0.30），确保大成交容量中军不会因为涨幅排名稍低就被打成「跟风」。

### 6.5 Feature Flag 设计（v2.1 修订）

> 原 `enable_context_scores` 单一 flag 拆分为三个正交维度。

```python
# config.py
compute_context_scores: bool = True       # 是否计算 HT8/9/10 并输出到 CSV
use_context_scores_in_total: bool = False  # 是否纳入 HT 轴总分
show_context_scores_in_discord: bool = False  # 是否展示到 Discord 推送

# 典型使用组合:
# experimental 阶段:  compute=True,  use=False, show=False  → 只算不用
# 回测观察阶段:       compute=True,  use=False, show=True   → 算+展示但不影响排序
# 正式启用:           compute=True,  use=True,  show=True   → 全部启用
# 临时关闭:           compute=False, use=False, show=False  → 完全关闭
```

### 6.6 增强后的 HT 轴（`use_context_scores_in_total=True` 时）

| 子项 | 权重 | 状态 |
|------|------|------|
| HT1 近5日收益百分位 | 8 | ✅ 现有 |
| HT2 近10日收益百分位 | 6 | ✅ 现有 |
| HT3 大涨天数 | 8 | ✅ 现有 |
| HT4 连续上涨天数 | 6 | ✅ 现有 |
| HT5 行业热度 | 7 | ✅ 现有 |
| HT6 概念热度 | 7 | ✅ 现有 |
| HT7 板块轮动动量 | 5 | ✅ 现有 |
| **HT8 市场确认度** | **6** | 🧪 experimental |
| **HT9 板块扩散度** | **5** | 🧪 experimental |
| **HT10 板块内辨识度** | **6** | 🧪 experimental |

### 6.7 工作量: ~6.5h（含降级逻辑和辅助输出增量）

---

## 七、Phase 4: 观察时机评估（Setup Timing）

### 7.1 文件

```
setup_timing.py            (新建, ~450行)
tests/test_setup_timing.py (新建, ~350行)
```

### 7.2 设计定位

| 属性 | 值 |
|------|-----|
| 输入 | **pass_stage1 tradeable** 的股票 + 近 120 日 OHLCV 日线 |
| 输出 | `SetupSignal`（评分 + 动作 + 关键价位 + 理由） |
| 观察周期 | 3~5 天 |
| 风格偏向 | 低吸/回踩 优先，分歧后修复 次之 |
| 数据频率 | 日频 |
| 运行时间 | ~2 分钟（daily 数据已缓存） |
| **初始状态** | **experimental — 回测通过后再正式推送** |

> **v2.1 关键变更**: 默认只对 `pass_stage1` (= tradeable) 运行。
> watch_only 默认不计算 setup_timing。
> 即使通过配置允许对 watch_only 计算，其 `action` 最高只能到 `watch`，不得到 `setup_ready`。

### 7.3 数据模型

```python
@dataclass
class SetupSignal:
    """单只股票的观察时机评估结果."""
    code: str
    name: str

    # 综合评分
    timing_score: float           # 0~100
    action: str                   # wait / watch / setup_ready / avoid_chase

    # 关键参考价位
    support_zone_low: float       # 支撑区间下沿
    support_zone_high: float      # 支撑区间上沿
    invalidation_level: float     # 失效位 (跌破则逻辑不成立)
    resistance_1: float           # 第一压力位
    ref_reward_risk: float        # 参考盈亏比
    level_confidence: str         # high / medium / low (v2.1 新增)

    # 分项评分 (0~1)
    trend_score: float            # 趋势状态
    pullback_score: float         # 回踩位置
    volume_score: float           # 量价确认
    repair_score: float           # 分歧后修复结构
    risk_score: float             # 风险度量

    # 大盘
    market_regime: str            # bull / neutral / bear
    market_multiplier: float      # 0.75~1.10

    # 辅助
    support_basis: str            # ma10 / ma20 / swing_low / box_low
    reason: str                   # 一句话理由
    warnings: List[str]           # 风险提示
```

### 7.4 评分维度

#### 7.4.1 趋势状态 (权重 22%)

| 指标 | 评分 |
|------|------|
| MA 排列 (MA5>10>20>60) | 4线多头=1.0, 3线=0.7, 2线=0.4, 空头=0.0 |
| 价格 vs MA20 | 站上=加分, 跌破=减分 |
| 近3日方向 | 上升=加分, 下降=减分 |
| MA20 斜率 | 上行=加分, 走平=中性, 下行=减分 |

**低吸适配**: 中等偏强（0.5~0.8）最适合低吸——趋势向上但正在回踩。

#### 7.4.2 回踩位置 (权重 28%) ⭐ 核心

```python
def score_pullback(close, ma10, ma20, high_20d, low_20d):
    dist_ma10 = (close - ma10) / ma10        # 理想: -2%~+1%
    dist_ma20 = (close - ma20) / ma20        # 理想: 0%~+5%
    range_pos = (close - low_20d) / (high_20d - low_20d)  # 理想: 40%~65%
    pullback_depth = (high_20d - close) / high_20d         # 理想: 3%~8%

    return weighted_avg([
        (bell_curve(dist_ma10, -0.005, 0.03),  0.30),
        (bell_curve(dist_ma20,  0.02,  0.05),  0.25),
        (bell_curve(range_pos,  0.50,  0.25),  0.20),
        (bell_curve(pullback_depth, 0.05, 0.06), 0.25),
    ])
```

#### 7.4.3 量价确认 (权重 22%)

| 指标 | 理想 | 评分 |
|------|------|------|
| 回调缩量 | 近3日量递减 + 量比<0.8 | 缩量=1.0 |
| 量比区间 | 0.5~1.5 | 区间内=高分 |
| 下跌缩量比 | 下跌日均量 / 上涨日均量 < 0.7 | 越小=卖压越轻 |
| 底部放量 | 回踩低点量比>1.5 | 加分(抄底信号) |

#### 7.4.4 分歧后修复结构 (权重 16%)

> 日频数据无法判断盘中弱转强/封板质量/竞价承接，只能识别跨日模式。

```python
def score_post_divergence_repair(daily_10d):
    signals = {
        'had_divergence':     # 近10日有大振幅日(>6%) 或 长上影日
        'held_support':       # 分歧后未跌破 MA20 (容差3%)
        'volume_contracted':  # 后3日均量 < 分歧日量的60%
        'repair_candle':      # 最近2日有阳线 + 量温和放大
    }
    return 0.25*a + 0.30*b + 0.25*c + 0.20*d
```

#### 7.4.5 风险度量 (权重 12%)

| 指标 | 安全=高分 |
|------|----------|
| ATR% (14日) | <3%=1.0, 3-5%=0.7, >8%=0.1 |
| 近5日最大回撤 | <3%=1.0, >10%=0.2 |
| 上影线密度（使用 `upper_reversal_ratio`） | 0天=1.0, ≥3天=0.1 |
| 一字板次日风险 | 非一字=1.0, 一字涨停=0.2 |

> **v2.1 权重归一**: 0.22 + 0.28 + 0.22 + 0.16 + 0.12 = **1.00**
> （原 v2.0 为 0.20+0.25+0.20+0.15+0.10=0.90，去掉 market_score 后忘记重新归一）

#### 7.4.6 大盘环境 (仅乘数调节，不作为独立评分维度)

```python
def compute_market_regime(index_data_20d):
    if 牛市信号: return 'bull', 1.10
    if 熊市信号: return 'bear', 0.75
    return 'neutral', 1.00
```

### 7.5 综合评分

```python
raw_score = (
    trend_score     * 0.22 +
    pullback_score  * 0.28 +
    volume_score    * 0.22 +
    repair_score    * 0.16 +
    risk_score      * 0.12
    # 权重和 = 1.00 ✓
    # 大盘环境不作为加法项，只作为乘数
) * 100

timing_score = clamp(raw_score * market_multiplier, 0, 100)
```

### 7.6 动作映射

| timing_score | action | 含义 |
|-------------|--------|------|
| ≥ 80 | `setup_ready` | 多维共振，观察窗口较好 |
| 65~79 | `watch` | 接近观察条件，密切跟踪 |
| 45~64 | `wait` | 趋势对但位置未到，等回踩 |
| < 45 | `avoid_chase` | 条件不满足或过热 |

**硬规则**:
- 最新日一字涨停 → 强制 `avoid_chase`，无论分数多高
- watch_only 候选（如果配置允许计算）→ `action` 最高 `watch`

### 7.7 关键参考价位

```python
# 支撑区间: MA10~MA20 附近 (容差1%)
support_zone_low  = min(ma10, ma20) * 0.99
support_zone_high = max(ma10, ma20) * 1.01
support_basis     = "ma10" if abs(close-ma10) < abs(close-ma20) else "ma20"

# 失效位: MA20 下方 1.5 倍 ATR
invalidation_level = ma20 - 1.5 * atr

# 压力位: 近20日高点
resistance_1 = max(daily[-20:].high)

# 参考盈亏比
ref_reward_risk = (resistance_1 - close) / (close - invalidation_level)
```

**价位置信度**（v2.1 新增）：

```python
def compute_level_confidence(close, ma10, ma20, atr, high_20d, low_20d,
                              latest_is_limit_board, limit_board_count):
    """评估参考价位的可靠性。"""
    reasons = []

    # 价格远离均线 → 均线支撑参考价值降低
    if abs(close - ma20) / ma20 > 0.12:
        reasons.append("price_far_from_ma20")

    # 高波动 → ATR 计算的失效位波动大
    if atr / close > 0.06:
        reasons.append("high_volatility")

    # 连板结构 → 均线尚未跟上，支撑位意义不大
    if limit_board_count >= 2:
        reasons.append("consecutive_limit_up")

    # 箱体不清晰（20日高低差太小）
    if high_20d > 0 and (high_20d - low_20d) / high_20d < 0.05:
        reasons.append("narrow_range")

    # 近20日高点太近 → 压力位=当前位置，盈亏比失真
    if high_20d > 0 and (high_20d - close) / high_20d < 0.02:
        reasons.append("near_resistance")

    if len(reasons) >= 2:
        return "low", reasons
    elif len(reasons) == 1:
        return "medium", reasons
    else:
        return "high", []
```

### 7.8 Experimental 阶段

- 默认 `enable_setup_timing=False`
- 开启后输出 CSV 但 **Discord 中仅作为独立附件，不放入主推送 Embed**
- **Phase 1 评估框架验证后**，确认 timing_score 分层有信息量（setup_ready 组 MFE > watch 组），再正式推送

### 7.9 工作量: ~6h（含 level_confidence + tradeable-only 增量）

---

## 八、Phase 5: daily_runner v2 全流程集成

### 8.1 流程

```python
def main():
    load_env()
    run_date = resolve_complete_trade_date()

    # ── Step 0: Universe ────────────────────
    universe = UniverseBuilder(token).build(run_date)

    # ── Step 1: Stage 1 ────────────────────
    exit_code, elapsed = run_screener(universe.codes, run_date, ...)

    # ── Step 2: Setup Timing (if enabled) ──
    if enable_setup_timing:
        # 只对 tradeable 运行（不含 watch_only）
        timing = run_setup_timing(tradeable_stocks, run_date)

    # ── Step 3: Discord ────────────────────
    send_full_report(...)

    # ── Step 4: Archive ────────────────────
    archive(...)
```

### 8.2 交易日解析增强（v2.1 修订）

```python
# 默认数据就绪时间: 17:30 CST (= 收盘后2小时)
DEFAULT_DATA_READY_HOUR_CST = 17
DEFAULT_DATA_READY_MINUTE_CST = 30

def resolve_complete_trade_date(run_date=None, allow_partial=False) -> TradeDate:
    """解析最近完整交易日.

    规则:
    - 交易日 data_ready_time(17:30 CST) 后 → 当天
    - 交易日 15:30~17:30 CST:
      - allow_partial=True  → 当天（标记 partial_data_risk=True）
      - allow_partial=False → 上一个交易日（默认，更安全）
    - 交易日盘中(<15:30 CST) → 上一个交易日
    - 非交易日(周末/节假日) → 最近的完整交易日
    - 使用 Tushare trade_cal，缓存30天
    - trade_cal 不可用时 fallback 到纯周末检查

    返回:
        TradeDate(
            trade_date_used: str,
            run_datetime: str,          # ISO 8601
            data_ready_policy: str,     # "default_17:30" / "partial_allowed" / "explicit"
            partial_data_risk: bool,    # True if 15:30~17:30 + allow_partial
        )
    """
```

> **v2.1 关键变更**: 默认 data_ready_time 从 15:30 改为 **17:30 CST**。
> Tushare 的 daily_basic、moneyflow、margin_detail 等接口数据通常在 16:00~17:00 才完全写入。
> 15:30 跑大概率拿到不完整数据。
> CLI 新增 `--allow-partial-current-day` 旗标（对应 `allow_partial=True`）。

### 8.3 Discord 推送改进（v2.1 修订）

**消息 1: 概览 Embed**
```
🔥 A股热点筛选日报 — 2026-04-22
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Universe: 1,523 只 (底仓1,241 + 动态282)
🔍 通过硬筛: 1,048 只 | ✅ Tradeable: 19 只 | 👁️ Watch: 4 只
⏱️ 耗时: 57.3 分钟
🌐 大盘: neutral (沪深300 近5日+0.3%)
📡 涨停池✅ 龙虎榜✅ 行业热度✅ 概念热度✅ 观察时机🧪
📋 数据策略: default_17:30 | partial_risk: No
```

**消息 2A: Tradeable Candidates（主推送）**
```
📈 Tradeable 候选 (19只, 按总分排序)

1. 兴森科技(002436) — ⭐ 0.7855
   HT=0.78 TF=0.73 LE=0.65 RC=0.91 | 元器件
   🔍 结构确认: 板块共振 | 扩散: 32% | 位置: capacity_core_like
   📎 概念映射: PCB, AI算力
   ⚠️ 减持flag(29.7%) | 偏离MA10=3.9%
   📊 来源: csi500 + zt_pool
...
```

**消息 2B: Watch-Only（高辨识度/题材锚点）**
```
👁️ Watch-Only 高辨识度 (4只, 仅观察不追高)

1. XX科技(600XXX) — ⭐ 0.7420 | ⚠️ 一字涨停
   等板开后观察 | 板块龙头锚点
   📎 概念映射: 半导体, 先进封装

2. YY股份(002YYY) — ⭐ 0.7180 | ⚠️ crowding_cap=0.65
   高位拥挤风险 | 等分歧后修复
```

> **v2.1 变更**: tradeable 和 watch_only 分开展示。
> watch_only 强调：强辨识度 / 题材锚点 / 不适合追高 / 等分歧或板开。

**消息 3: 观察时机（experimental 阶段为独立附件）**

正式推送后：
```
⏰ 观察时机评估 (3~5天 / 低吸回踩)

🟢 高关注 (≥80)
  002436 兴森科技 82分 | 支撑25.8~26.5(MA10) [置信:高] | 失效位24.2 | 盈亏比2.1
  → 缩量回踩MA10企稳，量比0.7x

🟡 可观察 (65~79)
  600118 中国卫星 71分 | 支撑42.0~43.5(MA20) [置信:中] | 失效位39.8
  → 回踩MA20+修复阳线，量略偏大

⏳ 等待 (45~64): 科华数据(53), ...
⛔ 避免追高 (<45): ...
```

**消息 4: CSV 附件**
- `{run_date}_stage1_hot_summary.csv`
- `{run_date}_setup_timing.csv`（如启用）
- **附件上传失败不阻塞 runner**，降级为推送本地路径

### 8.4 工作量: ~4h

---

## 九、实施计划

### 9.1 Phase 总览

| Phase | 内容 | 预估 | 依赖 | 交付物 |
|-------|------|------|------|--------|
| **0** | v1 准确度修复 | 4h | 无 | 上影线统一 + LE2真实换手 + watch_only(方案A) + coverage拆分 + HT5 degraded cap |
| **1** | 评估框架 | 4.5h | Phase 0 | evaluation harness + 全量标签生成 + 消融实验 |
| **2** | Universe 动态构建 | 3h | Phase 0 | universe_builder + 来源标记 + 行业过滤 + 月更底仓 |
| **3** | HT8/9/10 experimental | 6.5h | Phase 2 | 3个新 context score (3-flag 控制) + 降级/辅助 |
| **4** | 观察时机 experimental | 6h | Phase 0 | setup_timing (tradeable-only + level_confidence) |
| **5** | daily_runner v2 集成 | 4h | Phase 2+3+4 | 全流程自动化 + 17:30数据策略 + 分开推送 |
| **总计** | | **~28h** | | |

### 9.2 执行顺序

```
Phase 0 (准确度修复, 4h)
  └── 必须先完成，修好地基
         ↓
Phase 1 (评估框架, 4.5h)       Phase 2 (Universe, 3h)
  └── 回测基础设施                └── 扩大覆盖面
         ↓                              ↓
Phase 3 (HT8/9/10, 6.5h)      Phase 4 (观察时机, 6h)
  └── experimental 输出            └── experimental 输出
         ↓                              ↓
         └──────── Phase 5 (集成, 4h) ──────┘
                     └── 全流程串联
                            ↓
                   [回测验证 2~4 周]
                            ↓
                   HT8/9/10 正式纳入总分
                   观察时机正式进入 Discord 主推送
```

Phase 1 和 Phase 2 可以并行（无依赖）。Phase 3 和 Phase 4 也可以并行。

---

## 十、验收标准

### 10.1 功能性验收

| 模块 | 条件 |
|------|------|
| Phase 0 | 现有 785 tests 全部通过 + 新增 ≥18 tests；LE2 优先 turnover_rate_f；`pass_stage1` = tradeable only；watch_only 正确分类；HT5 degraded cap 生效 |
| Phase 1 | 可对历史 run 的**全量 scored stocks** 生成 MFE/MAE 标签；pass vs fail 分离度可计算；消融实验可运行 |
| Phase 2 | 成功构建 ≥1200 只候选池；来源标记正确；降级模式可运行；Level 1 失败正确终止 |
| Phase 3 | HT8/9/10 输出合理；3-flag 独立控制有效；HT8 Level3缺失 cap 0.60；HT10 Level2缺失 cap 0.80；≥30 新 tests |
| Phase 4 | timing_score 分布合理；权重和=1.00；avoid_chase 正确拦截一字涨停；level_confidence 输出合理；只对 tradeable 运行；≥25 新 tests |
| Phase 5 | 端到端运行；data_ready_time 17:30 生效；Discord tradeable/watch_only 分开展示；附件失败不崩溃 |

### 10.2 结果性验收（回测后）

| 指标 | 条件 |
|------|------|
| Universe 捕获率 | 动态 Universe 对未来 5 日涨停的捕获率 > 静态底仓 |
| HT8/9/10 有效性 | 加入后 Top 20 的 3/5 日 MFE 不劣于 v1 |
| **分数排序单调性** | total_score 十分位分组，MFE 递减 |
| **pass vs fail 分离度** | pass_tradeable 组 MFE 显著高于 failed_score 组 |
| 观察时机分层 | setup_ready 组 MFE > watch > wait；setup_ready 组 MAE 不显著高于基线 |
| 大盘调节 | bear regime 下 setup_ready 信号数量明显收缩 |
| watch_only 验证 | watch_only 组经常包含题材锚点（涨停池核心股） |

---

## 十一、显式排除（不做的事）

1. ❌ 不接入 LLM 做自动 Stage 2（性价比低、可靠性差）
2. ❌ 不做分钟级实时数据（日频足够，避免复杂度爆炸）
3. ❌ 不做自动下单（只提供信号，人工确认）
4. ❌ 不做 NLP 公告分析（Tushare 无公告文本接口）
5. ❌ **不让 Stage 1 冒充 Stage 2**——结构化市场信号 ≠ 催化真实性判断
6. ❌ **不让 HT6/ths_member 冒充公司真实受益判断**——概念成分映射 ≠ 公司实际相关
7. ❌ **不让观察时机模块输出直接交易建议**——输出观察信号和参考价位，不输出买卖指令
8. ❌ **不在没有回测验证的情况下，将 HT8/9/10 或观察时机信号正式纳入主排序/主推送**
9. ❌ **不在没有降级路径的情况下依赖高权限接口**——所有 Level 2/3 数据源必须有 fallback；Level 1 失败需区分可降级（index_weight/top_list）和不可降级（trade_cal/stock_basic/daily）

---

## 附录 A: v2.1 修订对照表

| # | 修订项 | 来源 | 变更摘要 |
|---|--------|------|---------|
| 1 | P0-E: HT5 degraded cap | Review 阻塞#1 | 新增：降级模式子分 cap 0.80 |
| 2 | pass_stage1 语义 | Review 阻塞#2 | `pass_stage1` = tradeable only；新增 `pass_stage1_watch`/`pass_stage1_any` |
| 3 | 评估样本范围 | Review 阻塞#3 | 标签生成覆盖全量 scored stocks，不仅 pass 组 |
| 4 | Setup Timing 权重 | Review 阻塞#4 | 0.90 → 1.00（重新归一：0.22/0.28/0.22/0.16/0.12） |
| 5 | data_ready_time | Review 阻塞#5 | 15:30 → 17:30 CST；新增 `--allow-partial-current-day` |
| 6 | 上影线双指标 | Review 非阻塞#6 | 新增 `upper_reversal_ratio`（风险）与 `upper_wick_ratio`（形态）并存 |
| 7 | Universe 上市天数 | Review 非阻塞#7 | 60天硬编码 → `universe_min_listing_days` 配置项（默认 20） |
| 8 | Level 1 可用性 | Review 非阻塞#8 | 删除"始终可用"；区分可降级/不可降级 |
| 9 | context_scores flag | Review 非阻塞#9 | 单一 flag → `compute` / `use_in_total` / `show_in_discord` 三个 |
| 10 | HT8 降级 | Review 非阻塞#10 | Level 3 缺失时 cap 0.60，禁止"板块共振确认" |
| 11 | HT9 辅助输出 | Review 非阻塞#11 | 新增 `sector_amount_breadth_ratio` / `top5_amount_concentration` |
| 12 | HT10 降级 | Review 非阻塞#12 | Level 2 缺失时重新归一 + cap 0.80 + confidence=medium |
| 13 | Setup Timing 范围 | Review 非阻塞#13 | 默认只对 tradeable 运行；watch_only 最高 action=watch |
| 14 | 价位置信度 | Review 非阻塞#14 | 新增 `level_confidence: high/medium/low` |
| 15 | Discord 展示 | Review 非阻塞#15 | tradeable / watch_only 分成两段展示 |
| 16 | 行业过滤 | Review P0-H 移入 | Phase 2 新增 `universe_excluded_industries` 配置项 |
| 17 | 文档状态 | Review 建议#5 | "终版，可执行" → "v2.1 修订版" |

### 未采纳项及原因

| 建议项 | 来源 | 不采纳原因 |
|--------|------|-----------|
| P0-A 收紧涨停proxy | Review 阻塞#1 | Session 2 已完成双重验证（OHLC+pct_chg），7个测试覆盖 |
| P0-D 小样本百分位细化 | Review 阻塞#1 | Session 2 已完成三档机制（pool≥30/2-29/\<2），8个测试覆盖 |
| P0-H 金融行业排除 | Review 阻塞#1 | 策略决策而非准确度修复，移入 Phase 2 作为配置项 |
| P0-I 涨停池信号纳入HT | Review 阻塞#1 | 新评分维度而非修复，与 Phase 3 HT8 重叠，归入 Phase 3 统一处理 |
