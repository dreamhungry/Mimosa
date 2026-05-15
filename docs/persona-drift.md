# persona-drift

> Research artifact for the empirical study of personality drift in
> self-evolving LLM companion agents.
>
> 仓库定位：Mimosa 产品仓的论文版精简副本。本文档**只**聚焦 persona-drift
> 这个研究课题与其工程实现，**不**讨论 Mimosa 的产品形态。
>
> 状态：设计阶段，待落地为独立仓库。

---

## 1. 一句话定位

**`persona-drift`** = 一个最小可复现的命令行实验工具，用于研究 LLM-based
companion agent 在长程对话中自演化人格时的 **bias / collapse / reversibility**
三类现象。

- 论文类型：Empirical phenomenon study（非 benchmark）
- 论文气质：参考 *Sycophancy in LLMs* / *The Reversal Curse*
- 输出形态：论文 + 数据 + 复现脚本，作为 supplementary material

---

## 2. 研究问题（与 Mimosa 共享，但在此独立陈述）

### 主问题（Main RQ）

> Does self-evolving personality in LLM-based companion agents drift
> faithfully, or is it systematically biased, collapsed, or irreversibly
> shaped by interaction?

### 三个子问题

| ID  | 问题                              | 对立假说 / 关注点                              |
| --- | --------------------------------- | ---------------------------------------------- |
| RQ1 | Mirror vs Intrinsic               | 漂移由用户带跑（H_mirror）还是 LLM prior 偏置（H_intrinsic）？ |
| RQ2 | Mode Collapse                     | 不同 baseline 是否都向 RLHF "helpful & harmless" 收敛？ |
| RQ3 | Monotonicity & Reversibility      | 漂移是否单调？被一类用户塑造后能否被另一类用户"治愈"？ |

详细假说与判别准则参见 [`research_questions.md`](./research_questions.md) 第 2 节。

---

## 3. 范围（Scope）

### 包含（in scope）

- LLM client（OpenAI 兼容接口）
- ChatHistory（对话历史持久化）
- LongTermMemory（长期记忆抽取 / 检索）
- PersonalityManager（Big Five 状态 + 持久化）
- PersonalityEvolver（自演化机制）
- SimulatedUser（LLM 扮演的用户，多种风格）
- ExperimentRunner（批量 session 跑批 / 并行）
- LLM-as-Judge（评分模块，多模型 + inter-judge agreement）
- Analyzer（drift 距离 / 终态方差 / 可逆性比 R 等指标计算）

### 排除（out of scope）

- ❌ ASR / TTS / VAD / Live2D / Web UI
- ❌ FastAPI / Uvicorn / WebSocket
- ❌ GPU 依赖（torch / sherpa / funasr）
- ❌ 真人 user study（IRB 不可控）
- ❌ benchmark leaderboard（不是本论文目标）

> **硬约束**：`pip install -e .` 应在 < 10 秒内完成；不依赖 GPU。

---

## 4. 仓库结构（设计版）

```
persona-drift/
├── README.md                       # 项目介绍 + 复现指南
├── pyproject.toml                  # 最小依赖：openai, pydantic, pyyaml, loguru
├── src/
│   └── persona_drift/
│       ├── core/                   # 从 Mimosa 拷贝的核心模块（模式 3 同步）
│       │   ├── llm_client.py
│       │   ├── chat_history.py
│       │   ├── long_term_memory.py
│       │   ├── personality_manager.py
│       │   └── personality_evolver.py
│       ├── agent/
│       │   └── agent_core.py       # 把 core 拼成可对话的 agent
│       ├── simulated_user/
│       │   ├── base.py             # SimulatedUser 接口
│       │   └── personas/           # 5 种用户风格 prompt
│       │       ├── cold.yaml
│       │       ├── aggressive.yaml
│       │       ├── anxious.yaml
│       │       ├── warm.yaml
│       │       └── neutral.yaml
│       ├── experiment/
│       │   ├── runner.py           # 单 condition 跑批
│       │   ├── batch.py            # 矩阵跑批 + 并行
│       │   └── conditions/         # 实验配置 yaml
│       │       ├── rq1_mirror_vs_intrinsic.yaml
│       │       ├── rq2_mode_collapse.yaml
│       │       └── rq3_reversibility.yaml
│       ├── eval/
│       │   ├── llm_judge.py        # 多模型 LLM-as-judge
│       │   ├── big_five_judge.py   # 从对话推断 Big Five
│       │   └── metrics.py          # drift L2 / KL / variance / R
│       └── analyze/
│           ├── trajectory.py       # 轨迹图
│           ├── distribution.py     # 终态分布图
│           └── report.py           # 自动生成 report 表格
├── scripts/
│   ├── run_hello_world.py          # 1 char × 1 user × 50 turns
│   ├── run_rq1.py
│   ├── run_rq2.py
│   └── run_rq3.py
├── data/
│   ├── characters/                 # baseline 人格配置
│   │   ├── neutral.yaml
│   │   ├── high_extra.yaml
│   │   ├── low_extra.yaml
│   │   ├── high_neuro.yaml
│   │   └── low_neuro.yaml
│   └── results/                    # 跑出来的 jsonl + 图（git 跟踪小文件）
└── tests/
    └── test_smoke.py               # 跑通 hello-world 即算通过
```

---

## 5. 关键接口设计（最小骨架）

### 5.1 `AgentCore`

```python
class AgentCore:
    """A pure-text agent: LLM + history + memory + personality + evolver.
    No ASR, TTS, or any media dependency.
    """
    def __init__(self, character_config: dict, llm_config: dict): ...

    def chat(self, user_input: str) -> str:
        """One turn: read history+memory+personality, call LLM, save."""

    def evolve(self) -> dict:
        """Trigger personality evolution, return delta."""

    def snapshot(self) -> dict:
        """Return full state (personality + memory + history pointer)."""
```

### 5.2 `SimulatedUser`

```python
class SimulatedUser:
    """LLM-driven user with a fixed persona prompt."""
    def __init__(self, persona_yaml: str, llm_config: dict): ...

    def reply(self, agent_output: str) -> str:
        """Read agent's last turn, produce next user message."""
```

### 5.3 `ExperimentRunner`

```python
@dataclass
class Condition:
    character: str        # baseline name
    user_persona: str     # simulated user style
    n_turns: int
    n_repeats: int
    seed: int
    evolve_every: int     # how often to trigger evolution

class ExperimentRunner:
    def run(self, condition: Condition) -> ExperimentResult:
        """Run n_repeats sessions, return trajectories + final states."""
```

### 5.4 实验配置文件示例

```yaml
# conditions/rq1_mirror_vs_intrinsic.yaml
name: rq1_mirror_vs_intrinsic
characters: [neutral, high_extra, low_extra]
user_personas: [cold, aggressive, anxious, warm, neutral]
n_turns: 100
n_repeats: 5
evolve_every: 10
seeds: [42, 43, 44, 45, 46]
judge_models: [gpt-4o, claude-3.5-sonnet, gemini-1.5-pro]
```

---

## 6. 实验设计（实现层面）

### 6.1 RQ1：Mirror vs Intrinsic

```
矩阵: 3 baseline × 5 user_persona × 5 repeat = 75 sessions
每 session: 100 turns, 每 10 turn 触发一次 evolve

输出:
  - per-session: trajectory.jsonl (Big Five over time)
  - aggregate: drift_direction_per_(baseline, user) heatmap
  - control: neutral user × all baselines → 测 LLM prior 漂移方向

判别:
  - 不同 user_persona 的漂移方向显著不同 → 偏向 H_mirror
  - neutral control 也有显著漂移且方向稳定 → 偏向 H_intrinsic
  - 用方差分解（user 主效应 vs prior 主效应）量化两种力量权重
```

### 6.2 RQ2：Mode Collapse

```
矩阵: 5 baseline (差异极大) × 1 user (neutral) × 5 repeat = 25 sessions
每 session: 100 turns

测量:
  - var(终态 Big Five) vs var(baseline Big Five)
  - 若终态方差 << baseline 方差 → 塌缩

可视化:
  - Big Five 5 维 PCA，看 baseline 点群 → 终态点群是否塌缩
```

### 6.3 RQ3：Reversibility

```
两阶段实验:
  Phase A: agent (baseline=X) × user=aggressive × 50 turns
           → ΔA1 = baseline_agree - phase_A_end_agree
  Phase B: 继承 Phase A 终态 × user=warm × 50 turns
           → ΔA2 = phase_B_end_agree - phase_A_end_agree

可逆性指标: R = ΔA2 / |ΔA1|

矩阵: 3 baseline × 5 repeat × 2 phase = 30 long sessions
```

---

## 7. 评测方法

### 7.1 客观指标（脚本可跑）

| 指标             | 公式                                                      | 目的         |
| ---------------- | --------------------------------------------------------- | ------------ |
| Drift L2         | `‖p_final - p_baseline‖_2`                                | 总漂移量     |
| Drift KL         | KL(N(p_final) ‖ N(p_baseline))                            | 分布漂移     |
| Final variance   | `Var(p_final)` over baselines                             | 测 collapse  |
| Reversibility R  | `ΔA_recovery / |ΔA_perturbation|`                         | 测可逆性     |
| Self-BLEU        | session 内 agent 回复的 BLEU                              | 风格一致性   |
| Topic entropy    | LDA topic 分布的香农熵                                    | 话题广度     |

### 7.2 LLM-as-Judge（带 inter-judge agreement）

- 评委：GPT-4o + Claude-3.5-Sonnet + Gemini-1.5-Pro
- 量表：1-5 Likert
- 维度：personality consistency / memory groundedness / perceived empathy
- 必报指标：Cohen's κ 或 Krippendorff's α

### 7.3 双通道 Big Five 测量

> 与 [llm-persona-gap](https://github.com/dreamhungry/llm-persona-gap) 对接。

- **Channel A**：直接读 PersonalityManager 的 self-reported state
- **Channel B**：让 judge 从对话推断 behavioral Big Five
- 两通道差异本身就是 drift faithfulness 的另一个度量

---

## 8. 与 Mimosa 的同步策略（模式 3）

```
方向 1：Mimosa → persona-drift（设计阶段，主流方向）
  - Mimosa.src.mimosa.{llm, conversation/chat_history,
    memory/long_term_memory, personality/*}
    → persona-drift.src.persona_drift.core/*
  - 手动 copy，不用 git submodule、不用 pip install

方向 2：persona-drift → Mimosa（论文发完后）
  - 仅 cherry-pick 已通过 sanity check 的产品价值改动
  - 实验性 prompt 变体、消融用代码不回流

不同步的部分（persona-drift 独有）:
  - simulated_user/
  - experiment/
  - eval/llm_judge.py
  - analyze/

不同步的部分（Mimosa 独有）:
  - ASR / TTS / VAD / Live2D
  - FastAPI / WebSocket
  - service_context.py
  - frontend/
```

**同步触发时机**：
- evolver 的核心算法在 Mimosa 改动后，手动 patch 到 persona-drift（约 30 分钟）
- 反之，persona-drift 论文阶段的 prompt 实验**不**自动回流

---

## 9. 实施路径

```
M0  Bootstrap（1-2 天）
  ├─ 创建 persona-drift 仓库（GitHub, MIT/Apache）
  ├─ pyproject.toml 最小依赖
  ├─ 从 Mimosa 拷贝 core 模块
  ├─ 写 AgentCore 包装层
  └─ 跑通 hello-world：1 char × 1 user × 50 turns，输出 jsonl

M1  RQ1 Pilot（1 周）
  ├─ 5 种 SimulatedUser persona prompt
  ├─ ExperimentRunner（先单进程，再加并行）
  ├─ 跑 RQ1 小规模（3 baseline × 5 user × 1 repeat）
  └─ Analyzer 出第一张 drift direction 图

M2  RQ1/2/3 Full Run（2-3 周）
  ├─ 全量跑 RQ1/2/3
  ├─ LLM-as-judge + inter-judge agreement
  └─ 数据归档到 data/results/

M3  Writing（2 周）
  ├─ Technical report → workshop / arXiv
  ├─ Project page
  └─ README 复现指南打磨

M4  Submission（视投稿窗口）
  └─ 投 EMNLP / ACL / NeurIPS / ICLR workshop
```

---

## 10. 开放问题（同步自 Mimosa 主文档，本仓库视角再问一次）

- [ ] simulated user 的 prompt 是否要加 "memory of own persona"，避免 LLM 在长对话里跑题？
- [ ] evolution_history.jsonl 字段够用吗？需要补 user_persona / condition_id 吗？
- [ ] Big Five 测量的 dual-channel：实现成本？
- [ ] 是否引入 "evolution disabled" 的 control agent？
- [ ] data/results/ 多大？要不要 git-lfs？
- [ ] 复现成本：N 次 LLM 调用 × 单价 = ?（需要在 README 给出预估）

---

## 11. 与 Mimosa 主文档的关系

- 研究问题、假说、判别准则 → 共享，详见 [`research_questions.md`](./research_questions.md)
- 仓库决策、命名、路线选择 → 共享，详见 [`research_questions.md`](./research_questions.md) 第 7 / 8 节
- 工程实现、接口设计、目录结构、实施路径 → **本文档独有**

> 一句话：`research_questions.md` 回答 *研究什么 / 为什么*；
> 本文档回答 *用 persona-drift 仓库怎么做*。
