# Mimosa 研究问题与思路

> 本文档记录 Mimosa 项目从"工程 demo"升级为"research prototype"过程中梳理出来的核心研究问题、子问题、实验思路与跨项目联动设计。
>
> 目标读者：项目作者本人（PhD 申请前的研究规划） + 潜在合作者 / 套磁导师
>
> 状态：思路阶段，待逐步落地为实验、analyzer 与 technical report

---

## 0. 项目研究主张（Research Thesis）

> **LLM agent personality faithfulness**
>
> 围绕 "LLM-based companion agent 的人格在长期交互中是否忠实、可控、可解释" 这一研究问题，
> Mimosa 与 [llm-persona-gap]([../../llm-persona-gap](https://github.com/dreamhungry/llm-persona-gap)) 共同构成一个研究纲领：
>
> - **llm-persona-gap**：横截面研究 —— 某一时刻，agent **声明的人格 vs 行为表现的人格** 是否一致？
> - **Mimosa**：纵向研究 —— 当 agent 被允许通过 self-reflection **演化人格** 时，演化轨迹是否忠实于用户、是否被 LLM prior 偏置、是否会塌缩或不可逆？

两者组合：
```
                  ┌──────────────────────────────────────────┐
                  │  LLM Agent Personality Faithfulness      │
                  └──────────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
     llm-persona-gap                       Mimosa
   (cross-sectional)                   (longitudinal)
   stated vs behavioral             drift bias / collapse /
   personality gap                       reversibility
```

---

## 1. 核心研究问题

### 主问题（Main RQ）

> **Does self-evolving personality in LLM-based companion agents drift faithfully, or is it systematically biased, collapsed, or irreversibly shaped by interaction?**

注意三个关键词：
- **bias**：演化方向是否被某种力量（用户风格 / LLM prior）系统性偏置
- **collapse / symmetry**：不同 baseline、不同用户风格是否对称影响，或全部塌缩到同一终态
- **reversibility**：被一类用户塑造的人格，能否被另一类用户"治愈"

这三个词构成本项目的研究骨架。

### 反例澄清（重要）

❌ "agent 人格会不会变？" —— 不是研究问题（机制本身就让它变）
❌ "我做了一个会演化人格的 agent" —— 不是研究问题（这是工程动作）
✅ "演化以什么方式发生、是否合理、能不能控制" —— 才是研究问题

> 关键洞察：**做机制的人 ≠ 研究机制的人**。
> 99% 的 self-evolving agent 项目只是"加个 self-reflection 然后 demo"，没有人系统测过 bias / collapse / reversibility。这就是空白。

---

## 2. 子问题（Sub-RQs）

### RQ1. 漂移是被用户带跑（mirror）还是 agent 内禀偏置（intrinsic）？

两个对立假说：

```
H_mirror: 用户镜像假说
  用户外向 → 对话热烈 → 反思看到"对话热烈"
  → agent 调高 extraversion → 进一步迎合用户
  结果: agent 变成用户的镜子，没有"自我"

H_intrinsic: 内禀演化假说
  无论用户什么风格，agent 都朝某个方向漂移
  （比如所有 agent 最后都变得 agreeableness↑、neuroticism↓）
  结果: LLM 反思机制本身有系统性 prior bias
```

**两种结果都是 bug，但表现完全相反**，且都是 publishable 的发现。

#### 实验设计骨架

| 用户风格 (LLM 模拟) | 重复 session 数 | 关注维度 |
|---|---|---|
| 冷淡寡言 | 10 | extraversion 怎么变？|
| 攻击性 | 10 | agreeableness 怎么变？|
| 焦虑诉苦 | 10 | neuroticism 怎么变？|
| 中性闲聊（control） | 10 | 应该不怎么变，作为 baseline 检测 LLM prior |

判别准则：
- 若不同用户风格对应不同漂移方向 → 偏向 H_mirror
- 若中性 control 也漂移、且漂移方向与用户风格无关 → 偏向 H_intrinsic
- 大概率是混合，可量化两种力量的权重

---

### RQ2. 人格演化是否会"塌缩"（mode collapse）？

LLM-as-judge / self-reflection 的经典毛病：朝 LLM 的 RLHF 偏好收敛。

具体猜想：
- 不管 baseline 是什么人格，跑 N 轮后是否所有 agent 都向 "高 agreeableness、低 neuroticism、中等 openness" 收敛？
- 这是 RLHF 模型 "helpful & harmless 单一人格" 问题在 personality evolution 中的复现。

#### 实验设计

- 5 个差异极大的 baseline 人格（高 / 低 extraversion 各一对、高 / 低 neuroticism 各一对、中性）
- 每个 baseline × 100 轮中性闲聊 × 重复 5 次
- 看终态 Big Five 分布

**预期结果（待验证）**：终态分布的方差显著小于 baseline 分布的方差 → 塌缩。

若证实塌缩：
- 这是一个重要负面结论
- 顺势提出方法：Mimosa 的 `max_delta=5` 是否够？要不要加 KL 约束回 baseline？anchor regularization？

---

### RQ3. 人格演化是否单调？是否可逆？

实验流程：
```
Phase 1: agent (baseline=X) × "攻击性用户" × 50 轮
         → 测 agreeableness 下降程度 ΔA1
Phase 2: 同一 agent（继承 Phase 1 终态） × "温柔用户" × 50 轮
         → 测 agreeableness 是否回升 ΔA2

可逆性指标: R = ΔA2 / |ΔA1|
  R ≈ 1: 完全可逆
  R ≈ 0: 完全不可逆（被塑造死）
  R ∈ (0, 1): 部分可逆
```

#### 这个问题为什么重要

- HAI 真实关切：**如果一个用户把虚拟陪伴养歪了，新用户接手能"治愈"它吗？**
- 反向：**长期用户是否害怕一次糟糕对话永久改变他们的伙伴？**
- 这直接对应"chatbot 数字遗产 / 共享 / 转移"等真实产品问题。

---

## 3. 备选研究方向：Memory × Personality 耦合（Phase 2）

主方向是 RQ1-RQ3，但记忆方向有一个**只有 Mimosa 能问**的独特问题，作为 Phase 2 候选。

### 不要做什么

❌ "我设计了三层记忆架构"（episodic / semantic / affective）
理由：业界已经普遍这么分，零创新。

### 真正有研究价值的问题

#### B1. Memory selection policy 的系统比较（该记什么？）

业界现状：mem0 / LangMem 都做"提取 + 检索"，但**提取规则普遍是 LLM ad-hoc**，是个黑箱。
**没有人系统比较过不同 selection policy 对长期关系质量的影响。**

可比较的 policy：
```
P1: 显式事实提取（"用户喜欢猫"）       ← 当前主流
P2: 情感事件优先（用户哭了/笑了的时刻）
P3: 自我披露优先（用户讲了隐私就记，闲聊不记）
P4: 用户主动标记（"记住这件事"）
P5: 反思冲突点（用户纠正过 agent 的话）
```

ablation 实验跑下来，**就算结果是"P3+P5 组合最好"也是 publishable 的** —— 因为这个比较从来没人系统做过。

#### B2. Memory ↔ Personality 双向耦合（Mimosa 独家角度）

> 业界做记忆的人不做人格，做人格的人不做记忆。Mimosa 同时有，可以问只有它能问的问题。

两个方向：
- **Personality → Memory**：不同人格的 agent 应该记不同的东西吗？
  - 高 conscientiousness → 偏向记事实、约定
  - 高 agreeableness → 偏向记用户的情绪、需要
  - 高 openness → 偏向记新颖话题
- **Memory → Personality**：记忆内容会反向塑造人格演化方向吗？
  - 直觉："你记住的东西定义了你"
  - 实验：人为操纵 memory（只保留情感事件 vs 只保留事实），看人格演化轨迹是否分叉

这是一条 **memory ↔ personality 的双向耦合环**，几乎没人研究过。

---

## 4. 量化方法：不需要真人用户

PhD 申请阶段做不动 IRB / 真用户研究，使用以下三类替代手段：

### 方法 1：客观指标（脚本可跑，零主观）

```
- Personality drift L2 / KL    : baseline 与终态的数值距离
- Final-state variance          : 多 baseline 终态分布方差（测塌缩）
- Reversibility ratio R         : ΔA_recovery / |ΔA_perturbation|
- Memory recall rate            : 后续对话提到过去事实的占比
- Contradiction rate            : agent 自相矛盾次数（NLI 模型自动检测）
- Self-BLEU / topic entropy     : 风格一致性、话题广度
```

### 方法 2：LLM-as-Judge（带 inter-judge agreement）

让 GPT-4 / Claude / Gemini 多模型当评委，对一段对话打 1-5 分：
- "AI 是否记住了用户之前说过的事？"（memory groundedness）
- "AI 是否表现出对用户的理解？"（perceived empathy）
- "回应是否一致符合人格设定？"（personality consistency）

学界已普遍接受 LLM-as-judge，前提是报告 inter-judge agreement（Cohen's κ / Krippendorff's α）。

### 方法 3：Simulated Longitudinal Study

让 GPT-4 扮演一个有固定背景的虚拟用户（如 "Alice，35岁，工作压力大，养猫 Mimi"），与 Mimosa 跑 30 天 × 每天 10 轮对话。

测量：
- 第 30 天 agent 是否还记得 Mimi
- 人格是否仍与 baseline 一致
- 不同记忆 policy 下行为差异

**先例**：Park et al. *Generative Agents* (2023) 即用此方法，审稿人接受。

---

## 5. 与 llm-persona-gap 的协同

SOP 写法模板：

> Through **llm-persona-gap**, I found that LLM agents exhibit measurable gaps between **expressed** and **behavioral** personality at any single point in time.
>
> This raised a follow-up question: when such agents are allowed to **evolve** their personality through self-reflection over long-term interaction, do they drift **faithfully** toward the user, or are they systematically biased by the LLM's own priors? Do their personalities **collapse** to a common attractor regardless of baseline? Are the changes **reversible**?
>
> **Mimosa** serves as the testbed for this longitudinal extension. Together the two projects form a research program on **LLM agent personality faithfulness**.

两个项目的角色：

| 项目 | 时间维度 | 测什么 | 方法 |
|---|---|---|---|
| llm-persona-gap | 横截面 (snapshot) | stated vs behavioral 的 gap | 双通道测量、统计检验 |
| Mimosa | 纵向 (longitudinal) | drift bias / collapse / reversibility | 模拟用户、批量 session、轨迹分析 |

**两个项目并不重复，是同一研究问题的两个角度。**

---

## 6. 候选论文标题（备忘）

- *Does Self-Evolving Personality in LLM-based Companions Drift Faithfully or Collapse? An Empirical Study of Bias, Symmetry, and Reversibility*
- *Mirror or Prior? Disentangling User Influence and LLM Bias in Self-Evolving Companion Agents*
- *On the Reversibility of Personality Drift in Long-term LLM Companions*

---

## 7. 实施阶段（执行视角）

```
Phase 1  申请前必做（核心 contribution）
  ├─ RQ1 实验：5 种用户风格 × 多 baseline × 批量 session
  ├─ RQ2 实验：多 baseline × 长 session × 终态分布分析
  ├─ RQ3 实验：扰动 → 恢复 双阶段实验
  └─ Analyzer + 图表 + 2-3 页 technical report

Phase 2  申请中 / 入学初期（自然延伸）
  ├─ B1 memory selection policy ablation
  └─ B2 memory ↔ personality 双向耦合实验
```

申请材料层面：
- Phase 1 → 进 CV、写进 SOP 主体、demo video、project page
- Phase 2 → 写在 SOP 的 "Future Work / Proposed Research" 段落，作为 PhD 第一年的研究纲领

---

## 8. 关键设计决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 主方向 | RQ1-RQ3（drift faithfulness） | 与 llm-persona-gap 天然成对，纵横互补 |
| 用户研究 | 不做真人 user study（Phase 1） | IRB / 招募 / 时间成本不可控；用 simulated user + LLM-as-judge 替代 |
| 记忆架构 | 不接 mem0 | 接 mem0 是工程动作；自研 + ablation 才有研究价值；但 Phase 1 先不动这块 |
| 多模态 | 文本情绪 OR 语音 valence/arousal 二选一，不做面部 | 面部硬件依赖、隐私问题、对申请加分有限 |
| 演化约束 | RQ2 若证实塌缩，提出 KL / anchor regularization | 顺势把"发现问题"变成"提出方法" |

---

## 9. 待办与开放问题

- [ ] 设计 simulated user 的 prompt 模板（5-6 种风格）
- [ ] 设计实验配置矩阵（baseline 人格 × 用户风格 × 重复次数）
- [ ] 设计 evolution_history.jsonl 的扩展字段（是否够用？需要补什么？）
- [ ] 选定 LLM-as-judge 的具体 prompt 与评分量表
- [ ] 决定 Big Five 的测量方式：直接读 self-state vs 让 judge 从对话推断（**两者都要，正好对接 llm-persona-gap 的 dual-channel**）
- [ ] technical report 大纲（Motivation / RQ / Setup / Findings / Limitations / Future Work）
- [ ] 是否引入 control：完全 disable evolution 的 agent 作为对照组？

---

## 参考文献待补充

需要在 technical report 阶段系统梳理：
- Generative Agents (Park et al., 2023)
- Character.AI / Replika 相关 HCI 研究
- LLM personality measurement 系列（PsychoBench、PersonalityEvaluator 等）
- RLHF mode collapse / preference homogenization 相关
- Memory-augmented dialogue agents（mem0、MemoryBank、LongMemEval 等）
