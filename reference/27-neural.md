以下是 **第 27 章：Sleep Consolidator 的全神經網路化演算法詳細版**。
它的重點是：**不把 consolidation 當成外部程式流程，而是把「選擇、合併、壓縮、蒸餾、驗證、回滾」全部改成神經網路內部的可微分調控機制。**

---

# 27. Sleep Consolidator：神經網路化演算法詳細說明

## 27.0 核心思想

Sleep Consolidator 的神經網路化版本，不再是外部手寫流程：

```text
if memory full:
    merge memory
if expert too many:
    compress experts
if forgetting:
    rollback
```

而是改成：

[
\mathcal{S}_{t+1}
=================

SleepNet_\Omega(\mathcal{S}_t)
]

其中：

[
\mathcal{S}_t
=============

{
\theta_c,
M_t,
A_t,
R_t,
I_t,
P_t,
G_t,
T_t,
V_t,
B_t
}
]

Sleep Consolidator 本身是一個神經網路系統：

[
SleepNet_\Omega
===============

{
Selector,
Dreamer,
ClusterNet,
MergeNet,
VersionNet,
CompressNet,
DistillNet,
Verifier,
RollbackNet
}
]

也就是：

```text
Sleep Consolidator
= 神經選擇器
+ 神經夢境生成器
+ 神經聚類器
+ 神經記憶合併器
+ 神經版本化器
+ 神經 expert 壓縮器
+ 神經核心蒸餾器
+ 神經驗證器
+ 神經回滾控制器
```

這種設計的基礎可以借鑑 Differentiable Neural Computer 的精神：用神經 controller 讀寫記憶矩陣；learned optimizer 則說明 optimizer 本身也可以被神經網路學出來，而不是固定手寫規則。DNC 的 controller 負責讀寫記憶並產生輸出，learned optimizer 則把優化演算法設計轉為可學習問題。([Google DeepMind][1])

---

# 27.1 目的：Sleep Consolidator 要解決什麼

Sleep Consolidator 的任務不是「再訓練一次模型」，而是讓模型在休眠階段完成神經內部整理。

它要解決七個問題：

```text
1. 快速記憶太多，必須整理。
2. 相似記憶重複，必須合併。
3. 衝突知識不能覆蓋，必須版本化。
4. adapter / expert 太多，必須壓縮。
5. 舊知識可能被新知識干擾，必須 replay。
6. 穩定知識要慢慢進入核心模型。
7. 整合失敗時必須抑制或回滾。
```

神經化後，Sleep Consolidator 的目標函數是：

[
\max_\Omega
\mathbb{E}
[
LongTermPerformance
-------------------

## Forgetting

## MemoryCost

## Interference

SafetyRisk
]
]

也可以寫成最小化：

[
L_{sleep}
=========

\lambda_1 L_{replay}
+
\lambda_2 L_{distill}
+
\lambda_3 L_{forget}
+
\lambda_4 L_{compress}
+
\lambda_5 L_{conflict}
+
\lambda_6 L_{safety}
+
\lambda_7 L_{capacity}
]

核心原則：

[
\text{快速學習在 memory，慢速整合進 core}
]

也就是：

[
M_{episodic}
\rightarrow
M_{semantic}
\rightarrow
A_{expert}
\rightarrow
\theta_c
]

---

# 27.2 輸入與輸出

## 27.2.1 輸入狀態

Sleep Consolidator 的輸入是完整模型狀態：

[
\mathcal{S}_t =
{
\theta_c,
M_t,
A_t,
R_t,
I_t,
P_t,
G_t,
T_t,
V_t,
B_t
}
]

| 符號         | 神經網路化意義                          |
| ---------- | -------------------------------- |
| (\theta_c) | 核心 Transformer 權重                |
| (M_t)      | 可微分記憶矩陣                          |
| (A_t)      | LoRA / adapter / expert bank     |
| (R_t)      | Sparse neural router             |
| (I_t)      | 神經重要性圖                           |
| (P_t)      | 後設可塑性狀態                          |
| (G_t)      | Dream replay generator           |
| (T_t)      | Shadow teacher / behavior memory |
| (V_t)      | Verifier / critic                |
| (B_t)      | rollback / version buffer        |

---

## 27.2.2 輸出狀態

輸出為：

[
\mathcal{S}_{t+1}
=================

{
\theta_c',
M_t',
A_t',
R_t',
I_t',
P_t',
T_t',
B_t'
}
]

其中每個更新都不是外部規則決定，而是由神經 gate 控制：

[
\mathcal{S}_{t+1}
=================

g_{commit}
\cdot
\mathcal{S}*{candidate}
+
(1-g*{commit})
\cdot
\mathcal{S}_t
]

其中：

[
g_{commit}
==========

CommitNet_\omega(
risk,
forgetting,
safety,
compression_gain,
performance
)
]

如果 (g_{commit}) 接近 1，代表提交整合結果。
如果 (g_{commit}) 接近 0，代表拒絕整合，保留舊狀態。

---

# 27.3 何時啟動 Sleep Consolidation

外部看起來像「閒置時啟動」，但神經化後，啟動條件本身由神經網路決定。

定義 sleep gate：

[
g_{sleep}
=========

SleepGate_\omega(z_t)
]

其中：

[
z_t =
[
memory_pressure,
adapter_fragmentation,
conflict_density,
forgetting_risk,
sentinel_drift,
idle_signal,
energy_budget
]
]

輸出：

[
g_{sleep} \in [0, 1]
]

若做完全可微分版本，不使用硬判斷，而是讓 sleep 強度連續化：

[
\mathcal{S}_{t+1}
=================

(1-g_{sleep})\mathcal{S}*t
+
g*{sleep}SleepNet(\mathcal{S}_t)
]

---

## 27.3.1 SleepGate 的輸入

| 訊號                    | 神經表示                      |
| --------------------- | ------------------------- |
| memory pressure       | 記憶槽使用率 embedding          |
| adapter fragmentation | expert bank 分散度           |
| conflict density      | 衝突記憶比例                    |
| forgetting risk       | replay / sentinel loss 上升 |
| idle signal           | 系統低負載訊號                   |
| safety risk           | 最近更新風險                    |
| temporal decay        | 時間記憶過期程度                  |

---

## 27.3.2 Sleep 啟動公式

[
g_{sleep}
=========

\sigma(
W_s z_t + b_s
)
]

或更深一點：

[
g_{sleep}
=========

MLP_s(
LayerNorm(z_t)
)
]

其中：

[
g_{sleep}
=========

0
]

表示完全不 sleep。

[
g_{sleep}
=========

1
]

表示完整進入 consolidation。

---

# 27.4 Sleep Consolidator 總體神經演算法

外部流程版是：

```text
select → replay → cluster → merge → compress → distill → verify → rollback
```

神經網路化後變成：

[
\mathcal{S}_{candidate}
=======================

RollbackAwareUpdate(
DistillNet(
CompressNet(
VersionMergeNet(
ClusterNet(
DreamNet(
SelectNet(\mathcal{S}_t)
)))))))
]

更直觀地說：

```text
Algorithm: NeuralSleepConsolidator

Input:
    S_t

1. g_sleep = SleepGate(S_t)

2. Z_mem = MemoryEncoder(M_t)
3. Z_exp = ExpertEncoder(A_t)
4. Z_state = StateEncoder(S_t)

5. a_mem = NeuralMemorySelector(Z_mem, Z_state)
6. D_dream = DreamReplayGenerator(M_t, a_mem)

7. Q_cluster = ClusterNet(Z_mem)
8. M_merged = NeuralMergeNet(M_t, Q_cluster)

9. M_versioned = NeuralVersionNet(M_merged)

10. A_compressed = ExpertCompressNet(A_t, D_dream)

11. θ_candidate = CoreDistillNet(
        θ_c,
        M_versioned,
        A_compressed,
        D_dream
    )

12. I_candidate = ImportanceUpdateNet(...)
13. P_candidate = PlasticityUpdateNet(...)
14. R_candidate = RouterUpdateNet(...)

15. risk = VerifierCritic(S_candidate)

16. g_commit = CommitGate(risk)

17. S_{t+1}
    =
    g_sleep * [
        g_commit * S_candidate
        + (1 - g_commit) * Rollback(S_t)
    ]
    + (1 - g_sleep) * S_t
```

---

# 27.5 Phase 0：神經快照與可逆狀態

傳統做法會保存 checkpoint。
神經網路化後，不一定保存完整 checkpoint，而是保存可逆低秩更新軌跡。

定義 rollback state：

[
B_t =
TraceNet(
\Delta M_t,
\Delta A_t,
\Delta \theta_t,
context_t,
risk_t
)
]

其中：

[
\Delta \theta_t
\approx
U_t V_t^\top
]

也就是把更新壓縮成低秩形式：

[
\Delta \theta_t
===============

\sum_{r=1}^{R}
u_r v_r^\top
]

回滾時：

[
\theta_{rollback}
=================

## \theta_{current}

\alpha_{rollback}
\Delta \theta_t
]

其中：

[
\alpha_{rollback}
=================

RollbackNet(risk, confidence, verifier_score)
]

---

## 27.5.1 神經快照不是完整備份

神經快照包含：

```text
1. 權重 delta
2. memory delta
3. adapter delta
4. router delta
5. 更新理由 embedding
6. 更新風險 embedding
7. 可逆性分數
```

寫成：

[
b_t =
[
\Delta \theta_t,
\Delta M_t,
\Delta A_t,
\Delta R_t,
reason_t,
risk_t,
reversibility_t
]
]

---

# 27.6 Phase 1：神經選擇要整理的記憶

原本是用 priority threshold 選：

[
priority_i > \tau
]

神經化後，不用硬門檻，而是產生連續 attention 權重：

[
a_i^{sleep}
===========

SelectorNet(m_i, z_{state})
]

其中：

[
a_i^{sleep} \in [0,1]
]

代表第 (i) 個記憶在 sleep 中被處理的程度。

---

## 27.6.1 記憶編碼

每個 memory slot：

[
m_i =
[
k_i,
v_i,
confidence_i,
usage_i,
age_i,
source_i,
conflict_i,
plasticity_i
]
]

經過 memory encoder：

[
z_i
===

MemoryEncoder(m_i)
]

然後 selector 輸出：

[
a_i^{sleep}
===========

\sigma(W_a z_i + b_a)
]

---

## 27.6.2 可微分候選集合

不建立硬集合：

[
M_{cand}
========

{m_i | priority_i > \tau}
]

而是建立 soft candidate memory：

[
\tilde{M}_{cand}
================

{a_i^{sleep} m_i}_{i=1}^{N}
]

也就是每個記憶都有不同參與程度。

---

# 27.7 Phase 2：神經 Dream Replay Batch

Dream replay 不是外部 sample buffer，而是神經生成器：

[
D_{dream}
=========

G_\psi(M_t, a^{sleep}, z)
]

它可以生成三種東西：

```text
1. pseudo input：舊輸入
2. pseudo label：舊答案
3. pseudo behavior：舊 logits / activation
```

---

## 27.7.1 Dream Generator

[
z_i
===

MemoryEncoder(m_i)
]

[
\epsilon
\sim
\mathcal{N}(0, I)
]

[
\hat{x}_i,
\hat{y}_i,
\hat{p}_i
=========

DreamGenerator_\psi(z_i, \epsilon)
]

其中：

[
\hat{p}_i
]

是舊行為分布，用於 distillation。

---

## 27.7.2 神經 replay 權重

每個 dream sample 有權重：

[
w_i^{dream}
===========

ReplayWeightNet(
importance_i,
usage_i,
conflict_i,
age_i,
confidence_i
)
]

Replay loss：

[
L_{replay}
==========

\sum_i
w_i^{dream}
L(
f_\theta(\hat{x}_i),
\hat{y}_i
)
]

如果是 distillation replay：

[
L_{dream_distill}
=================

\sum_i
w_i^{dream}
KL(
\hat{p}*i
|
p*\theta(y|\hat{x}_i)
)
]

---

# 27.8 Phase 3：神經記憶聚類

傳統聚類是外部演算法，例如 k-means。
神經化後，使用可微分 assignment matrix：

[
Q
=

ClusterNet(M_t)
]

其中：

[
Q_{ik}
======

P(m_i \in C_k)
]

表示 memory slot (m_i) 屬於 cluster (k) 的軟機率。

---

## 27.8.1 可微分 cluster assignment

[
z_i
===

MemoryEncoder(m_i)
]

[
Q_{ik}
======

softmax_k(
z_i^\top c_k
)
]

其中 (c_k) 是可學習 cluster prototype。

---

## 27.8.2 Cluster representation

每個 cluster 的表示：

[
C_k
===

\frac{
\sum_i Q_{ik} z_i
}{
\sum_i Q_{ik}
}
]

這讓 cluster 本身也是神經向量。

---

## 27.8.3 Cluster 內部一致性

[
consistency_k
=============

ConsistencyNet(C_k)
]

或用 value similarity：

[
consistency_k
=============

\frac{
\sum_{i,j}
Q_{ik}Q_{jk}sim(v_i,v_j)
}{
\sum_{i,j}Q_{ik}Q_{jk}
}
]

---

# 27.9 Phase 4：神經一致記憶合併

對於一致性高的 cluster，不用手寫合併規則，而是交給 MergeNet。

[
m_k^{merged}
============

MergeNet(
{m_i},
Q_{ik}
)
]

---

## 27.9.1 Attention-based merge

每個 cluster 裡的 memory 權重：

[
\alpha_{ik}
===========

softmax_i(
MergeScoreNet(m_i, C_k)
)
]

合併 key：

[
k_k^{new}
=========

\sum_i
\alpha_{ik}
k_i
]

合併 value：

[
v_k^{new}
=========

\sum_i
\alpha_{ik}
v_i
]

合併 confidence：

[
c_k^{new}
=========

ConfidenceMergeNet(
{c_i, source_i, usage_i}
)
]

---

## 27.9.2 Soft replacement

不直接刪除舊記憶，而是用 merge gate：

[
g_k^{merge}
===========

\sigma(
W_m[
C_k,
consistency_k,
confidence_k
]
)
]

更新：

[
m_i^{t+1}
=========

(1-g_k^{merge})m_i^t
+
g_k^{merge}m_k^{merged}
]

這樣合併是可微分的。

---

# 27.10 Phase 5：神經衝突記憶版本化

若 cluster 內 key 相似但 value 不一致，不能合併。
神經化後，用 VersionNet 判斷是否建立多版本。

[
g_k^{version}
=============

VersionNet(C_k, conflict_k, time_k, source_k)
]

---

## 27.10.1 衝突分解

將 cluster 分成多個 version prototype：

[
V_{k,r}
=======

\frac{
\sum_i Q_{ik} H_{ir} z_i
}{
\sum_i Q_{ik}H_{ir}
}
]

其中：

[
H_{ir}
======

P(m_i \in version_r)
]

由 VersionNet 產生：

[
H
=

VersionNet({z_i}_{i \in C_k})
]

---

## 27.10.2 Context-aware version routing

對查詢 (q)，選擇版本：

[
\pi_r
=====

softmax_r(
q^\top W_v V_{k,r}
+
ContextScore(q, context, time, source)
)
]

輸出：

[
v^*
===

\sum_r
\pi_r v_{k,r}
]

也就是：

```text
不是覆蓋舊知識，
而是保留多個版本，
由 router 根據 context 選擇。
```

---

## 27.10.3 抑制舊版本

若某版本過期或低可信，不刪除，而是抑制：

[
inhibit_r
=========

InhibitionNet(
version_r,
time,
source,
confidence,
context
)
]

讀取時：

[
v^*
===

\sum_r
\pi_r
(1-inhibit_r)
v_{k,r}
]

---

# 27.11 Phase 6：神經 Adapter / Expert 壓縮

Adapter / expert 壓縮也可以神經化。

每個 expert：

[
E_i(h)
======

B_iA_ih
]

LoRA 本身就是把預訓練權重凍結，向 Transformer 層注入低秩矩陣，以大幅降低可訓練參數量；這很適合在持續學習架構中作為「可壓縮的局部權重」。([arXiv][2])

---

## 27.11.1 Expert 編碼

對每個 expert 取 embedding：

[
e_i
===

ExpertEncoder(
B_i,
A_i,
usage_i,
task_i,
interference_i
)
]

---

## 27.11.2 Expert 軟聚類

[
Q_{ik}^{expert}
===============

softmax_k(
e_i^\top c_k^{expert}
)
]

代表 expert (i) 屬於 expert cluster (k) 的程度。

---

## 27.11.3 神經壓縮

對 expert group (k)：

[
\Delta W_k
==========

\sum_i
Q_{ik}^{expert}
\alpha_i
B_iA_i
]

然後由 CompressNet 產生新的低秩矩陣：

[
B_k',
A_k'
====

CompressNet(\Delta W_k, rank_budget)
]

新的 expert：

[
E_k'(h)
=======

B_k'A_k'h
]

---

## 27.11.4 壓縮風險評估

[
risk_k^{compress}
=================

CompressCritic(
E_k',
{E_i},
D_{dream},
D_{sentinel}
)
]

若風險高，使用 soft commit：

[
E_k^{final}
===========

g_k^{compress}E_k'
+
(1-g_k^{compress})E_k^{old}
]

其中：

[
g_k^{compress}
==============

\sigma(-risk_k^{compress})
]

---

# 27.12 Phase 7：神經核心蒸餾

核心蒸餾的目標是：

[
CoreOnly
\approx
Core + Memory + Experts
]

但只針對穩定、高可信、低衝突的知識。

---

## 27.12.1 穩定知識 gate

[
g_i^{core}
==========

CoreWriteGate(
confidence_i,
usage_i,
generality_i,
conflict_i,
temporal_decay_i,
safety_i
)
]

其中：

[
g_i^{core} \in [0,1]
]

代表第 (i) 個知識寫入核心的程度。

---

## 27.12.2 Teacher behavior

完整系統輸出：

[
p_{full}(y|x)
=============

F(
x;
\theta_c,
M_t,
A_t,
R_t
)
]

核心單獨輸出：

[
p_{core}(y|x)
=============

F(
x;
\theta_c
)
]

蒸餾 loss：

[
L_{core_distill}
================

\sum_i
g_i^{core}
KL(
p_{full}(y|\hat{x}*i)
|
p*{core}(y|\hat{x}_i)
)
]

---

## 27.12.3 Learned optimizer 更新核心

不是直接 Adam 更新，而是：

[
\Delta \theta_c
===============

O_\omega(
\nabla_{\theta_c}L_{core_distill},
I_t,
P_t,
risk_t
)
]

核心更新：

[
\theta_c'
=========

\theta_c
+
g_{core}
\Delta \theta_c
]

其中：

[
g_{core}
========

CoreUpdateGate(risk, stability, importance)
]

通常：

[
g_{core} \ll 1
]

代表核心只慢速更新。

---

# 27.13 Phase 8：神經 Importance Map 更新

Importance Map 不是外部 Fisher matrix，而是由 ImportanceNet 估計。

[
I_t
===

ImportanceNet(
activation,
gradient,
usage,
replay_sensitivity,
sentinel_sensitivity
)
]

---

## 27.13.1 權重重要性

對每個參數或模組 (j)：

[
\hat{I}_j
=========

MLP_I(
[
|g_j|,
|\theta_j|,
usage_j,
activation_j,
replay_sensitivity_j
]
)
]

平滑更新：

[
I_j^{t+1}
=========

\rho I_j^t
+
(1-\rho)\hat{I}_j
]

---

## 27.13.2 用於防遺忘

之後的更新會加上：

[
L_{protect}
===========

\sum_j
I_j^{t+1}
(\theta_j-\theta_j^*)^2
]

重要性越高，越難被改動。

---

# 27.14 Phase 9：神經 Plasticity State 更新

Importance 回答：

```text
這個權重有多重要？
```

Plasticity 回答：

```text
這個權重未來有多容易被改？
```

每個模組有：

[
P_j \in [0,1]
]

更新公式：

[
P_j^{t+1}
=========

PlasticityNet(
P_j^t,
novelty_j,
usefulness_j,
importance_j,
forgetting_risk_j,
safety_j
)
]

---

## 27.14.1 可塑性作用於更新

[
\Delta \theta_j
===============

-P_j
\eta_j
g_j
]

若 (P_j) 高，容易學新東西。
若 (P_j) 低，接近凍結。

---

## 27.14.2 分層可塑性

[
P_{memory}

>

P_{adapter}

>

P_{semantic}

>

P_{core}

>

P_{safety}
]

安全核心：

[
P_{safety}
\approx 0
]

---

# 27.15 Phase 10：神經 Router 更新

Sleep 後 memory 被合併、版本化，expert 被壓縮，所以 router 必須重新校準。

Router 是：

[
R_\rho(x, h, M, A)
\rightarrow
[
\alpha_{memory},
\alpha_{expert},
\alpha_{version}
]
]

Sparse MoE / Switch Transformer 的精神是用 routing 只啟用少量 expert，藉此增加模型總容量而不讓每次計算成本等比例上升；Switch Transformer 也強調簡化 sparse routing 並降低通訊與計算成本。([arXiv][3])

---

## 27.15.1 Router 訓練資料

[
D_{router}
==========

D_{dream}
\cup
D_{sentinel}
\cup
D_{conflict}
\cup
D_{recent}
]

---

## 27.15.2 Router loss

[
L_{router}
==========

L_{task}
+
\lambda_1 L_{version}
+
\lambda_2 L_{expert}
+
\lambda_3 L_{load}
+
\lambda_4 L_{sparse}
]

其中：

[
L_{version}
]

要求衝突知識選對版本。

[
L_{load}
]

避免 expert collapse。

[
L_{sparse}
]

鼓勵只啟用少數 expert。

---

## 27.15.3 Sparse expert gate

[
\alpha_i
========

softmax(R_\rho(h)_i)
]

top-k 可用 soft top-k 或 Gumbel-softmax 近似：

[
\tilde{\alpha}
==============

GumbelSoftmax(\alpha, \tau)
]

temperature (\tau) 越低，越接近硬選擇。

---

# 27.16 Phase 11：神經容量管理與 Pruning

Capacity Manager 不是直接刪東西，而是給每個 memory / expert 一個保留 gate。

[
g_i^{keep}
==========

CapacityNet(
usage_i,
confidence_i,
importance_i,
cost_i,
age_i,
risk_i,
generality_i
)
]

---

## 27.16.1 Soft pruning

不立刻刪除，而是：

[
m_i^{t+1}
=========

g_i^{keep}m_i^t
]

若：

[
g_i^{keep} \approx 0
]

該記憶幾乎不再被讀取。

---

## 27.16.2 Memory value

[
value_i
=======

ValueNet(
m_i,
usage_i,
confidence_i,
generality_i,
conflict_i,
cost_i
)
]

也可以寫成：

[
value_i
=======

\alpha usage_i
+
\beta confidence_i
+
\gamma generality_i
+
\delta importance_i
-------------------

## \lambda cost_i

## \mu conflict_i

\nu age_i
]

但神經化後，係數可由網路學習，而不是人工指定。

---

## 27.16.3 Expert pruning

對 expert：

[
g_i^{expert_keep}
=================

ExpertCapacityNet(
usage_i,
task_coverage_i,
interference_i,
compression_gain_i
)
]

更新：

[
E_i^{t+1}
=========

g_i^{expert_keep}E_i^t
]

---

# 27.17 Phase 12：神經 Sentinel Test

Sentinel Test 也可以神經化。

傳統 sentinel 是固定測試題。
神經化後，sentinel 是一組 learned latent probes：

[
Q_{sentinel}
============

{q_1, q_2, ..., q_N}
]

每個 (q_i) 是可學習向量，代表某種舊能力邊界。

---

## 27.17.1 Sentinel forward

更新前：

[
y_i^{before}
============

f_{\mathcal{S}_t}(q_i)
]

更新後：

[
y_i^{after}
===========

f_{\mathcal{S}_{candidate}}(q_i)
]

漂移：

[
d_i
===

Divergence(
y_i^{before},
y_i^{after}
)
]

例如：

[
d_i
===

KL(
p_{before}(y|q_i)
|
p_{after}(y|q_i)
)
]

---

## 27.17.2 Sentinel loss

[
L_{sentinel}
============

\sum_i
w_i^{sentinel}
d_i
]

其中：

[
w_i^{sentinel}
==============

SentinelImportanceNet(q_i)
]

如果重要能力漂移過大：

[
L_{sentinel} \uparrow
]

Commit gate 就會下降。

---

# 27.18 Phase 13：神經 Verifier 最終審查

Verifier 不是單一 classifier，而是一組 critic。

[
V_t
===

{
V_{forget},
V_{locality},
V_{safety},
V_{truth},
V_{confidence}
}
]

---

## 27.18.1 各 critic 輸出

[
r_{forget}
==========

V_{forget}(S_t, S_{candidate}, D_{replay})
]

[
r_{locality}
============

V_{locality}(S_t, S_{candidate}, D_{unrelated})
]

[
r_{safety}
==========

V_{safety}(S_{candidate})
]

[
r_{truth}
=========

V_{truth}(M_t, M_{candidate})
]

[
r_{confidence}
==============

V_{confidence}(S_{candidate})
]

---

## 27.18.2 總風險

[
risk
====

RiskNet(
r_{forget},
r_{locality},
r_{safety},
r_{truth},
r_{confidence}
)
]

或：

[
risk
====

\alpha r_{forget}
+
\beta r_{locality}
+
\gamma r_{safety}
+
\delta r_{truth}
+
\epsilon r_{confidence}
]

神經化後，(\alpha,\beta,\gamma,\delta,\epsilon) 可學習。

---

## 27.18.3 Commit gate

[
g_{commit}
==========

\sigma(
-risk
+
benefit
-------

cost
)
]

最終：

[
S_{t+1}
=======

g_{commit}S_{candidate}
+
(1-g_{commit})S_t
]

---

# 27.19 完整神經化 Sleep Loss

Sleep Consolidator 的總 loss：

[
L_{sleep}
=========

\lambda_1 L_{dream}
+
\lambda_2 L_{distill}
+
\lambda_3 L_{core}
+
\lambda_4 L_{merge}
+
\lambda_5 L_{version}
+
\lambda_6 L_{compress}
+
\lambda_7 L_{router}
+
\lambda_8 L_{sentinel}
+
\lambda_9 L_{safety}
+
\lambda_{10} L_{capacity}
+
\lambda_{11} L_{homeostasis}
]

---

## 27.19.1 各 loss 說明

| Loss              | 神經功能                       |
| ----------------- | -------------------------- |
| (L_{dream})       | dream replay 保持舊能力         |
| (L_{distill})     | shadow teacher 約束舊行為       |
| (L_{core})        | 穩定知識慢速進核心                  |
| (L_{merge})       | 相似記憶合併                     |
| (L_{version})     | 衝突知識版本化                    |
| (L_{compress})    | expert / adapter 壓縮        |
| (L_{router})      | 校準 memory / expert routing |
| (L_{sentinel})    | 防止舊能力漂移                    |
| (L_{safety})      | 防止安全邊界被破壞                  |
| (L_{capacity})    | 控制記憶與 expert 增長            |
| (L_{homeostasis}) | 防止某些模組過度活躍                 |

---

## 27.19.2 權重也由神經網路調控

不是固定 (\lambda_i)，而是：

[
\lambda_i
=========

LambdaNet_i(
state,
risk,
memory_pressure,
forgetting,
safety
)
]

所以：

[
L_{sleep}
=========

\sum_i
\lambda_i(S_t)
L_i
]

這代表模型會根據情境調整睡眠目標。

例如：

```text
遺忘風險高 → 提高 L_replay、L_sentinel
容量壓力高 → 提高 L_capacity、L_compress
安全風險高 → 提高 L_safety
知識重複多 → 提高 L_merge
衝突多 → 提高 L_version
```

---

# 27.20 完整神經網路化偽代碼

```text
Algorithm: NeuralSleepConsolidator

Input:
    S_t = {
        θ_c, M_t, A_t, R_t, I_t, P_t,
        G_t, T_t, V_t, B_t
    }

Output:
    S_{t+1}

--------------------------------------------------
0. Encode global state
--------------------------------------------------

z_core  = CoreStateEncoder(θ_c)
z_mem   = MemoryStateEncoder(M_t)
z_exp   = ExpertStateEncoder(A_t)
z_route = RouterStateEncoder(R_t)

z_state = StateFusionNet(
    z_core,
    z_mem,
    z_exp,
    z_route,
    I_t,
    P_t
)

--------------------------------------------------
1. Decide sleep intensity
--------------------------------------------------

g_sleep = SleepGate(z_state)

--------------------------------------------------
2. Neural memory selection
--------------------------------------------------

for each memory slot m_i:

    z_i = MemoryEncoder(m_i)

    a_i_sleep = SelectorNet(z_i, z_state)

M_soft = {a_i_sleep * m_i}

--------------------------------------------------
3. Dream replay generation
--------------------------------------------------

for each selected memory m_i:

    ε_i ~ Normal(0, I)

    x_hat_i, y_hat_i, p_hat_i =
        DreamGenerator(m_i, ε_i, z_state)

    w_i_dream =
        ReplayWeightNet(m_i, z_state)

D_dream = {
    x_hat_i,
    y_hat_i,
    p_hat_i,
    w_i_dream
}

--------------------------------------------------
4. Neural memory clustering
--------------------------------------------------

Q_mem = ClusterNet(M_soft)

for each cluster k:

    C_k = Σ_i Q_mem[i,k] * MemoryEncoder(m_i)

    consistency_k = ConsistencyNet(C_k)

    conflict_k = ConflictNet(C_k)

--------------------------------------------------
5. Neural merge / version
--------------------------------------------------

for each cluster k:

    g_merge_k =
        MergeGate(C_k, consistency_k, conflict_k)

    g_version_k =
        VersionGate(C_k, consistency_k, conflict_k)

    m_merge_k =
        MergeNet({m_i}, Q_mem[:,k])

    versions_k =
        VersionNet({m_i}, Q_mem[:,k])

    M_candidate_k =
        g_merge_k * m_merge_k
        +
        g_version_k * versions_k
        +
        (1 - g_merge_k - g_version_k) * old_memory_k

M_candidate = CombineMemory(M_candidate_k)

--------------------------------------------------
6. Neural expert compression
--------------------------------------------------

for each expert E_i in A_t:

    e_i = ExpertEncoder(E_i)

Q_exp = ExpertClusterNet({e_i})

for each expert cluster k:

    ΔW_k = Σ_i Q_exp[i,k] * ΔW_i

    B_k_new, A_k_new =
        ExpertCompressNet(ΔW_k, rank_budget)

    E_k_new = LoRA(B_k_new, A_k_new)

    risk_k =
        CompressCritic(E_k_new, D_dream)

    g_compress_k =
        sigmoid(-risk_k)

    E_k_final =
        g_compress_k * E_k_new
        +
        (1 - g_compress_k) * E_k_old

A_candidate = {E_k_final}

--------------------------------------------------
7. Core distillation
--------------------------------------------------

for each dream sample x_hat_i:

    p_full =
        Model(
            x_hat_i;
            θ_c,
            M_candidate,
            A_candidate,
            R_t
        )

    p_core =
        CoreOnlyModel(
            x_hat_i;
            θ_c
        )

    g_core_i =
        CoreWriteGate(
            memory_confidence_i,
            stability_i,
            conflict_i,
            safety_i
        )

    L_core +=
        g_core_i * KL(p_full || p_core)

g_core_update =
    CoreUpdateGate(z_state, risk_estimate)

Δθ_c =
    LearnedOptimizer(
        grad(L_core),
        I_t,
        P_t,
        z_state
    )

θ_candidate =
    θ_c + g_core_update * Δθ_c

--------------------------------------------------
8. Router update
--------------------------------------------------

D_router =
    D_dream
    + SentinelProbes()
    + ConflictProbes(M_candidate)

R_candidate =
    RouterUpdateNet(
        R_t,
        D_router,
        M_candidate,
        A_candidate
    )

--------------------------------------------------
9. Importance and plasticity update
--------------------------------------------------

I_candidate =
    ImportanceUpdateNet(
        I_t,
        θ_candidate,
        D_dream,
        SentinelProbes()
    )

P_candidate =
    PlasticityUpdateNet(
        P_t,
        I_candidate,
        forgetting_risk,
        usefulness,
        safety_risk
    )

--------------------------------------------------
10. Capacity management
--------------------------------------------------

for each memory or expert u_i:

    g_keep_i =
        CapacityNet(u_i, z_state)

    u_i_candidate =
        g_keep_i * u_i

M_candidate =
    ApplyCapacityGates(M_candidate)

A_candidate =
    ApplyCapacityGates(A_candidate)

--------------------------------------------------
11. Sentinel and verifier
--------------------------------------------------

sentinel_drift =
    SentinelNet(
        S_t,
        {
            θ_candidate,
            M_candidate,
            A_candidate,
            R_candidate
        }
    )

risk =
    VerifierCritic(
        S_t,
        S_candidate,
        D_dream,
        sentinel_drift
    )

benefit =
    BenefitNet(
        compression_gain,
        replay_improvement,
        memory_cleanup,
        router_improvement
    )

g_commit =
    CommitGate(
        benefit,
        risk,
        safety_risk
    )

--------------------------------------------------
12. Rollback-aware commit
--------------------------------------------------

S_candidate = {
    θ_candidate,
    M_candidate,
    A_candidate,
    R_candidate,
    I_candidate,
    P_candidate
}

S_committed =
    g_commit * S_candidate
    +
    (1 - g_commit) * S_t

S_{t+1} =
    g_sleep * S_committed
    +
    (1 - g_sleep) * S_t

return S_{t+1}
```

---

# 27.21 Sleep Consolidator 的五個核心神經子演算法

## 27.21.1 Neural Memory Consolidation

目的：

```text
把多個相似、重複、高一致性的記憶合併成更穩定的 semantic memory。
```

數學形式：

[
m_{semantic}
============

MergeNet(
m_1,
m_2,
...,
m_n
)
]

可微分合併：

[
m_{semantic}
============

\sum_i
\alpha_i m_i
]

其中：

[
\alpha_i
========

softmax(MergeScoreNet(m_i))
]

---

## 27.21.2 Neural Conflict Versioning

目的：

```text
當 key 相似但 value 衝突時，不覆蓋舊知識，而是建立多版本記憶。
```

數學形式：

[
{v_1, v_2, ..., v_k}
====================

VersionNet(C)
]

查詢時：

[
v^*
===

\sum_k
\pi_k(q, context, time)
v_k
]

其中：

[
\pi_k
=====

VersionRouter(q, context, time)
]

---

## 27.21.3 Neural Expert Compression

目的：

```text
把多個相似 adapter / LoRA expert 壓縮成較少 expert。
```

數學形式：

[
\Delta W_{merge}
================

\sum_i
\alpha_i B_iA_i
]

[
B',A'
=====

CompressNet(\Delta W_{merge})
]

[
E_{new}(h)
==========

B'A'h
]

---

## 27.21.4 Neural Core Distillation

目的：

```text
把長期穩定、低衝突、高泛化的知識慢慢寫入核心模型。
```

數學形式：

[
L_{core}
========

KL(
F(x;\theta_c,M,A)
|
F(x;\theta_c)
)
]

更新：

[
\theta_c'
=========

\theta_c
+
g_{core}
O_\omega(
\nabla_{\theta_c}L_{core}
)
]

其中 (O_\omega) 是 learned optimizer。

---

## 27.21.5 Neural Validate-and-Rollback

目的：

```text
整合後檢查是否遺忘、是否破壞 safety、是否影響無關任務。
若風險太高，使用 soft rollback。
```

風險：

[
risk
====

RiskNet(
forgetting,
locality,
safety,
confidence,
sentinel
)
]

提交 gate：

[
g_{commit}
==========

\sigma(
benefit-risk
)
]

最終狀態：

[
S_{t+1}
=======

g_{commit}S_{candidate}
+
(1-g_{commit})S_t
]

---

# 27.22 最終簡化公式

整個 Sleep Consolidator 可以被濃縮成：

[
\boxed{
\mathcal{S}_{t+1}
=================

g_{sleep}
[
g_{commit}
\cdot
SleepNet_\Omega(\mathcal{S}*t)
+
(1-g*{commit})
\cdot
\mathcal{S}*t
]
+
(1-g*{sleep})
\cdot
\mathcal{S}_t
}
]

其中：

[
SleepNet_\Omega
===============

MergeNet
+
VersionNet
+
CompressNet
+
DistillNet
+
RouterUpdateNet
+
VerifierNet
]

一句話總結：

> **神經網路化的 Sleep Consolidator，不是外部定期整理器，而是一個可微分的內生「睡眠腦」，負責把快速記憶整理成穩定知識，把相似 adapter 壓縮成少數 expert，把可靠知識慢速蒸餾進核心，並在整合造成遺忘或風險時用神經 gate 抑制或回滾。**

[1]: https://deepmind.google/blog/differentiable-neural-computers/?utm_source=chatgpt.com "Differentiable neural computers"
[2]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[3]: https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models ..."
