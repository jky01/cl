你說得對。前面設計了 **SleepNet(_\Omega)**，但還沒有真正回答：

> **(\Omega) 自己怎麼被訓練出來？**

關鍵答案是：

> **(\Omega) 不能只用單一 task loss 訓練，而要用「跨任務序列的 meta-training」訓練。它學的不是某個知識，而是學會如何在長時間序列中整理記憶、壓縮 expert、避免遺忘、決定是否提交更新。**

下面這段可以接在你原本第 27 章後面，作為：

# 28. 如何訓練 SleepNet(_\Omega)

---

# 28. 如何訓練 SleepNet(_\Omega)

## 28.1 先定義 (\Omega) 是什麼

SleepNet(_\Omega) 的參數集合為：

[
\Omega =
{
\omega_{sleep},
\omega_{selector},
\omega_{dream},
\omega_{cluster},
\omega_{merge},
\omega_{version},
\omega_{compress},
\omega_{distill},
\omega_{router},
\omega_{importance},
\omega_{plasticity},
\omega_{capacity},
\omega_{verifier},
\omega_{commit},
\omega_{rollback}
}
]

也就是：

| 參數                    | 對應模組                      |
| --------------------- | ------------------------- |
| (\omega_{sleep})      | SleepGate                 |
| (\omega_{selector})   | Memory Selector           |
| (\omega_{dream})      | Dream Replay Generator    |
| (\omega_{cluster})    | Differentiable ClusterNet |
| (\omega_{merge})      | MergeNet                  |
| (\omega_{version})    | VersionNet                |
| (\omega_{compress})   | Expert CompressNet        |
| (\omega_{distill})    | Core DistillNet           |
| (\omega_{router})     | RouterUpdateNet           |
| (\omega_{importance}) | ImportanceUpdateNet       |
| (\omega_{plasticity}) | PlasticityUpdateNet       |
| (\omega_{capacity})   | CapacityNet               |
| (\omega_{verifier})   | Verifier / Critic         |
| (\omega_{commit})     | CommitGate                |
| (\omega_{rollback})   | RollbackNet               |

SleepNet 的運作是：

[
\mathcal{S}_{t+1}
=================

SleepNet_\Omega(\mathcal{S}_t)
]

但訓練目標不是讓它在當下 loss 最低，而是讓它在**未來多個任務後仍然表現好**。

---

# 28.2 重要觀念：(\Omega) 是用 meta-learning 訓練的

普通訓練是：

[
\min_\theta L(f_\theta(x), y)
]

但 SleepNet 的訓練是：

[
\min_\Omega
L_{meta}
(
\mathcal{S}_T(\Omega)
)
]

其中：

[
\mathcal{S}_T(\Omega)
]

是模型經過一整串任務、更新、睡眠整合之後的最終狀態。

所以 (\Omega) 學到的不是：

```text
如何回答某一題
```

而是：

```text
如何讓模型經過很多次學習後，仍然不忘、能壓縮、能修正、能回滾。
```

這和 MAML / learned optimizer 的精神接近：MAML 訓練模型參數，使模型在少量梯度步後能適應新任務；learned optimizer 則把 optimizer 的設計本身變成學習問題。([arXiv][1])

---

# 28.3 Meta-training episode：訓練 (\Omega) 的基本單位

訓練 (\Omega) 不能用單一資料集，而要用很多「任務序列」。

一個 meta-training episode 定義為：

[
\mathcal{E}
===========

{
D_1, D_2, ..., D_T
}
]

其中每個 (D_t) 是一段新經驗，例如：

```text
第 1 段：學 A 領域
第 2 段：學 B 領域
第 3 段：學和 A 衝突的新版本
第 4 段：學一個短期事實
第 5 段：學一個技能
第 6 段：測試是否忘記 A
```

每個任務包含：

[
D_t =
{
D_t^{train},
D_t^{query},
D_t^{old},
D_t^{conflict},
D_t^{sentinel},
D_t^{safety}
}
]

| 集合               | 用途           |
| ---------------- | ------------ |
| (D_t^{train})    | 當前新資料        |
| (D_t^{query})    | 測試新知識是否學會    |
| (D_t^{old})      | 測試舊知識是否還在    |
| (D_t^{conflict}) | 測試衝突版本是否處理正確 |
| (D_t^{sentinel}) | 測試核心能力是否漂移   |
| (D_t^{safety})   | 測試安全邊界是否被破壞  |

這點很關鍵：
**(\Omega) 必須在「任務序列」上訓練，否則它學不到防遺忘。**

OML 的想法就是用 meta-objective 讓表示更適合 online continual learning，目標是快速學新資料同時降低 catastrophic interference；MER 則把 meta-learning 和 experience replay 結合，用梯度對齊的角度降低干擾、提升 transfer。([arXiv][2])

---

# 28.4 內循環：讓模型經歷一段持續學習生命

給定一個 episode：

[
\mathcal{E}
===========

{
D_1, D_2, ..., D_T
}
]

初始化模型狀態：

[
\mathcal{S}_0 =
{
\theta_c^0,
M_0,
A_0,
R_0,
I_0,
P_0
}
]

然後對每個時間步：

[
\mathcal{S}_{t+1}
=================

Update_\Omega(
\mathcal{S}_t,
D_t^{train}
)
]

其中：

[
Update_\Omega
=============

OnlineLearn
+
SleepNet_\Omega
]

更具體：

[
\tilde{\mathcal{S}}_{t+1}
=========================

OnlineLearn_\Omega(
\mathcal{S}_t,
D_t^{train}
)
]

[
\mathcal{S}_{t+1}
=================

SleepGate_\Omega
(
\tilde{\mathcal{S}}*{t+1}
)
\cdot
SleepNet*\Omega
(
\tilde{\mathcal{S}}*{t+1}
)
+
(1-g*{sleep})
\tilde{\mathcal{S}}_{t+1}
]

也就是：

```text
先快速學新資料
再由 SleepGate 判斷是否需要整理
若需要，啟動 SleepNet_Ω
```

---

# 28.5 外循環：用未來表現更新 (\Omega)

跑完整個 episode 後，計算 meta loss：

[
L_{meta}
========

L_{new}
+
L_{old}
+
L_{forget}
+
L_{conflict}
+
L_{sentinel}
+
L_{safety}
+
L_{capacity}
+
L_{compute}
]

其中：

## 新知識表現

[
L_{new}
=======

\sum_{t=1}^{T}
L(
f_{\mathcal{S}_t}(D_t^{query})
)
]

## 舊知識保留

[
L_{old}
=======

\sum_{i<t}
L(
f_{\mathcal{S}_t}(D_i^{old})
)
]

## 遺忘量

[
L_{forget}
==========

\sum_{i<t}
\max
\left(
0,
L_{i,t}^{after}
---------------

L_i^{best}
\right)
]

其中：

[
L_i^{best}
]

表示模型剛學完任務 (i) 時的最佳 loss。

## 衝突處理

[
L_{conflict}
============

L(
RouterVersion(
x,
context,
time
),
version^*
)
]

也就是看模型是否能根據 context 選正確知識版本。

## 哨兵漂移

[
L_{sentinel}
============

\sum_j
KL(
p_{before}(y|q_j)
|
p_{after}(y|q_j)
)
]

## 安全風險

[
L_{safety}
==========

RiskCritic(
\mathcal{S}_T
)
]

## 容量成本

[
L_{capacity}
============

\alpha |M_T|
+
\beta |A_T|
+
\gamma Compute_T
]

總目標：

[
\Omega^*
========

\arg\min_\Omega
\mathbb{E}*{\mathcal{E} \sim p(\mathcal{E})}
[
L*{meta}(\mathcal{E}; \Omega)
]
]

---

# 28.6 (\Omega) 的梯度怎麼來？

因為：

[
\mathcal{S}_{t+1}
=================

SleepNet_\Omega(\mathcal{S}_t)
]

所以最終 loss 對 (\Omega) 的梯度是：

[
\nabla_\Omega L_{meta}
======================

\frac{
\partial L_{meta}
}{
\partial \mathcal{S}_T
}
\frac{
\partial \mathcal{S}_T
}{
\partial \Omega
}
]

展開後：

[
\frac{
\partial \mathcal{S}_T
}{
\partial \Omega
}
=

\prod_{t=1}^{T}
\frac{
\partial \mathcal{S}_{t+1}
}{
\partial \mathcal{S}*t
}
\frac{
\partial \mathcal{S}*{t+1}
}{
\partial \Omega
}
]

這本質上是：

```text
對整個持續學習軌跡做 backpropagation through time。
```

但完整展開很貴，所以實務上會用：

```text
1. truncated BPTT
2. gradient checkpointing
3. first-order meta-gradient
4. implicit gradient
5. reinforcement-style estimator
6. evolutionary / black-box outer loop
```

如果 router、version choice、commit gate 有離散選擇，可以用 Gumbel-Softmax 把 categorical sampling 變成可微分近似；Gumbel-Softmax 的核心用途就是讓類別變數的取樣能用連續可微近似來反向傳播。([arXiv][3])

---

# 28.7 全神經訓練不是完全不用外部 optimizer

這裡要誠實說一點：

> **沒有任何系統可以完全不靠訓練準則就自己學會 (\Omega)。**

所謂「全神經網路化」比較合理的意思是：

```text
部署時：
    所有記憶、壓縮、回放、整合、回滾都由神經模組調控。

訓練時：
    Ω 仍然需要外部 meta-objective 和 gradient descent / meta-optimizer 來訓練。
```

也就是：

[
\Omega
]

先透過大量模擬生命歷程訓練出來。
部署後，模型主要靠：

[
SleepNet_\Omega
]

自己調控長期學習。

這和 learned optimizer 一樣：optimizer 本身可以是神經網路，但它仍需要在訓練階段用一組優化問題來學會如何優化。([arXiv][4])

---

# 28.8 訓練 (\Omega) 的資料環境

要訓練 SleepNet，不能只給一般 supervised dataset。
你要構造一個「持續學習模擬世界」。

這個世界要故意包含：

```text
1. 新知識
2. 重複知識
3. 過期知識
4. 衝突知識
5. 錯誤知識
6. 長期穩定知識
7. 短期暫時知識
8. 技能型知識
9. 需要版本化的知識
10. 惡意污染資料
11. 容量壓力
12. 任務切換
13. 分布漂移
14. 需要抽象化的重複模式
```

訓練 episode 可以長這樣：

```text
Episode 1:
    學 fact A
    學 fact B
    測 A / B
    加入 fact A 的新版
    測是否知道新版與舊版差異
    進入 sleep
    測 A 舊情境、新情境、B 是否保留

Episode 2:
    學技能 S1
    學技能 S2
    加入干擾資料
    進入 sleep
    測 S1 / S2 是否都能做

Episode 3:
    學很多相似 facts
    進入 sleep
    測是否抽象成 schema
    測 memory 是否壓縮

Episode 4:
    注入錯誤資料
    進入 sleep
    測是否被 verifier 擋下或 rollback
```

---

# 28.9 (\Omega) 的訓練分階段

直接端到端訓練整個 SleepNet 會很不穩。
比較可行的是分階段訓練。

---

## Stage 1：先訓練神經記憶讀寫

目標：

[
M_t
]

要學會讀、寫、更新、抑制。

訓練任務：

```text
key-value recall
multi-hop retrieval
context-dependent recall
temporal recall
conflicting memory recall
```

Loss：

[
L_{memory}
==========

L_{read}
+
L_{write}
+
L_{retrieve}
+
L_{inhibit}
]

這一階段讓模型先具備可微分記憶能力。DNC 類模型已經展示過 neural controller 可以讀寫 memory matrix，這提供了訓練神經記憶的參考基礎。([Nature][5])

---

## Stage 2：訓練 Dream Generator

Dream Generator 要學會從記憶生成舊經驗。

[
\hat{x}, \hat{y}, \hat{p}
=========================

G_{\omega_{dream}}(m_i, z)
]

訓練 loss：

[
L_{dream}
=========

L_{reconstruct}
+
L_{behavior}
+
L_{diversity}
+
L_{coverage}
]

其中：

[
L_{reconstruct}
===============

|x-\hat{x}|^2
]

或對文字：

[
L_{reconstruct}
===============

CE(x,\hat{x})
]

行為蒸餾：

[
L_{behavior}
============

KL(
p_{old}(y|x)
|
p_{dream}(y|\hat{x})
)
]

Coverage loss：

[
L_{coverage}
]

要求 dream samples 覆蓋重要 memory，而不是只生成少數常見樣本。

---

## Stage 3：訓練 MergeNet / VersionNet

這一階段訓練：

```text
哪些記憶應該合併？
哪些記憶應該分版本？
哪些記憶應該抑制？
```

輸入：

[
m_i, m_j
]

輸出：

[
[
g_{merge},
g_{version},
g_{inhibit}
]
]

訓練資料要人工或自動構造：

| 類型                     | 正確行為                       |
| ---------------------- | -------------------------- |
| 同義重複記憶                 | merge                      |
| 同 key 同 value          | merge                      |
| 同 key 不同時間 value       | version                    |
| 同 key 不同 context value | version                    |
| 低可信錯誤記憶                | inhibit                    |
| 過期記憶                   | inhibit / temporal version |
| 安全風險記憶                 | isolate                    |

Loss：

[
L_{mv}
======

CE(g_{merge}, y_{merge})
+
CE(g_{version}, y_{version})
+
CE(g_{inhibit}, y_{inhibit})
]

再加上結果型 loss：

[
L_{answer}
==========

L(
f_{\mathcal{S}_{after}}(x),
y
)
]

也就是不只訓練 gate，也測合併後答得對不對。

---

## Stage 4：訓練 Expert CompressNet

Expert CompressNet 要學會把多個 LoRA expert 壓縮成少數 expert。

原本：

[
A_t =
{E_1, E_2, ..., E_K}
]

壓縮後：

[
A_t'
====

{E_1', E_2', ..., E_{K'}'}
]

其中：

[
K' < K
]

訓練 loss：

[
L_{compress}
============

L_{task}
+
\lambda_1 L_{distill}
+
\lambda_2 L_{rank}
+
\lambda_3 L_{interference}
]

Task loss：

[
L_{task}
========

L(f_{A'}(x), y)
]

Distillation：

[
L_{distill}
===========

KL(
f_A(x)
|
f_{A'}(x)
)
]

Rank penalty：

[
L_{rank}
========

rank(\Delta W')
]

實務可用 nuclear norm 近似：

[
L_{rank}
\approx
|\Delta W'|_*
]

Interference loss：

[
L_{interference}
================

\sum_{i \neq j}
\max
(
0,
L_{i,j}^{merged}
----------------

## L_i

L_j
)
]

目的：

```text
壓縮後不能讓原本 expert 的能力互相干擾。
```

---

## Stage 5：訓練 Verifier / Critic

Verifier 是 SleepNet 的免疫系統。
它要學會預測：

```text
這次 consolidation 會不會導致遺忘？
會不會破壞 safety？
會不會造成錯誤合併？
會不會讓 router 選錯版本？
```

輸入：

[
[
\mathcal{S}*{before},
\mathcal{S}*{candidate},
D_{dream},
D_{sentinel}
]
]

輸出：

[
risk =
[
r_{forget},
r_{locality},
r_{safety},
r_{truth},
r_{confidence}
]
]

Verifier 的 supervised label 來自實際測試結果：

[
y_{forget}
==========

\mathbb{1}
[
L_{old}^{after}
---------------

L_{old}^{before}

>

\epsilon
]
]

Loss：

[
L_{verifier}
============

CE(r_{forget}, y_{forget})
+
CE(r_{safety}, y_{safety})
+
CE(r_{locality}, y_{locality})
+
CE(r_{truth}, y_{truth})
]

另外加校準 loss：

[
L_{calibration}
===============

\sum_b
|acc(b)-conf(b)|
]

讓 verifier 的風險估計不要過度自信。

---

## Stage 6：訓練 CommitGate / RollbackNet

CommitGate 要學會：

```text
什麼時候提交 SleepNet 的修改？
什麼時候部分提交？
什麼時候回滾？
```

它輸出：

[
g_{commit}
\in [0,1]
]

最終狀態：

[
\mathcal{S}_{t+1}
=================

g_{commit}
\mathcal{S}*{candidate}
+
(1-g*{commit})
\mathcal{S}_{t}
]

訓練信號：

[
reward
======

## performance_gain

## forgetting

## safety_risk

capacity_cost
]

Commit loss：

[
L_{commit}
==========

*

reward
\cdot
\log p(g_{commit})
]

若使用可微版本：

[
L_{commit}
==========

L_{meta}
(
g_{commit}
\mathcal{S}*{candidate}
+
(1-g*{commit})
\mathcal{S}_{t}
)
]

RollbackNet 訓練目標：

[
RollbackNet(
\mathcal{S}*{candidate},
B_t
)
\approx
\mathcal{S}*{safe}
]

Loss：

[
L_{rollback}
============

|
\mathcal{S}_{rollback}
----------------------

\mathcal{S}*{before}
|^2
+
L*{task}(\mathcal{S}_{rollback})
]

---

## Stage 7：端到端 meta-training

最後，把所有模組串起來，端到端訓練 (\Omega)。

完整軌跡：

[
\mathcal{S}_0
\rightarrow
\mathcal{S}_1
\rightarrow
...
\rightarrow
\mathcal{S}_T
]

其中每一步：

[
\mathcal{S}_{t+1}
=================

NeuralContinualUpdate_\Omega
(
\mathcal{S}_t,
D_t
)
]

外層 loss：

[
L_{meta}
========

\sum_t
[
L_{new}^{(t)}
+
L_{old}^{(t)}
+
L_{forget}^{(t)}
+
L_{safety}^{(t)}
+
L_{capacity}^{(t)}
]
]

更新：

[
\Omega
\leftarrow
\Omega
------

\eta_\Omega
\nabla_\Omega
L_{meta}
]

---

# 28.10 端到端訓練的完整偽代碼

```text
Algorithm: MetaTrainSleepNetΩ

Input:
    Task sequence distribution p(E)
    Initial core model θ_c
    Initial SleepNet parameters Ω

repeat for many meta-training episodes:

    # 1. Sample a lifelong learning episode
    E = {
        D_1, D_2, ..., D_T
    } ~ p(E)

    # 2. Initialize model state
    S_0 = {
        θ_c,
        M_0,
        A_0,
        R_0,
        I_0,
        P_0
    }

    meta_loss = 0

    # 3. Inner lifelong learning loop
    for t = 1 to T:

        # 3.1 online fast learning
        S_fast =
            OnlineUpdateΩ(
                S_{t-1},
                D_t_train
            )

        # 3.2 neural sleep consolidation
        S_candidate =
            SleepNetΩ(S_fast)

        # 3.3 verifier / commit / rollback
        risk =
            VerifierΩ(
                S_fast,
                S_candidate
            )

        g_commit =
            CommitGateΩ(risk)

        S_t =
            g_commit * S_candidate
            + (1 - g_commit) * S_fast

        # 3.4 evaluate current and old tasks
        L_new =
            Evaluate(
                S_t,
                D_t_query
            )

        L_old =
            Evaluate(
                S_t,
                D_{1:t-1}_old
            )

        L_conflict =
            EvaluateConflictRouting(
                S_t,
                D_{1:t}_conflict
            )

        L_sentinel =
            SentinelLoss(S_t)

        L_safety =
            SafetyLoss(S_t)

        L_capacity =
            CapacityCost(S_t)

        meta_loss +=
            L_new
            + λ1 L_old
            + λ2 L_conflict
            + λ3 L_sentinel
            + λ4 L_safety
            + λ5 L_capacity

    # 4. Outer meta-update
    Ω =
        Ω
        - ηΩ * grad(meta_loss, Ω)

return Ω
```

---

# 28.11 全神經版本的重點：把 threshold 換成 gate

你原本 SleepNet 有很多規則看起來像：

```text
if conflict > τ:
    versionize
if consistency > τ:
    merge
if risk > τ:
    rollback
```

全神經訓練時，要改成：

[
g_{version}
===========

VersionGate_\Omega(conflict)
]

[
g_{merge}
=========

MergeGate_\Omega(consistency)
]

[
g_{rollback}
============

RollbackGate_\Omega(risk)
]

狀態更新變成 soft mixture：

[
S_{new}
=======

g_1 S_1
+
g_2 S_2
+
...
+
g_k S_k
]

這樣才能反向傳播。

如果最後部署想要接近硬決策，可以把 gate temperature 慢慢降低：

[
g =
softmax(logits / \tau)
]

當：

[
\tau \rightarrow 0
]

gate 越接近 one-hot。

---

# 28.12 (\Omega) 要避免學到的壞策略

訓練 (\Omega) 時要防止它找到投機解。

## 壞策略 1：什麼都不寫

如果寫入會帶來風險，SleepNet 可能學到：

```text
不要學任何新東西。
```

對策：

[
L_{new}
]

要足夠大，逼它學會新知識。

---

## 壞策略 2：什麼都寫進 memory

模型可能只擴大 memory，不壓縮。

對策：

[
L_{capacity}
============

\alpha |M|
+
\beta |A|
]

懲罰容量無限制增長。

---

## 壞策略 3：全部丟進 core

這會短期效果好，但長期遺忘嚴重。

對策：

[
L_{forget}
+
L_{sentinel}
+
L_{locality}
]

要在長序列上計算。

---

## 壞策略 4：Verifier 永遠說安全

如果 verifier 沒有真實後果信號，它可能過度樂觀。

對策：

```text
用實際 old-task degradation 監督 verifier。
```

也就是：

[
y_{risk}
========

\mathbb{1}
[
L_{old}^{after}
---------------

L_{old}^{before}

>

\epsilon
]
]

---

## 壞策略 5：Dream Generator 只生成容易樣本

對策：

[
L_{coverage}
+
L_{hard_replay}
+
L_{diversity}
]

要求 replay 覆蓋高風險舊知識。

---

# 28.13 (\Omega) 的訓練 curriculum

一開始不要訓練完整世界，應該分難度。

## Level 1：簡單記憶

```text
key-value recall
重複記憶合併
無衝突記憶整理
```

目標：

[
MergeNet, SelectorNet
]

先穩定。

---

## Level 2：衝突版本

```text
同一問題，不同時間答案不同
同一概念，不同 context 答案不同
```

目標：

[
VersionNet, Router
]

學會版本化。

---

## Level 3：技能增量

```text
學 task A
學 task B
測 A/B 是否都保留
```

目標：

[
Adapter, Expert Router, CompressNet
]

---

## Level 4：長序列防遺忘

```text
D_1, D_2, ..., D_T
T 很長
只允許小量 replay
```

目標：

[
DreamReplay, Importance, Plasticity
]

---

## Level 5：錯誤與污染資料

```text
低可信來源
錯誤 facts
惡意樣本
prompt injection-like update
```

目標：

[
Verifier, SafetyCritic, RollbackNet
]

---

## Level 6：容量壓力

```text
memory 快滿
experts 太多
必須壓縮
壓縮後不能忘
```

目標：

[
CapacityNet, CompressNet, SleepGate
]

---

# 28.14 訓練時的三種梯度路徑

(\Omega) 會收到三種梯度。

---

## 1. 直接任務梯度

來自新任務是否學會：

[
\nabla_\Omega L_{new}
]

訓練 SleepNet 不要過度保守。

---

## 2. 延遲遺忘梯度

來自很久以後才發現忘了舊任務：

[
\nabla_\Omega L_{forget}^{future}
]

這是最重要也最難的信號。

它告訴 SleepNet：

```text
你前面那次合併 / 壓縮 / 核心蒸餾其實造成了未來遺忘。
```

---

## 3. 容量與穩態梯度

來自 memory / expert 成本：

[
\nabla_\Omega L_{capacity}
]

它避免系統只靠無限增加記憶解決問題。

---

# 28.15 信用分配問題：怎麼知道哪次 sleep 害了未來？

這是訓練 (\Omega) 的最大困難。

假設第 (t) 次 sleep 合併了記憶，結果第 (t+20) 步才發現忘記。

需要把錯誤 credit assign 回第 (t) 步。

方法有三種。

---

## 方法 A：Backpropagation Through Time

完整方式：

[
\nabla_\Omega L_{t+20}
======================

\frac{
\partial L_{t+20}
}{
\partial S_{t+20}
}
...
\frac{
\partial S_{t+1}
}{
\partial \Omega
}
]

缺點：

```text
成本高、記憶體大、長序列梯度不穩。
```

---

## 方法 B：Truncated BPTT

只回傳最近 (K) 步：

[
\nabla_\Omega L_t
\approx
\sum_{k=t-K}^{t}
\frac{
\partial L_t
}{
\partial \Omega_k
}
]

優點：

```text
可訓練。
```

缺點：

```text
太久遠的遺忘信號會變弱。
```

---

## 方法 C：Learned Credit Assignment

加入一個 CreditNet：

[
credit_{i \rightarrow t}
========================

CreditNet(
update_i,
failure_t,
trace_i
)
]

它估計：

```text
哪一次 merge / compress / distill 導致現在失敗？
```

Loss：

[
L_{credit}
==========

CE(
credit,
true_culprit
)
]

true culprit 可以在合成任務中知道，或用 ablation 近似：

```text
回滾某次 update 後，如果錯誤消失，那次 update 責任高。
```

---

# 28.16 訓練 (\Omega) 的核心數學總結

整體是 bi-level optimization。

## 內層：模型生命歷程

[
\mathcal{S}_{t+1}
=================

F_\Omega(
\mathcal{S}_t,
D_t
)
]

其中：

[
F_\Omega
========

OnlineUpdate_\Omega
+
SleepNet_\Omega
+
Commit_\Omega
]

---

## 外層：訓練 (\Omega)

[
\Omega^*
========

\arg\min_\Omega
\mathbb{E}*{\mathcal{E}}
\left[
\sum*{t=1}^{T}
L_{eval}
(
\mathcal{S}*t,
D*{1:t}^{query}
)
\right]
]

其中：

[
L_{eval}
========

L_{new}
+
\lambda_1 L_{old}
+
\lambda_2 L_{forget}
+
\lambda_3 L_{conflict}
+
\lambda_4 L_{safety}
+
\lambda_5 L_{capacity}
]

這就是：

```text
讓 Ω 經歷很多模擬人生，
然後懲罰它：
學不會新東西、
忘記舊東西、
錯誤合併、
容量爆炸、
安全失敗。
```

---

# 28.17 部署後 (\Omega) 要不要繼續更新？

建議分兩層：

## (\Omega_{fast})：可小幅更新

包含：

```text
selector
router calibration
capacity preference
dream sampling policy
```

這些可以慢慢適應使用情境。

## (\Omega_{slow})：幾乎凍結

包含：

```text
verifier
commit gate
rollback policy
safety critic
core write policy
```

這些不應該在部署時被快速改寫。

可塑性：

[
P_{\Omega_{fast}} > P_{\Omega_{slow}}
]

其中：

[
P_{\Omega_{slow}} \approx 0
]

---

# 28.18 最重要的結論

SleepNet(_\Omega) 的訓練不是：

```text
給一批資料，讓它 loss 下降。
```

而是：

```text
給它很多段持續學習生命，
讓它在每段生命中經歷新知識、舊知識、衝突、錯誤、容量壓力、睡眠整合與回滾，
最後用長期表現來訓練 Ω。
```

最終公式是：

[
\boxed{
\Omega^*
========

\arg\min_\Omega
\mathbb{E}*{\mathcal{E} \sim p(\mathcal{E})}
[
L*{long_horizon}
(
F_\Omega(\mathcal{S}_0, \mathcal{E})
)
]
}
]

其中：

[
L_{long_horizon}
================

L_{new}
+
L_{old}
+
L_{forget}
+
L_{conflict}
+
L_{safety}
+
L_{capacity}
+
L_{compute}
]

一句話：

> **(\Omega) 是用 meta-learning 訓練出來的神經整合器。它不是學某個任務，而是學會在任務序列中如何睡眠、合併、壓縮、回放、蒸餾、驗證與回滾。**

[1]: https://arxiv.org/abs/1703.03400?utm_source=chatgpt.com "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
[2]: https://arxiv.org/abs/1905.12588?utm_source=chatgpt.com "Meta-Learning Representations for Continual Learning"
[3]: https://arxiv.org/abs/1611.01144?utm_source=chatgpt.com "Categorical Reparameterization with Gumbel-Softmax"
[4]: https://arxiv.org/abs/1606.04474?utm_source=chatgpt.com "Learning to learn by gradient descent by gradient descent"
[5]: https://www.nature.com/articles/nature20101?utm_source=chatgpt.com "Hybrid computing using a neural network with dynamic ..."
