以下內容本身就是 **Markdown 格式**，可以直接複製成 `.md` 文件使用。

---

# 神經調控式持續學習小模型架構

## 0. 核心目標

設計一個小型 AI 模型，使其能夠在沒有全量聯合訓練的情境下：

1. 持續學習新知識。
2. 避免災難性遺忘。
3. 長期更新內部權重與記憶。
4. 自行判斷什麼該學、什麼不該學。
5. 由神經網路內部機制完成記憶、路由、更新、保護、回放、壓縮與回滾。

這個系統不是單純的：

[
\theta_{t+1} = \theta_t - \eta \nabla L
]

而是：

[
\mathcal{S}*{t+1} = U*\phi(\mathcal{S}_t, x_t, y_t, feedback_t)
]

其中：

* (\mathcal{S}_t)：模型在時間 (t) 的完整內部狀態。
* (U_\phi)：神經調控更新器。
* (x_t)：新輸入。
* (y_t)：新目標或回饋。
* (feedback_t)：外部或內部驗證訊號。

---

# 1. 總體架構

## 1.1 系統狀態

整個模型狀態定義為：

[
\mathcal{S}_t =
{
\theta_c,
M_t,
A_t,
R_t,
C_t,
I_t,
P_t,
G_t,
T_t,
V_t,
H_t,
B_t,
K_t
}
]

| 符號         | 名稱                        | 說明                 |
| ---------- | ------------------------- | ------------------ |
| (\theta_c) | Stable Core               | 穩定核心 Transformer   |
| (M_t)      | Differentiable Memory     | 可微分神經記憶            |
| (A_t)      | Adapter / Expert Bank     | LoRA / Adapter 專家庫 |
| (R_t)      | Router                    | 神經路由器              |
| (C_t)      | Controller                | 神經調控中樞             |
| (I_t)      | Importance Map            | 權重重要性估計            |
| (P_t)      | Plasticity State          | 後設可塑性狀態            |
| (G_t)      | Dream Replay Generator    | 夢境回放生成器            |
| (T_t)      | Shadow Teacher            | 慢速教師模型             |
| (V_t)      | Verifier / Critic         | 內部驗證器              |
| (H_t)      | Homeostasis Controller    | 穩態控制器              |
| (B_t)      | Rollback Buffer           | 回滾與版本記憶            |
| (K_t)      | Safety / Alignment Kernel | 安全與對齊核心            |

---

## 1.2 高層資料流

```text
Input x
  ↓
Core Transformer
  ↓
Context / Task Inference
  ↓
Novelty / Surprise / Conflict / Uncertainty Heads
  ↓
Neural Controller
  ↓
┌─────────────────────────────────────┐
│ Differentiable Memory Read / Write   │
│ LoRA / Adapter Expert Routing        │
│ Dream Replay                         │
│ Importance Protection                │
│ Learned Gradient Projection          │
│ Verifier / Critic                    │
│ Safety Gate                          │
│ Sleep Consolidation                  │
└─────────────────────────────────────┘
  ↓
Answer / Update / Reject / Rollback
```

---

# 2. 設計原則

## 2.1 快速記憶，慢速權重

新知識不應直接寫入核心權重，而是依照穩定性分層：

| 層級               |  更新速度 | 用途          |
| ---------------- | ----: | ----------- |
| Working Memory   |    極快 | 當前上下文       |
| Episodic Memory  |     快 | 單次經驗、事件、事實  |
| Semantic Memory  |     中 | 穩定事實與概念     |
| Adapter / Expert |    中慢 | 局部技能與領域能力   |
| Core Model       |    極慢 | 通用能力與長期穩定知識 |
| Safety Kernel    | 幾乎不可塑 | 安全與對齊邊界     |

---

## 2.2 局部可塑，全域穩定

[
\text{Plasticity local, Stability global}
]

意思是：

* 新知識先局部寫入。
* 局部知識經過驗證後才進入 adapter。
* 長期穩定知識才慢慢壓縮進核心模型。
* 核心權重受到強保護。
* 安全規則受到最高保護。

---

## 2.3 不是所有資料都應該學

模型必須先判斷：

```text
這是新知識嗎？
這是真的嗎？
這會不會和舊知識衝突？
這是暫時知識還是長期知識？
這應該寫入 memory、adapter，還是 core？
這是否應該被拒絕學習？
```

---

# 3. Stable Core Transformer

## 3.1 描述

Stable Core 是模型的穩定基座，負責：

* 語言理解
* 基礎推理
* 指令遵循
* 基本世界模型
* 通用表示能力

核心權重記為：

[
\theta_c
]

核心模型輸出：

[
h_t = F_{\theta_c}(x_t)
]

---

## 3.2 設計原則

核心模型不應頻繁更新。
只有當某個知識或技能已被長期驗證，且不會破壞舊能力時，才允許進入核心。

核心更新條件：

[
\Delta L_{old} < \epsilon
]

[
confidence > \tau_c
]

[
stability > \tau_s
]

---

## 3.3 核心更新公式

[
\theta_c^{t+1}
==============

\theta_c^t
+
g_{core}
\cdot
\Delta \theta_c
]

其中：

* (g_{core})：由 controller 產生的核心更新 gate。
* (\Delta \theta_c)：安全更新方向。
* (g_{core}) 通常很小。

[
g_{core} \approx 0
]

代表核心大部分時間保持穩定。

---

# 4. Differentiable Memory Matrix

## 4.1 描述

神經記憶是模型內部可讀寫的記憶矩陣，不是外部資料庫。

[
M_t = {m_1, m_2, ..., m_N}
]

每個記憶槽：

[
m_i =
[
k_i,
v_i,
c_i,
s_i,
a_i,
p_i,
r_i
]
]

| 欄位    | 意義                       |
| ----- | ------------------------ |
| (k_i) | key，用於檢索                 |
| (v_i) | value，記憶內容               |
| (c_i) | confidence，可信度           |
| (s_i) | source / provenance，來源向量 |
| (a_i) | age，記憶年齡                 |
| (p_i) | plasticity，可塑性           |
| (r_i) | reliability，可靠性          |

---

## 4.2 記憶讀取

給定查詢向量：

[
q_t = W_q h_t
]

計算相似度：

[
score_i = q_t^\top k_i
]

加入可信度、時間新鮮度與上下文適配：

[
score_i'
========

q_t^\top k_i
+
\alpha c_i
+
\beta freshness_i
+
\gamma context_i
----------------

\delta inhibition_i
]

讀取權重：

[
a_i = softmax(score_i')
]

讀取結果：

[
r_t = \sum_i a_i v_i
]

---

## 4.3 記憶寫入

新記憶候選：

[
\tilde{m}_t = W_m[h_t, y_t, feedback_t]
]

寫入 gate：

[
w_i = C_\phi(h_t, novelty_t, surprise_t, conflict_t)
]

更新：

[
m_i^{t+1}
=========

(1 - w_i)m_i^t
+
w_i \tilde{m}_t
]

---

## 4.4 記憶分類

| 記憶類型              | 說明        |
| ----------------- | --------- |
| Working Memory    | 短期上下文     |
| Episodic Memory   | 單次事件或經驗   |
| Semantic Memory   | 穩定知識      |
| Temporal Memory   | 有時間有效性的知識 |
| Procedural Memory | 技能與操作流程   |
| Preference Memory | 使用者偏好     |
| Safety Memory     | 安全相關約束    |

---

# 5. Temporal Memory

## 5.1 描述

時間記憶處理會過期的知識，例如：

* 公司 CEO
* 法規版本
* 軟體版本
* 商品價格
* 新聞事件
* 使用者短期偏好

每筆記憶包含：

[
m_i =
[
k_i,
v_i,
valid_from_i,
valid_until_i,
freshness_i,
confidence_i
]
]

---

## 5.2 時間衰減

[
freshness_i(t)
==============

e^{-\lambda (t - t_i)}
]

讀取分數：

[
score_i
=======

sim(q, k_i)
\cdot
confidence_i
\cdot
freshness_i(t)
]

---

## 5.3 是否進入核心

時間性知識通常不應進入核心：

[
g_{core} = 0
]

除非它被證明是長期穩定規則。

---

# 6. Source / Provenance Memory

## 6.1 描述

每個知識都要知道來源：

```text
這個知識從哪裡來？
什麼時候學到？
可靠性如何？
是否有新版？
是否和其他來源衝突？
```

來源記憶：

[
source_i = E(source)
]

完整記憶：

[
m_i =
[
k_i,
v_i,
source_i,
time_i,
confidence_i,
version_i
]
]

---

## 6.2 來源可靠性

[
reliability_i = SourceNet(source_i, history_i, consistency_i)
]

更新記憶可信度：

[
confidence_i^{t+1}
==================

\eta confidence_i^t
+
(1-\eta) reliability_i
]

---

# 7. LoRA / Adapter Expert Bank

## 7.1 描述

Adapter / Expert Bank 負責局部技能與領域知識。

[
A_t = {E_1, E_2, ..., E_K}
]

每個 expert 是一個低秩更新：

[
E_i(h) = B_i A_i h
]

對某層權重：

[
W_i' = W_i + \Delta W_i
]

[
\Delta W_i = B_i A_i
]

其中 rank 很小：

[
rank(\Delta W_i) \ll rank(W_i)
]

---

## 7.2 Expert Routing

Router 產生 expert 權重：

[
\alpha = R_\rho(h_t, r_t, z_t)
]

只選 top-k：

[
\mathcal{E}_{active} = topk(\alpha)
]

adapter 輸出：

[
e_t =
\sum_{i \in \mathcal{E}_{active}}
\alpha_i E_i(h_t)
]

最終 hidden state：

[
h_t'
====

h_t
+
W_r r_t
+
e_t
]

---

## 7.3 開新 Expert 的條件

若新資料具有：

[
novelty > \tau_n
]

[
conflict < \tau_c
]

[
repetition > \tau_r
]

則建立新 expert：

[
E_{K+1} = InitExpert(h_t, y_t)
]

---

# 8. Hypernetwork Adapter Generator

## 8.1 描述

除了固定 expert bank，也可以用 hypernetwork 動態生成 adapter。

[
A_t, B_t = HyperNet_\psi(z_t)
]

其中：

[
z_t = ContextInferencer(h_t, M_t)
]

---

## 8.2 動態 LoRA

[
\Delta W_t = B_t A_t
]

[
W_t' = W + \Delta W_t
]

這代表模型可以根據任務情境即時生成局部權重。

---

# 9. Sparse Router

## 9.1 描述

Router 決定目前輸入該使用：

* 哪些 memory
* 哪些 expert
* 哪些 adapter
* 哪種回答模式
* 是否啟動 verifier
* 是否允許學習

---

## 9.2 Router 輸入

[
r_t =
R_\rho(
h_t,
M_t,
novelty_t,
conflict_t,
uncertainty_t,
task_t
)
]

---

## 9.3 Router 輸出

[
r_t =
[
g_{memory},
g_{expert},
g_{core},
g_{replay},
g_{verify},
g_{safety},
g_{consolidate}
]
]

| Gate              | 作用            |
| ----------------- | ------------- |
| (g_{memory})      | 是否讀寫記憶        |
| (g_{expert})      | 是否啟用 expert   |
| (g_{core})        | 是否更新核心        |
| (g_{replay})      | 是否啟動 replay   |
| (g_{verify})      | 是否啟動 verifier |
| (g_{safety})      | 是否啟動安全檢查      |
| (g_{consolidate}) | 是否進行整合        |

---

# 10. Task / Context Identity Inference

## 10.1 描述

模型要知道目前處於什麼任務或情境。

[
z_t = ContextNet(h_t, M_t)
]

可能的 context：

```text
數學推理
程式設計
使用者偏好
新事實
舊事實
暫時知識
安全敏感任務
技能學習
多版本知識
```

---

## 10.2 Context 用途

[
R_t = Router(z_t)
]

[
M_{selected} = MemorySelect(z_t)
]

[
A_{selected} = ExpertSelect(z_t)
]

---

# 11. Novelty / Surprise / Conflict Detector

## 11.1 Novelty

判斷是否為新知識：

[
novelty_t =
1 -
\max_i sim(h_t, k_i)
]

若：

[
novelty_t > \tau_n
]

表示此輸入包含新資訊。

---

## 11.2 Surprise

判斷模型是否真的感到意外：

[
surprise_t = L(f_{\theta}(x_t), y_t)
]

或：

[
surprise_t = -\log p_\theta(y_t|x_t)
]

---

## 11.3 Conflict

判斷新知識是否與舊記憶衝突：

[
conflict_t =
ConflictNet(h_t, M_t)
]

也可以定義為：

[
conflict_t =
1 -
sim(v_{new}, v_{retrieved})
]

若：

[
conflict_t > \tau_c
]

則不覆蓋舊記憶，而是建立版本。

---

# 12. Uncertainty / Confidence Calibrator

## 12.1 描述

模型需要知道自己是否確定。

[
u_t = UncertaintyHead(h_t)
]

[
confidence_t = 1 - u_t
]

---

## 12.2 決策規則

| 狀態      | 行為                |
| ------- | ----------------- |
| 高信心、低衝突 | 直接回答              |
| 低信心、高新奇 | 寫入暫存記憶            |
| 高衝突     | 啟動 verifier       |
| 低可信來源   | 不進入長期記憶           |
| 知識可能過期  | 進 temporal memory |

---

# 13. Neural Controller

## 13.1 描述

Controller 是整個神經調控系統的中樞。

[
C_\phi
]

輸入：

[
[
h_t,
r_t,
novelty_t,
surprise_t,
conflict_t,
uncertainty_t,
source_t,
time_t,
loss_t
]
]

輸出：

[
[
g_{read},
g_{write},
g_{adapter},
g_{core},
g_{replay},
g_{verify},
g_{rollback},
g_{consolidate}
]
]

---

## 13.2 Controller 目標

Controller 不是只學單次準確率，而是學長期表現：

[
J =
\sum_{t=1}^{T}
Performance_t
-------------

## \alpha Forgetting_t

## \beta ComputeCost_t

## \gamma MemoryCost_t

\delta SafetyRisk_t
]

---

# 14. Neural Importance Estimator

## 14.1 描述

重要性估計器判斷哪些權重不能亂動。

[
I_t = ImportanceNet(h_t, g_t, usage_t, replay_loss_t)
]

每個權重或模組都有重要性：

[
I_j \in [0, 1]
]

---

## 14.2 EWC 形式

[
L_{ewc}
=======

\sum_j
I_j
(\theta_j - \theta_j^*)^2
]

總 loss：

[
L
=

L_{new}
+
\lambda_{ewc} L_{ewc}
]

其中：

[
\lambda_{ewc}
=============

C_\phi(conflict_t, uncertainty_t, importance_t)
]

---

# 15. Metaplasticity State

## 15.1 描述

後設可塑性表示：

> 不只是權重會變，權重的「可變程度」也會變。

每個權重有一個可塑性狀態：

[
p_j \in [0, 1]
]

更新：

[
\theta_j^{t+1}
==============

## \theta_j^t

\eta
p_j
g_j
]

---

## 15.2 可塑性分層

| 權重類型            | 可塑性 |
| --------------- | --: |
| Working memory  |   高 |
| Episodic memory |   高 |
| Adapter         |   中 |
| Semantic memory |  中低 |
| Core model      |   低 |
| Safety kernel   |  極低 |

---

# 16. Homeostatic Plasticity Controller

## 16.1 描述

穩態控制器防止某些 memory 或 expert 過度活躍。

[
H_t = Homeostasis(M_t, A_t, R_t)
]

---

## 16.2 控制目標

```text
防止 expert collapse
防止 memory 污染
防止路由偏置
防止過度適應
防止核心被快速改壞
```

---

## 16.3 使用率懲罰

若某 expert 過度使用：

[
usage(E_i) > \tau_u
]

則降低其 gate：

[
\alpha_i' =
\alpha_i
--------

\lambda usage(E_i)
]

---

# 17. Learned Optimizer

## 17.1 描述

Optimizer 本身由神經網路學出來。

傳統更新：

[
\theta_{t+1}
============

## \theta_t

\eta \nabla L
]

神經更新：

[
\Delta \theta_t
===============

O_\omega(
g_t,
I_t,
P_t,
conflict_t,
memory_t
)
]

[
\theta_{t+1}
============

\theta_t
+
\Delta \theta_t
]

---

## 17.2 Optimizer 輸出

```text
每層 learning rate
每層 update mask
是否更新 core
是否更新 adapter
是否只寫 memory
是否需要 replay
是否啟動 rollback
```

---

# 18. Learned Gradient Projector

## 18.1 描述

避免新梯度破壞舊知識。

新梯度：

[
g_{new}
=======

\nabla_\theta L_{new}
]

舊知識梯度基底：

[
U_{old}
=======

[g_1, g_2, ..., g_k]
]

安全梯度：

[
g_{safe}
========

Projector_\omega(g_{new}, U_{old}, I_t)
]

---

## 18.2 理想約束

[
g_{safe}^\top g_{old}^{(i)} \geq 0
]

代表新更新不能讓舊任務變差。

---

## 18.3 更新

[
\theta_{t+1}
============

## \theta_t

\eta g_{safe}
]

---

# 19. Dream Replay Generator

## 19.1 描述

模型不一定保存舊資料，而是內部生成舊經驗。

[
\hat{x}*{old}, \hat{y}*{old}
============================

G_\psi(M_t, z)
]

---

## 19.2 Replay Loss

[
L_{replay}
==========

L(f_\theta(\hat{x}*{old}), \hat{y}*{old})
]

---

## 19.3 Replay 類型

| 類型              | 說明                |
| --------------- | ----------------- |
| Input replay    | 生成舊輸入             |
| Label replay    | 生成舊答案             |
| Latent replay   | 生成舊 hidden states |
| Logit replay    | 生成舊輸出分布           |
| Boundary replay | 生成舊決策邊界樣本         |
| Sentinel replay | 生成關鍵舊能力測試         |

---

# 20. Shadow Teacher / Distillation

## 20.1 描述

保留慢速教師，用來約束學生不要偏離舊能力。

[
\theta_{teacher}^{t+1}
======================

\tau \theta_{teacher}^{t}
+
(1-\tau)\theta_{student}^{t}
]

其中：

[
\tau \approx 1
]

---

## 20.2 Distillation Loss

[
L_{distill}
===========

KL(
p_{teacher}(y|x)
|
p_{student}(y|x)
)
]

---

## 20.3 壓縮教師記憶

若不保存完整 teacher，可保存：

```text
old logits sketches
activation prototypes
task centroids
behavior anchors
latent constraints
```

---

# 21. Verifier / Critic

## 21.1 描述

內部驗證器檢查：

```text
新知識是否可信？
新答案是否一致？
更新是否破壞舊能力？
記憶是否被污染？
adapter 是否錯誤啟用？
是否存在安全風險？
```

---

## 21.2 Critic 類型

| Critic            | 功能       |
| ----------------- | -------- |
| FactCritic        | 事實一致性    |
| ConsistencyCritic | 內部邏輯一致性  |
| CausalCritic      | 因果合理性    |
| ForgettingCritic  | 是否遺忘     |
| SafetyCritic      | 安全風險     |
| LocalityCritic    | 是否影響無關輸入 |

---

## 21.3 驗證分數

[
v_t =
Verifier(x_t, y_t, M_t, A_t, \theta_t)
]

若：

[
v_t < \tau_v
]

則禁止寫入長期記憶或核心權重。

---

# 22. Locality Constraint

## 22.1 描述

更新新知識時，不應影響無關輸入。

局部性 loss：

[
L_{locality}
============

|
f_{\theta'}(x_{unrelated})
--------------------------

f_{\theta}(x_{unrelated})
|^2
]

---

## 22.2 總更新目標

[
L
=

L_{new}
+
\lambda_1 L_{replay}
+
\lambda_2 L_{distill}
+
\lambda_3 L_{ewc}
+
\lambda_4 L_{locality}
]

---

# 23. Inhibition Gate

## 23.1 描述

記憶不是只有讀取，還要能抑制。

某些記憶可能：

```text
過期
低可信
不適用當前情境
和新版知識衝突
只適合特定使用者
具有安全風險
```

---

## 23.2 抑制公式

[
inhibit_i
=========

InhibitionNet(q_t, context_t, time_t, source_i, conflict_i)
]

記憶讀取：

[
r_t
===

\sum_i
a_i
(1 - inhibit_i)
v_i
]

---

# 24. Rollback / Versioning Mechanism

## 24.1 描述

持續學習一定會學錯，因此必須可以回滾。

每次更新保存：

[
u_t =
[
\Delta M_t,
\Delta A_t,
\Delta \theta_t,
reason_t,
confidence_t,
time_t
]
]

---

## 24.2 回滾

[
\mathcal{S}*{t+1}
\rightarrow
\mathcal{S}*{t}
]

近似形式：

[
\theta_{rollback}
=================

## \theta_{current}

\Delta \theta_t
]

---

## 24.3 更安全的版本化

```text
不直接覆蓋舊 adapter
而是建立 adapter_v2

不直接刪除舊記憶
而是標記為 inhibited / outdated

不直接改核心
而是先用低秩可逆更新
```

---

# 25. Capacity Manager

## 25.1 描述

模型容量有限，所以要管理 memory 與 expert。

[
cap_t =
CapacityManager(M_t, A_t, \theta_t)
]

---

## 25.2 記憶價值評分

[
value_i
=======

usage_i
\cdot
confidence_i
\cdot
generality_i
------------

## cost_i

conflict_i
]

若：

[
value_i < \tau_{drop}
]

則：

```text
降低權重
合併
壓縮
抑制
刪除
```

---

## 25.3 Expert 壓縮

多個 LoRA expert：

[
\Delta W
========

\sum_i
\alpha_i B_i A_i
]

壓縮成：

[
\Delta W
\approx
B' A'
]

---

# 26. Schema Abstractor

## 26.1 描述

模型不應只記住單筆資料，還要抽象出規則。

從多個 episodic memories：

[
m_1, m_2, ..., m_n
]

抽象出 schema：

[
schema =
Abstractor(m_1, m_2, ..., m_n)
]

---

## 26.2 記憶升級

```text
單次事件
  ↓
重複模式
  ↓
穩定事實
  ↓
抽象規則
  ↓
可泛化技能
  ↓
核心知識
```

---

# 27. Sleep Consolidator

## 27.1 描述

睡眠整合模組在閒置時運作，負責長期壓縮與穩定。

[
\mathcal{S}_{t+1}
=================

Consolidator(\mathcal{S}_t)
]

---

## 27.2 Consolidation 任務

```text
合併相似記憶
版本化衝突記憶
壓縮 adapter
刪除低價值記憶
生成 replay
更新 importance map
更新 router
把穩定知識蒸餾進 core
```

---

## 27.3 睡眠訓練目標

[
L_{sleep}
=========

L_{replay}
+
L_{distill}
+
L_{compression}
+
L_{stability}
+
L_{abstraction}
]

---

# 28. Sentinel / Canary Test Memory

## 28.1 描述

模型內部保存一組小型哨兵測試，用於快速檢查是否遺忘。

[
Q_{sentinel}
============

{
q_1,
q_2,
...,
q_n
}
]

---

## 28.2 Sentinel Loss

[
L_{sentinel}
============

\sum_i
L(f_\theta(q_i), a_i)
]

---

## 28.3 用途

```text
檢查語言能力
檢查數學能力
檢查安全邊界
檢查舊知識
檢查使用者偏好
檢查 routing 是否偏移
```

---

# 29. Safety / Alignment Gate

## 29.1 描述

安全與對齊不能和普通知識同等可塑。

[
s_t =
SafetyCritic(x_t, y_t, update_t, memory_t)
]

若：

[
s_t > \tau_{risk}
]

則禁止更新。

---

## 29.2 安全分層

| 層級     |     可塑性 |
| ------ | ------: |
| 普通知識   |       高 |
| 使用者偏好  |       中 |
| 任務技巧   |       中 |
| 模型核心能力 |       低 |
| 安全規則   |      極低 |
| 權限控制   | 不可塑或硬約束 |

---

## 29.3 安全更新限制

[
plasticity_{safety}
\approx 0
]

也就是安全核心幾乎不可被本地持續學習改寫。

---

# 30. 完整 Loss Function

整體訓練目標：

[
L_{total}
=========

L_{new}
+
\lambda_1 L_{replay}
+
\lambda_2 L_{distill}
+
\lambda_3 L_{ewc}
+
\lambda_4 L_{locality}
+
\lambda_5 L_{sentinel}
+
\lambda_6 L_{safety}
+
\lambda_7 L_{capacity}
+
\lambda_8 L_{homeostasis}
]

其中：

| Loss              | 作用       |
| ----------------- | -------- |
| (L_{new})         | 學習新知識    |
| (L_{replay})      | 保持舊知識    |
| (L_{distill})     | 維持舊模型行為  |
| (L_{ewc})         | 保護重要權重   |
| (L_{locality})    | 限制更新影響範圍 |
| (L_{sentinel})    | 哨兵測試     |
| (L_{safety})      | 安全約束     |
| (L_{capacity})    | 控制容量增長   |
| (L_{homeostasis}) | 維持系統穩態   |

---

# 31. 完整更新流程

```text
for each new experience (x_t, y_t):

    # 1. 核心模型理解輸入
    h_t = CoreTransformer(x_t)

    # 2. 推斷任務與情境
    z_t = ContextInferencer(h_t)

    # 3. 檢查新奇性、驚訝度、衝突與不確定性
    novelty_t = NoveltyHead(h_t, M_t)
    surprise_t = SurpriseHead(loss_t)
    conflict_t = ConflictHead(h_t, M_t)
    uncertainty_t = UncertaintyHead(h_t)

    # 4. Controller 決定行為
    gates_t = Controller(
        h_t,
        z_t,
        novelty_t,
        surprise_t,
        conflict_t,
        uncertainty_t
    )

    # 5. 讀取記憶
    r_t = MemoryRead(M_t, h_t)

    # 6. 選擇 expert / adapter
    experts_t = Router(h_t, r_t, z_t)

    # 7. 產生輸出
    y_pred = Core + Memory + Experts

    # 8. Verifier 檢查
    verify_score = Verifier(x_t, y_pred, M_t)

    # 9. Safety gate 檢查
    safety_score = SafetyCritic(x_t, y_pred, update_t)

    # 10. 若通過檢查，計算 loss
    L_new = TaskLoss(y_pred, y_t)
    L_replay = ReplayLoss(DreamGenerator(M_t))
    L_distill = DistillLoss(Teacher, Student)
    L_ewc = ImportancePenalty(I_t)
    L_locality = LocalityLoss()
    L_sentinel = SentinelLoss()

    L_total =
        L_new
        + λ1 L_replay
        + λ2 L_distill
        + λ3 L_ewc
        + λ4 L_locality
        + λ5 L_sentinel

    # 11. 產生梯度
    g_t = grad(L_total)

    # 12. Learned optimizer / gradient projector 產生安全更新
    g_safe = NeuralGradientProjector(g_t, I_t, conflict_t)

    # 13. 根據 gates 更新 memory / adapter / core
    update Memory if g_write > threshold
    update Adapter if g_adapter > threshold
    update Core only if g_core is high and risk is low

    # 14. 保存 rollback trace
    save ΔM_t, ΔA_t, Δθ_t

    # 15. 更新 importance / plasticity / homeostasis
    I_t = ImportanceNet(...)
    P_t = MetaplasticityUpdate(...)
    H_t = HomeostasisUpdate(...)

    # 16. 若需要，進入 sleep consolidation
    if g_consolidate > threshold:
        SleepConsolidator(S_t)
```

---

# 32. 分層更新策略

## 32.1 新知識寫入決策

| 狀態       | 寫入位置             |
| -------- | ---------------- |
| 單次、低信心資料 | Working Memory   |
| 新但未驗證事實  | Episodic Memory  |
| 多次驗證事實   | Semantic Memory  |
| 有時間效性的知識 | Temporal Memory  |
| 穩定技能     | Adapter / Expert |
| 高泛化規則    | Core Model       |
| 衝突知識     | Versioned Memory |
| 高風險資料    | 不寫入或隔離           |

---

## 32.2 更新風險等級

| 更新位置            | 風險 | 策略      |
| --------------- | -: | ------- |
| Working Memory  |  低 | 可快速更新   |
| Episodic Memory |  低 | 可寫入、可抑制 |
| Semantic Memory |  中 | 需驗證     |
| Adapter         |  中 | 可回滾     |
| Core            |  高 | 慢速更新    |
| Safety Kernel   | 極高 | 幾乎不可更新  |

---

# 33. 系統閉環

完整持續學習閉環：

```text
Observe
  ↓
Understand
  ↓
Detect novelty / conflict / uncertainty
  ↓
Verify
  ↓
Decide whether to learn
  ↓
Choose where to store
  ↓
Apply safe update
  ↓
Replay old knowledge
  ↓
Check forgetting
  ↓
Rollback if needed
  ↓
Consolidate during sleep
  ↓
Update plasticity and routing
```

數學表示：

[
observe
\rightarrow
evaluate
\rightarrow
gate
\rightarrow
update
\rightarrow
verify
\rightarrow
rollback/consolidate
]

---

# 34. 最終模型定義

可以將整體模型定義為：

[
f_t(x)
======

F(
x;
\theta_c,
M_t,
A_t,
R_t,
C_t
)
]

其中推理為：

[
h_t = F_{\theta_c}(x)
]

[
r_t = Read(M_t, h_t)
]

[
e_t = ExpertRoute(A_t, h_t, r_t)
]

[
y_t =
Decoder(h_t + r_t + e_t)
]

學習為：

[
\mathcal{S}_{t+1}
=================

U_\phi(
\mathcal{S}_t,
x_t,
y_t,
feedback_t
)
]

---

# 35. 一句話總結

這個架構不是單純讓小模型反覆 fine-tune，而是讓模型成為一個具備神經可塑性的系統：

```text
穩定核心負責通用能力；
可微分記憶負責快速學習；
LoRA / Adapter expert 負責局部技能；
Router 負責情境選擇；
Controller 負責調控學習；
Importance / Metaplasticity 保護舊知識；
Dream Replay / Distillation 維持舊行為；
Verifier / Safety Gate 防止錯誤學習；
Rollback / Inhibition 修正污染；
Sleep Consolidation 把穩定知識壓縮成長期能力。
```

最終形式：

[
\boxed{
\mathcal{S}_{t+1}
=================

NeuralContinualUpdate(
\mathcal{S}_t,
x_t,
y_t,
feedback_t
)
}
]

也就是：

> **模型不只是學知識，而是學會如何安全地修改自己的記憶、權重、可塑性與行為策略。**

