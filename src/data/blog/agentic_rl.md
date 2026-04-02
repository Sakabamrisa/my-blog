---
title: 浅谈Agentic RL
pubDatetime: 2026-04-02
description: 这篇博客试图用最直白的语言，把从经典强化学习到今天大模型训练中用到的GRPO，一条线串起来讲清楚。如果你对RL有一点点了解但又觉得公式劝退，希望这篇文章能帮到你。
tags: [强化学习, Agentic RL, TRPO, PPO, GRPO, 技术， 浅谈]
category: 技术
draft: false
---

# 从RL到Agentic RL：一篇讲透强化学习如何驱动大模型
> 这篇博客试图用最直白的语言，把从经典强化学习到今天大模型训练中用到的GRPO，一条线串起来讲清楚。如果你对RL有一点点了解但又觉得公式劝退，希望这篇文章能帮到你。
>

---

## 一、从RL到Agentic RL
### 1.1 详解MDP，及在LLM范式下的MDP
强化学习的所有故事，都要从一个叫 **MDP（Markov Decision Process，马尔可夫决策过程）** 的数学框架讲起。

MDP说白了就是在回答一个问题：**一个智能体（Agent），在一个环境里，怎么通过不断做决策来获得最多的奖励？**

一个标准的MDP由五元组定义：

$ \text{MDP} = (S, A, P, R, \gamma) $

我们一个一个拆开来看：

+ $ S $：**状态空间（State Space）**。就是智能体所能处在的所有状态的集合。你可以想象成一个棋盘上棋子的所有可能摆法。
+ $ A $：**动作空间（Action Space）**。在每个状态下，智能体能做的所有动作的集合。比如"往左走""往右走""开火"。
+ $ P(s'|s, a) $：**状态转移概率（Transition Probability）**。在状态 $ s $ 下执行动作 $ a $ 后，环境以概率 $ P $ 转移到新状态 $ s' $。这个函数描述的是"环境的规则"。
+ $ R(s, a) $：**奖励函数（Reward Function）**。在状态 $ s $ 下执行动作 $ a $，环境给你的即时奖励。这是你优化的目标信号。
+ $ \gamma $：**折扣因子（Discount Factor）**，取值范围 $ [0, 1] $。它表示你有多在乎未来的奖励。$ \gamma = 0 $ 意味着只看眼前，$ \gamma = 1 $ 意味着未来的每一分奖励和现在一样重要。

这就是经典MDP。但现在问题来了——**LLM的场景下，MDP长什么样？**

其实非常优雅。我们把LLM生成文本的过程，也建模成一个MDP：

+ **状态** $ s_t $：就是 prompt 加上模型已经生成的所有 token。比如用户问"什么是RL？"，模型已经生成了"强化学习是"，那当前状态就是 `[什么是RL？| 强化学习是]`。每生成一个新 token，状态就往后推一步。
+ **动作** $ a_t $：就是模型在当前时刻选择输出的那个 token。动作空间就是整个词表（vocabulary），通常有几万到十几万个候选。
+ **状态转移** $ P(s_{t+1}|s_t, a_t) $：这里特别简单，几乎是确定性的——新状态就是旧状态拼接上刚生成的 token。也就是说 $ s_{t+1} = [s_t; a_t] $，概率为1。环境的"物理规则"就是字符串拼接。
+ **奖励** $ R $：这是最有讲究的部分。在LLM的RLHF训练中，奖励通常在**整个回复生成完毕后**才给出一个分数（比如通过一个奖励模型打分）。中间每一步的即时奖励通常是0。这种"只在结尾给奖励"的设定叫做 **sparse reward（稀疏奖励）**，也是LLM强化学习中的一大难点。
+ **折扣因子** $ \gamma $：在LLM场景中，通常取接近1的值（比如 $ \gamma = 1.0 $），因为我们希望模型同等重视回复开头和结尾的质量。

看到了吗？LLM生成文本的过程，本质上就是一个**序列决策问题**：在每一步，根据当前上下文（状态），选择一个token（动作），直到生成结束，拿到一个奖励。这就是为什么RL能天然地应用到LLM训练中。

而所谓 **Agentic RL**，就是把这个范式进一步扩展：模型不仅仅是在生成token，它还可以调用工具、浏览网页、执行代码、和环境交互。每一次"行动"的粒度从一个token变成了一个完整的"操作"（比如"调用搜索引擎"），但底层的MDP框架是完全一样的。

### 1.2 策略：神经网络如何表示 $ \pi $？
在MDP里，智能体需要一个**策略（Policy）** 来指导自己在每个状态下该怎么做。策略用 $ \pi $ 表示，数学上写成：

$ \pi(a|s) $

这个符号的意思是：**在状态 $ s $ 下，选择动作 $ a $ 的概率。** 策略本质上就是一个条件概率分布——给定当前状态，它告诉你每个动作被选中的概率是多少。

那在LLM的语境下，$ \pi $ 是什么？**就是模型本身。**

具体来说，一个LLM（比如GPT、LLaMA、DeepSeek）就是一个参数化的策略 $ \pi_\theta $，其中 $ \theta $ 代表模型的所有参数（几十亿甚至上千亿个浮点数）。给定当前上下文 $ s_t $（prompt + 已生成的token），模型通过前向传播，在最后一层输出一个 softmax 分布：

$ \pi_\theta(a_t | s_t) = \text{softmax}(f_\theta(s_t)) $

这里 $ f_\theta(s_t) $ 是模型对每个词表中token的原始打分（logits），softmax 把它们变成概率分布。于是，$ \pi_\theta(a_t|s_t) $ 就是"模型觉得下一个token应该是 $ a_t $ 的概率"。

所以，**训练LLM的策略，就是调整 $ \theta $，让模型在面对各种输入时，能选出更好的token序列。** 这就是RL在LLM训练中的核心目标。

### 1.3 如何从梯度的角度来更新策略：策略梯度方法
好了，现在我们知道策略 $ \pi_\theta $ 是一个神经网络，目标是最大化累积奖励。那怎么优化 $ \theta $ 呢？

先定义我们的优化目标。我们想最大化的东西是**期望回报**：

$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] $

逐个拆解：

+ $ J(\theta) $：这就是我们要最大化的目标函数，它是关于模型参数 $ \theta $ 的函数。
+ $ \mathbb{E}_{\tau \sim \pi_\theta} $：期望值，其中 $ \tau $ 是一条完整的轨迹（trajectory），也就是一整个状态-动作序列 $ (s_0, a_0, s_1, a_1, \ldots, s_T, a_T) $。$ \tau \sim \pi_\theta $ 意味着这条轨迹是按照策略 $ \pi_\theta $ 采样出来的。
+ $ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) $：沿着轨迹，把每一步的奖励乘以折扣因子后加起来。这就是这条轨迹的总回报。

想要最大化 $ J(\theta) $，最自然的想法就是对 $ \theta $ 求梯度，然后做梯度上升（gradient ascent）。这就是大名鼎鼎的**策略梯度定理（Policy Gradient Theorem）**：

$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right] $

这个公式看着吓人，但核心思想极其优美。逐符号拆解：

+ $ \nabla_\theta J(\theta) $：目标函数对参数 $ \theta $ 的梯度。我们想知道"参数往哪个方向调，目标函数会增大"。
+ $ \nabla_\theta \log \pi_\theta(a_t|s_t) $：这是 $ \log $ 概率对参数的梯度。$ \log \pi_\theta(a_t|s_t) $ 就是模型在状态 $ s_t $ 下选择动作 $ a_t $ 的对数概率。对参数求梯度后，它告诉我们"参数往哪个方向调，能让这个动作的概率增大"。
+ $ G_t $：这是从时刻 $ t $ 开始到结束的**累积回报（Return）**，定义为 $ G_t = \sum_{k=t}^{T} \gamma^{k-t} R(s_k, a_k) $。简单说就是"从这一步往后，你总共能拿到多少奖励"。

这个公式的直觉是什么？**如果一个动作带来了高回报（$ G_t $ 大），就增大它的概率；如果一个动作带来了低回报甚至负回报，就减小它的概率。** 这就是策略梯度的灵魂——用结果的好坏来"加权"地调整每个动作的概率。

最基础的策略梯度算法叫 **REINFORCE**，它就是直接用采样到的轨迹来估计上面这个梯度，然后做梯度上升。但REINFORCE有一个致命问题：**方差太大**。因为 $ G_t $ 是从一条采样轨迹算出来的，不同轨迹之间的 $ G_t $ 可能差异巨大，导致梯度估计非常不稳定。

怎么办？这就引出了Actor-Critic框架。

---

## 二、Actor-Critic框架
REINFORCE的问题在于，用整条轨迹的回报 $ G_t $ 来评估每个动作的好坏，太粗糙了。我们能不能引入一个"评委"来更精确地评价每个动作？

这就是 **Actor-Critic** 的核心思想：

+ **Actor（演员）**：就是策略网络 $ \pi_\theta $，负责选动作。
+ **Critic（评委）**：是另一个网络，负责评估"当前状态有多好"或"某个动作有多好"。

### 2.1 优势函数——Q-value的"归一化"
在讲优势函数之前，我们先快速过一下 **Q-value（动作价值函数）** 和 **V-value（状态价值函数）**。

**V-value** $ V^\pi(s) $ 表示：如果我在状态 $ s $，之后一直按照策略 $ \pi $ 来行动，我期望能拿到多少总回报。数学定义是：

$ V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R(s_{t+k}, a_{t+k}) \mid s_t = s \right] $

就是"在状态 $ s $ 下按策略 $ \pi $ 走下去的期望回报"。

**Q-value** $ Q^\pi(s, a) $ 则更细一层，它表示：在状态 $ s $ 下，**先执行动作** $ a $，之后再按策略 $ \pi $ 行动，期望拿到多少总回报：

$ Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R(s_{t+k}, a_{t+k}) \mid s_t = s, a_t = a \right] $

$ V $ 和 $ Q $ 的关系很直观：$ V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] $。也就是说，状态价值就是所有动作价值在策略下的期望。

现在，重点来了。**优势函数（Advantage Function）** 定义为：

$ A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) $

逐个看：

+ $ A^\pi(s, a) $：在状态 $ s $ 下，选择动作 $ a $ 相对于"平均水平"的**优势**。
+ $ Q^\pi(s, a) $：选择动作 $ a $ 能带来的期望回报。
+ $ V^\pi(s) $：在这个状态下"随便按策略选"能带来的平均期望回报。

优势函数的含义极其清晰：**如果 $ A > 0 $，说明这个动作比平均水平好；如果 $ A < 0 $，说明这个动作比平均水平差。**

为什么叫"归一化"？因为 $ V^\pi(s) $ 就像一个 baseline（基线）。不同状态下，Q-value的绝对值可能差异巨大（有的状态本身就很好，所有动作的Q值都高），减去V就消除了状态本身好坏的影响，只留下"这个动作相对于其他动作有多好"的信息。这和我们对数据做减均值的归一化是一个道理。

有了优势函数，策略梯度公式就变成了：

$ \nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right] $

用优势函数代替原来的 $ G_t $，方差会显著降低，因为我们减掉了那个会导致大幅波动的baseline。

### 2.2 Critic网络
Critic网络的任务很明确：**估计 $ V^\pi(s) $，也就是状态价值函数。**

在实际实现中，Critic是一个参数为 $ \phi $ 的神经网络 $ V_\phi(s) $，它的训练目标是最小化**TD误差（Temporal Difference Error）**：

$ \delta_t = R(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) $

拆解一下：

+ $ R(s_t, a_t) $：在时刻 $ t $ 实际拿到的即时奖励。
+ $ \gamma V_\phi(s_{t+1}) $：对下一个状态价值的估计（乘以折扣因子）。
+ $ R(s_t, a_t) + \gamma V_\phi(s_{t+1}) $：这是用"实际奖励 + 对未来的估计"得到的 $ V(s_t) $ 的估计值，叫做 **TD target**。
+ $ V_\phi(s_t) $：Critic当前对 $ V(s_t) $ 的预测。
+ $ \delta_t $：TD target和当前预测之间的差距，就是TD误差。

Critic的训练就是最小化 $ \delta_t^2 $，让自己的预测越来越准确。

在LLM的RLHF训练中（比如PPO），Critic网络通常是从预训练模型或奖励模型初始化的，然后和Actor一起训练。这意味着你需要同时在GPU上放两个大模型——一个Actor（策略模型），一个Critic（价值模型），外加一个冻结的参考模型和奖励模型。这就是为什么PPO训练LLM的显存开销那么大。

### 2.3 GAE：多步TD误差的累加
前面讲的TD误差 $ \delta_t $ 是一步的。但一步的估计偏差可能很大（高偏差，低方差）。如果我们用很多步的实际奖励来替代估计呢？偏差会降低，但方差会增大。有没有办法在偏差和方差之间取一个平衡？

**GAE（Generalized Advantage Estimation，广义优势估计）** 就是干这个的。GAE的公式如下：

$ \hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $

逐个拆解这个公式中的新符号：

+ $ \hat{A}_t^{\text{GAE}} $：时刻 $ t $ 的GAE优势估计。头上的 $ \hat{} $ 表示这是一个估计值。
+ $ \lambda $：GAE的超参数，取值范围 $ [0, 1] $。这是偏差-方差的权衡旋钮。
+ $ (\gamma \lambda)^l $：指数衰减权重。$ l $ 是"往未来看几步"，$ \gamma \lambda $ 的 $ l $ 次方意味着越远的TD误差权重越小。
+ $ \delta_{t+l} $：未来第 $ l $ 步的TD误差，即 $ \delta_{t+l} = R_{t+l} + \gamma V_\phi(s_{t+l+1}) - V_\phi(s_{t+l}) $。

GAE的本质是什么？就是把从当前时刻开始、未来每一步的TD误差 $ \delta $，用一个指数衰减的权重加起来。

两个极端情况很好理解：

+ 当 $ \lambda = 0 $ 时：$ \hat{A}_t = \delta_t $，退化为一步TD误差。偏差高，方差低。
+ 当 $ \lambda = 1 $ 时：$ \hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} $，等价于用完整的蒙特卡洛回报减去baseline。偏差低，方差高。

实际中 $ \lambda $ 通常取 0.95 左右，在偏差和方差之间找到一个甜蜜点。

展开来写，GAE也可以递归计算：

$ \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1} $

这个递推式实现起来非常高效——从轨迹末尾往前算就行了。

---

## 三、TRPO-PPO-GRPO
好了，铺垫终于做完了。现在进入正题——现代策略优化的三大核心算法：TRPO、PPO和GRPO。它们是一脉相承的演进关系，后一个是前一个的简化和改进。

### 3.1 TRPO（Trust Region Policy Optimization）
#### Pipeline
TRPO的训练流程大致是这样的：

1. 用当前策略 $ \pi_{\theta_{\text{old}}} $ 采集一批轨迹数据。
2. 用这批数据计算每个时刻的优势函数 $ \hat{A}_t $（用GAE）。
3. 构造一个"替代目标函数"（surrogate objective），衡量新策略相比旧策略的提升。
4. 在约束条件下（新旧策略之间的KL散度不超过某个阈值）最大化这个替代目标。
5. 用共轭梯度法和线搜索来求解这个约束优化问题。
6. 更新参数，回到第1步。

#### 创新在哪？
在TRPO之前，策略梯度的一个大问题是：**你不知道步子该迈多大。**

如果学习率太大，策略可能一步就跑偏了，导致采集到的新数据变差，然后越训越差（这叫策略崩溃）。如果学习率太小，训练慢得让人想砸键盘。

TRPO的核心创新是引入了**信任域（Trust Region）**的概念。它不是简单地走一步梯度，而是说：

> "我可以更新策略，但新策略不能和旧策略差太多。具体来说，新旧策略之间的 **KL散度** 不能超过一个阈值 $ \delta $。"
>

数学上，TRPO解决的是这个约束优化问题：

$ \max_\theta \quad \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] $

$ \text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)) \right] \leq \delta $

逐个拆解：

+ $ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $：这个比值叫做**重要性采样比（Importance Sampling Ratio）**，通常记为 $ r_t(\theta) $。因为我们的数据是用旧策略 $ \pi_{\theta_{\text{old}}} $ 采集的，但我们想评估新策略 $ \pi_\theta $ 有多好，所以要用这个比值来修正。如果新策略选某个动作的概率是旧策略的2倍，这个比值就是2。
+ $ \hat{A}_t $：GAE估计的优势函数。
+ $ r_t(\theta) \cdot \hat{A}_t $：这就是"替代目标函数"。直觉是：如果一个动作的优势 $ \hat{A}_t > 0 $（好动作），我们希望 $ r_t $ 大一些（增大这个动作的概率）；反之，减小概率。
+ $ D_{\text{KL}} $：KL散度，衡量两个概率分布之间的差异。$ D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) $ 越大，两个策略越不一样。
+ $ \delta $：信任域的大小，一个超参数，通常取 0.01 左右。

TRPO的理论保证很强：**只要在信任域内更新，新策略的性能就一定不会比旧策略差太多。** 这是它的最大卖点。

#### 代码详解
TRPO的实现比较复杂，因为它需要用**共轭梯度法（Conjugate Gradient）** 来近似求解约束优化问题，还需要**线搜索（Line Search）** 来确保KL约束确实被满足。

```python
import torch
import torch.nn as nn
import numpy as np

def conjugate_gradient(Avp_fn, b, nsteps=10, residual_tol=1e-10):
    """
    共轭梯度法：用来求解 Ax = b，其中 A 是Fisher信息矩阵（二阶导数矩阵）。
    我们不显式构造A，而是通过 Avp_fn（矩阵-向量积函数）来隐式使用A。
    
    参数:
    - Avp_fn: 一个函数，输入向量v，返回 A @ v
    - b: 等式右边的向量（这里就是策略梯度）
    - nsteps: 迭代次数
    - residual_tol: 残差容忍度，足够小就提前停止
    """
    x = torch.zeros_like(b)  # 初始解设为零向量
    r = b.clone()             # 初始残差 r = b - A @ x = b（因为x=0）
    p = b.clone()             # 初始搜索方向
    rdotr = torch.dot(r, r)   # 残差的内积
    
    for _ in range(nsteps):
        Avp = Avp_fn(p)                     # 计算 A @ p
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)  # 步长
        x += alpha * p                       # 更新解
        r -= alpha * Avp                     # 更新残差
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:         # 残差够小了就停
            break
        beta = new_rdotr / rdotr             # 更新搜索方向的系数
        p = r + beta * p                     # 新的搜索方向
        rdotr = new_rdotr
    return x


def compute_kl(old_policy, new_policy, states):
    """
    计算新旧策略之间的平均KL散度。
    这就是TRPO约束条件中的那个东西。
    """
    with torch.no_grad():
        old_dist = old_policy(states)  # 旧策略的动作分布
    new_dist = new_policy(states)       # 新策略的动作分布
    # 用PyTorch自带的kl_divergence计算
    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    return kl


def trpo_update(policy, value_fn, trajectories, max_kl=0.01, gamma=0.99, lam=0.95):
    """
    TRPO的一次策略更新。
    
    参数:
    - policy: 策略网络（Actor）
    - value_fn: 价值网络（Critic）
    - trajectories: 一批采集到的轨迹
    - max_kl: 信任域大小 δ
    - gamma: 折扣因子
    - lam: GAE的λ
    """
    states, actions, rewards, next_states, dones = trajectories
    
    # ===== Step 1: 计算GAE优势函数 =====
    with torch.no_grad():
        values = value_fn(states)          # V(s_t)
        next_values = value_fn(next_states) # V(s_{t+1})
    
    # 计算TD误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # 从后往前递推计算GAE
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        # A_t = δ_t + γλ * A_{t+1}，这就是GAE的递推式
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 归一化优势函数（让均值为0，标准差为1，进一步降低方差）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # ===== Step 2: 计算策略梯度 =====
    old_dist = policy(states)
    old_log_probs = old_dist.log_prob(actions).detach()
    
    def compute_loss():
        dist = policy(states)
        log_probs = dist.log_prob(actions)
        # 重要性采样比 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs)
        # 替代目标函数
        return (ratio * advantages).mean()
    
    loss = compute_loss()
    # 策略梯度 g = ∇_θ L(θ)
    grads = torch.autograd.grad(loss, policy.parameters())
    flat_grads = torch.cat([g.view(-1) for g in grads])
    
    # ===== Step 3: 计算Fisher矩阵向量积（用于共轭梯度） =====
    def fisher_vector_product(v):
        """
        计算 F @ v，其中 F 是Fisher信息矩阵。
        Fisher矩阵就是KL散度的Hessian，衡量参数空间中策略变化的"曲率"。
        """
        kl = compute_kl(policy, policy, states)
        kl_grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_kl_grads = torch.cat([g.view(-1) for g in kl_grads])
        kl_v = (flat_kl_grads * v).sum()
        kl_v_grads = torch.autograd.grad(kl_v, policy.parameters())
        flat_kl_v_grads = torch.cat([g.view(-1) for g in kl_v_grads])
        return flat_kl_v_grads + 0.1 * v  # 加个阻尼防止数值不稳定
    
    # ===== Step 4: 用共轭梯度法求解 F @ step_dir = g =====
    step_dir = conjugate_gradient(fisher_vector_product, flat_grads)
    
    # ===== Step 5: 计算最大步长并做线搜索 =====
    # 最大步长由信任域大小决定：
    # step_size = sqrt(2δ / (step_dir^T @ F @ step_dir))
    shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir))
    max_step = torch.sqrt(max_kl / (shs + 1e-8))
    full_step = max_step * step_dir
    
    # 线搜索：尝试不同的步长，找到满足KL约束且确实改善目标的那个
    old_params = torch.cat([p.data.view(-1) for p in policy.parameters()])
    for fraction in [1.0, 0.5, 0.25, 0.125]:
        new_params = old_params + fraction * full_step
        # 把新参数加载到策略网络
        idx = 0
        for p in policy.parameters():
            numel = p.numel()
            p.data.copy_(new_params[idx:idx+numel].view(p.shape))
            idx += numel
        # 检查KL约束和目标改善
        new_kl = compute_kl(policy, policy, states)
        new_loss = compute_loss()
        if new_kl <= max_kl and new_loss >= loss:
            break  # 找到了，就用这个步长
    
    # ===== Step 6: 更新Critic =====
    returns = advantages + values  # 回报 = 优势 + 旧的V值
    value_loss = ((value_fn(states) - returns) ** 2).mean()
    value_optimizer.step()  # Critic用简单的SGD就行
```

可以看到，TRPO的代码相当复杂：需要共轭梯度、Fisher矩阵向量积、线搜索……这就是为什么大家后来更喜欢用PPO。

### 3.2 PPO（Proximal Policy Optimization）
#### Pipeline
PPO的训练流程和TRPO非常相似，但在"如何保证策略不跑偏"这件事上，做了大幅简化：

1. 用当前策略 $ \pi_{\theta_{\text{old}}} $ 采集一批轨迹数据。
2. 计算GAE优势函数 $ \hat{A}_t $。
3. 对采集到的数据，做多个epoch的小批量梯度更新（这是PPO的一个优点：数据利用率高）。
4. 每次更新时，用裁剪（clip）后的目标函数来限制策略变化。
5. 同时更新Actor和Critic。
6. 回到第1步。

#### 创新在哪？
PPO的核心创新就一句话：**用裁剪（clipping）替代KL散度约束。**

TRPO需要显式地计算KL散度、求解约束优化问题，很麻烦。PPO说：不用那么复杂，我直接把重要性采样比 $ r_t(\theta) $ 裁剪到一个范围里不就行了？

PPO的目标函数叫 **Clipped Surrogate Objective**：

$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $

逐个拆解：

+ $ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $：重要性采样比（同TRPO）。如果新策略和旧策略完全一样，$ r_t = 1 $。
+ $ \epsilon $：裁剪范围的超参数，通常取 0.1 或 0.2。
+ $ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) $：把 $ r_t $ 限制在 $ [1-\epsilon, 1+\epsilon] $ 的范围内。如果 $ r_t $ 超出了这个范围，就被"截断"到边界值。
+ $ \min(\cdot, \cdot) $：取两者中较小的。

这个设计的精妙之处在于：

当 $ \hat{A}_t > 0 $（好动作）时，我们想增大 $ r_t $（提高这个动作的概率）。但 $ \text{clip} $ 把 $ r_t $ 的上限卡在了 $ 1+\epsilon $，所以即使新策略特别偏爱这个动作，目标函数也不会再增加了。$ \min $ 确保了用裁剪后的更保守的值。

当 $ \hat{A}_t < 0 $（坏动作）时，我们想减小 $ r_t $。但 $ \text{clip} $ 把 $ r_t $ 的下限卡在了 $ 1-\epsilon $，防止概率降太多。同样 $ \min $ 选更保守的值。

最终效果：**新策略不会偏离旧策略太远**，和TRPO的目标一样，但实现简单得多——就是一个普通的梯度下降，不需要共轭梯度、不需要线搜索。

#### 代码详解
```python
import torch
import torch.nn as nn
import torch.optim as optim

def ppo_update(actor, critic, trajectories, 
               clip_epsilon=0.2, gamma=0.99, lam=0.95,
               ppo_epochs=4, mini_batch_size=64,
               actor_lr=3e-4, critic_lr=1e-3):
    """
    PPO的一次完整更新。
    
    参数:
    - actor: 策略网络
    - critic: 价值网络
    - trajectories: 采集的数据
    - clip_epsilon: 裁剪范围 ε
    - ppo_epochs: 对同一批数据重复训练的次数
    - mini_batch_size: 小批量大小
    """
    states, actions, rewards, next_states, dones, old_log_probs = trajectories
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    
    # ===== Step 1: 计算GAE优势函数 =====
    with torch.no_grad():
        values = critic(states).squeeze()
        next_values = critic(next_states).squeeze()
    
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 回报 = 优势 + V值（用作Critic的训练目标）
    returns = advantages + values
    
    # 归一化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # ===== Step 2: 多轮PPO更新 =====
    dataset_size = len(states)
    
    for _ in range(ppo_epochs):
        # 随机打乱数据，分成小批量
        indices = torch.randperm(dataset_size)
        
        for start in range(0, dataset_size, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]
            
            # 取出这个小批量的数据
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # --- Actor更新 ---
            dist = actor(batch_states)
            new_log_probs = dist.log_prob(batch_actions)
            
            # 重要性采样比 r_t(θ) = exp(log π_new - log π_old)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # 未裁剪的目标：r_t * A_t
            surr1 = ratio * batch_advantages
            
            # 裁剪后的目标：clip(r_t, 1-ε, 1+ε) * A_t
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            
            # PPO目标：取min，再取均值，再取负（因为PyTorch做的是最小化）
            actor_loss = -torch.min(surr1, surr2).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_optimizer.step()
            
            # --- Critic更新 ---
            value_pred = critic(batch_states).squeeze()
            # Critic的损失就是简单的MSE
            critic_loss = ((value_pred - batch_returns) ** 2).mean()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            critic_optimizer.step()
    
    return actor_loss.item(), critic_loss.item()
```

对比TRPO的代码，PPO简洁了太多。核心就是那个 `torch.clamp` + `torch.min`，几行代码就搞定了"信任域"的效果。这就是PPO能成为工业界标配的原因——效果和TRPO差不多，但实现简单、调参容易、计算高效。

### 3.3 GRPO（Group Relative Policy Optimization）
#### Pipeline
GRPO是DeepSeek在训练DeepSeek-R1等模型时提出的算法，它的训练流程和PPO有一个关键区别——**它不需要Critic网络**。

1. 对每个prompt $ x $，用当前策略 $ \pi_\theta $ 采样 **一组**（group）回复 $ \{y_1, y_2, \ldots, y_G\} $（通常 $ G $ 取几十到上百个）。
2. 用奖励模型（或规则）对每个回复打分，得到 $ \{r_1, r_2, \ldots, r_G\} $。
3. 在**这组回复内部**计算优势函数：用组内的均值和标准差来归一化奖励。
4. 用PPO风格的裁剪目标函数更新策略。
5. 回到第1步。

#### 创新在哪？
GRPO最大的创新是：**用组内相对排序来替代Critic网络估计优势函数。**

前面说了，PPO需要一个Critic网络来估计 $ V(s) $，进而算出优势 $ A = Q - V $。但在LLM场景下，Critic网络本身就是一个巨大的模型，训练和推理的开销极其可观。

GRPO的思路是：既然优势函数本质上就是"这个动作比平均水平好多少"，那我干脆**直接采样一组回复，用它们的奖励来估计"平均水平"** 不就行了？

具体来说，GRPO的优势函数定义为：

$ \hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})} $

逐个拆解：

+ $ \hat{A}_i $：第 $ i $ 个回复的优势估计。
+ $ r_i $：第 $ i $ 个回复的奖励分数。
+ $ \text{mean}(\{r_1, \ldots, r_G\}) $：这组回复奖励的均值，充当 baseline（就是 $ V(s) $ 的估计）。
+ $ \text{std}(\{r_1, \ldots, r_G\}) $：这组回复奖励的标准差，用于归一化。

是不是简单得有点过分？就是普通的 z-score 标准化。但它的效果惊人地好。

这里有一个精妙之处：**同一个prompt下的多个回复，它们共享同一个初始状态**，所以用组内均值来近似 $ V(s) $ 在理论上是合理的。如果采样的组足够大，组内均值就越接近真实的 $ V(s) $。

GRPO的完整目标函数是：

$ L^{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( r_i(\theta) \hat{A}_i, \; \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right] $

新符号说明：

+ $ G $：组大小，即每个prompt采样的回复数量。
+ $ r_i(\theta) $：第 $ i $ 个回复中每个token的重要性采样比的乘积（或对数之和）。
+ $ \beta $：KL惩罚的系数，控制新策略不偏离参考策略太远。
+ $ \pi_{\text{ref}} $：参考策略，通常就是SFT之后、RL训练之前的那个模型。
+ $ D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) $：当前策略和参考策略之间的KL散度。

注意最后那个KL惩罚项。在PPO中，KL约束是隐式的（通过clip实现）；在GRPO中，它被显式地加到了目标函数里作为正则项。这提供了双重保障：clip防止单步更新太大，KL惩罚防止策略在多步更新后整体偏移太远。

#### 代码详解
```python
import torch
import torch.nn as nn
import torch.optim as optim

def grpo_update(policy, ref_policy, reward_model, prompts,
                group_size=16, clip_epsilon=0.2, beta=0.01,
                grpo_epochs=1, lr=1e-6):
    """
    GRPO的一次完整更新。
    
    注意：这里没有Critic网络！这是GRPO和PPO最大的区别。
    
    参数:
    - policy: 当前策略模型 π_θ
    - ref_policy: 参考策略模型 π_ref（冻结的）
    - reward_model: 奖励模型，输入prompt+response，输出分数
    - prompts: 一批训练prompt
    - group_size: 每个prompt采样的回复数量 G
    - clip_epsilon: 裁剪范围 ε
    - beta: KL惩罚系数 β
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    all_loss = 0
    for prompt in prompts:
        # ===== Step 1: 对每个prompt，采样一组回复 =====
        with torch.no_grad():
            # 采样 G 个回复
            responses = []
            old_log_probs_list = []
            for _ in range(group_size):
                response, log_probs = policy.generate_with_logprobs(prompt)
                responses.append(response)
                old_log_probs_list.append(log_probs)
        
        # ===== Step 2: 用奖励模型给每个回复打分 =====
        with torch.no_grad():
            rewards = torch.tensor([
                reward_model.score(prompt, resp) for resp in responses
            ])
        
        # ===== Step 3: 计算组内相对优势（核心创新！）=====
        # 就是简单的 z-score 归一化，不需要Critic！
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8  # 加小常数防止除零
        advantages = (rewards - reward_mean) / reward_std
        
        # ===== Step 4: PPO风格的裁剪更新 =====
        for epoch in range(grpo_epochs):
            total_loss = torch.tensor(0.0)
            
            for i in range(group_size):
                # 重新计算当前策略对第i个回复的log概率
                new_log_probs = policy.compute_log_probs(prompt, responses[i])
                old_log_probs = old_log_probs_list[i]
                
                # 重要性采样比（token级别求和后取exp）
                # r_i(θ) = exp(Σ_t log π_θ(a_t|s_t) - Σ_t log π_old(a_t|s_t))
                log_ratio = (new_log_probs - old_log_probs).sum()
                ratio = torch.exp(log_ratio)
                
                # 裁剪
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                
                # 取min
                surr1 = ratio * advantages[i]
                surr2 = clipped_ratio * advantages[i]
                policy_loss = -torch.min(surr1, surr2)
                
                # KL惩罚项
                with torch.no_grad():
                    ref_log_probs = ref_policy.compute_log_probs(prompt, responses[i])
                # KL ≈ Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))
                kl = (new_log_probs - ref_log_probs).sum()
                
                total_loss += policy_loss + beta * kl
            
            # 平均所有回复的损失
            total_loss = total_loss / group_size
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
        
        all_loss += total_loss.item()
    
    return all_loss / len(prompts)
```

看，没有Critic网络，没有GAE，没有TD误差。GRPO的核心就是"多采样几个回复，做个归一化"。在大模型训练的场景下，这节省的计算资源是极其可观的——你少了一整个Critic模型的前向、反向传播和显存占用。

### 3.4 拉表环节！谁是超大杯？
好了，三个算法都讲完了，该做个总结对比了。

| 对比维度 | TRPO | PPO | GRPO |
| :--- | :--- | :--- | :--- |
| **需要的模型** | Actor + Critic + Ref模型 | Actor + Critic + Ref模型 + 奖励模型 | Actor + Ref模型 + 奖励模型（**无Critic**） |
| **信任域实现** | 严格KL散度约束 + 共轭梯度求解 | clip裁剪近似（`torch.clamp`一行搞定） | clip裁剪 + 显式KL惩罚项（双重保障） |
| **优势函数来源** | Critic网络 + GAE | Critic网络 + GAE | 组内奖励z-score归一化（**不需要Critic**） |
| **核心优化方式** | 约束优化（共轭梯度 + 线搜索） | 无约束优化（普通SGD/Adam） | 无约束优化（普通SGD/Adam） |
| **显存占用** | 🔴 最高（Actor + Critic + 二阶导数计算） | 🟡 高（Actor + Critic 两个大模型） | 🟢 较低（省掉了整个Critic模型） |
| **计算开销** | 🔴 最重（共轭梯度、Fisher矩阵向量积、线搜索） | 🟡 中等 | 🟡 训练轻，但推理重（每个prompt采样G个回复） |
| **数据利用率** | 🔴 低（每批数据只做一次更新） | 🟢 高（同一批数据可训多个epoch） | 🟢 高（多epoch + 组内多回复天然数据丰富） |
| **实现复杂度** | 🔴 很高（共轭梯度、线搜索、Fisher信息矩阵） | 🟢 很低（核心就是clamp + min） | 🟢 很低（核心就是z-score + clamp + min） |
| **理论保证** | 🟢 最强（严格单调改进保证） | 🟡 经验上好，理论保证弱于TRPO | 🟡 经验上好，理论保证类似PPO |
| **适用场景** | 学术研究、理论验证 | 通用RL任务的工业标配（ChatGPT等） | LLM专属（DeepSeek-R1等） |
| **代表作** | Schulman et al. 2015 | OpenAI ChatGPT / InstructGPT | DeepSeek-R1 |


一句话总结：**TRPO偏学术，PPO偏商务，GRPO偏运动。**

---

## 四、一些思考
### 4.1 Reward Hacking，Ref模型与clip
如果你只告诉模型"奖励越高越好"，模型一定会找到某种方式"作弊"来获得高奖励，而不是真正变得更好。这种现象叫 **Reward Hacking（奖励黑客）**。

举几个经典例子：如果奖励模型偏好长回复，模型就会学会无意义地灌水来凑字数。如果奖励模型对某些格式（比如列了很多条目、加了很多Markdown标记）给分高，模型就会过度使用这些格式。如果奖励模型不太擅长判断某类问题的正确性，模型就会学会用自信的语气说错误的内容——因为奖励模型会被"骗"到。

这就是为什么我们需要 **参考模型（Reference Model）** 和 **clip** 机制。

参考模型 $ \pi_{\text{ref}} $ 通常是SFT训练好之后、RL训练之前的那个模型的快照，在整个RL训练过程中保持冻结。KL惩罚 $ D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) $ 的作用就是一根"橡皮筋"，把正在训练的策略拴在参考策略附近。如果策略试图为了迎合奖励模型而走向极端（比如学会了一种刷分但质量低的回复模式），KL惩罚会把它拽回来。

clip机制则是更微观层面的约束。它限制了每次更新中，每个动作的概率变化幅度。就算某个方向的梯度特别大，clip也会说"不行，你这一步只能走这么远"。

两者形成了宏观和微观的双重防线：KL惩罚防止策略在整个训练过程中漂移太远，clip防止单次更新跳得太猛。没有它们，RL训练LLM几乎一定会崩。

### 4.2 DeepSeek-R1：奖励的规则化
DeepSeek-R1的一个重要实践洞察是：**奖励不一定要用一个训练好的奖励模型，可以用规则。**

在DeepSeek-R1中，对于数学和代码这类有明确正确答案的任务，奖励直接用**规则判定**：答案对了给正奖励，错了给零或负奖励。这种做法看似粗暴，却有几个深刻的好处。

首先，**规则奖励不会被hack。** 数学题的答案要么对要么错，没有中间地带，模型无法通过花式格式或自信语气来骗分。这从根源上解决了Reward Hacking的问题。

其次，**规则奖励完全可解释。** 你清楚地知道模型为什么拿到了高分——因为它答对了。而训练出来的奖励模型是一个黑箱，你很难理解它为什么给某个回复打了高分。

第三，**规则奖励让模型涌现出推理能力。** DeepSeek-R1中一个令人惊叹的发现是，仅仅用"答案对不对"这样简单的奖励信号，模型就自发学会了 Chain-of-Thought 推理、自我验证、回溯纠错等高级行为。模型并没有被教导"你要一步步思考"，它是自己发现"一步步思考能提高答对概率"，从而自发习得了这种行为模式。

当然，规则奖励的局限也很明显：对于开放式问答、创意写作、对话体验等没有标准答案的任务，你没法定义简单的规则。这些场景仍然需要奖励模型。DeepSeek-R1的做法是对不同类型的任务使用不同的奖励来源，刚性规则和柔性模型各司其职。

### 4.3 优势函数本质上是否在塑造模型的"性格"？
这是一个有趣的角度。优势函数 $ A(s, a) $ 说的是"这个动作比平均水平好多少"。当我们用优势函数来更新策略时，我们实质上是在告诉模型："在这种情况下，你更应该选择这类表达方式，而不是那类。"

如果你仔细想想，这不就是在塑造模型的"性格"吗？

想象一下：如果奖励模型偏好谨慎、有条件限定的回答（"这取决于具体情况"），那优势函数就会让这类回答获得正优势，而武断的回答获得负优势。经过RL训练后，模型就会变得更"谨慎"——这不就是一种性格塑造吗？

如果奖励模型偏好幽默、生动的表达，优势函数就会引导模型变得更"活泼"。如果奖励模型偏好详细、全面的回答，模型就会变得更"啰嗦"（或者说"严谨"，看你怎么看）。

从这个角度来说，RLHF中的奖励模型定义了"什么是好的回答"，而优势函数则是把这个标准转化为具体的梯度信号，逐token地塑造模型的行为倾向。这些行为倾向的总和，就是我们感知到的模型"性格"。

这也解释了为什么不同的RLHF训练能产生截然不同"性格"的模型——ChatGPT、Claude、Gemini各有各的"调性"，本质上就是因为它们被不同的奖励信号（和优势函数）塑造过。

更进一步，这也引出了一个深层问题：**人的性格是否也是类似机制的产物？** 我们从环境中获得奖励和惩罚（社会认可、他人反馈），大脑中可能有某种类似"优势函数"的机制在评估每个行为的相对好坏，然后逐渐塑造出我们的行为偏好——也就是"性格"。当然，这只是一个隐喻，但它确实让人深思。

### 4.4 通用机器人为什么需要世界模型（World Model）？
最后聊一个看似跑题但其实一脉相承的话题：**为什么要做机器人的人都在说世界模型？**

让我们回到MDP的定义。MDP中有一个关键组件：$ P(s'|s, a) $——状态转移概率，也就是环境的动力学模型。在LLM的场景中，这个转移几乎是确定性的（新状态 = 旧状态 + 新token），所以我们根本不需要去学习它。但机器人不一样。

当一个机器人在物理世界中行动时，状态转移是极其复杂的。你推一下杯子，杯子会滑多远？取决于材质、摩擦力、推力大小、桌面是否有倾斜……而且你推之前**不知道**会发生什么。这就意味着，机器人要想做出好的决策，它需要能在脑中"模拟"行动的后果。

**世界模型（World Model）** 本质上就是一个学习到的 $ P(s'|s, a) $。它让机器人能在脑中"想象"不同行动的结果，然后选择结果最好的那个。这就是所谓的 **Model-Based RL（基于模型的强化学习）**，和前面讲的PPO/GRPO这种 **Model-Free RL（无模型强化学习）** 形成对比。

为什么LLM不需要世界模型但机器人需要？因为LLM面对的"环境"极其简单（确定性的token拼接），可以疯狂地靠试错（Model-Free RL）来学习。但物理世界中，试错的代价太高了——机器人不能为了学习而反复摔倒、撞墙、打碎杯子。世界模型让机器人可以在"脑中"试错，大幅减少在真实环境中交互的次数。

此外，通用机器人面对的任务是开放的、变化的。你不可能为"帮我做一杯咖啡""把脏衣服放进洗衣机""在下雨的路上骑车"每一个任务都单独训练一个策略。世界模型提供了一个统一的"物理直觉"，让机器人能在面对新任务时也能做合理的规划。

从Agentic RL的角度看，世界模型就是Agent理解其所处环境的"大脑"。LLM的Agent之所以不太需要它，是因为文本世界足够简单；而物理世界的Agent非做不可，因为物理世界太复杂了。

---

## 写在最后
从最朴素的MDP，到REINFORCE的策略梯度，到Actor-Critic的引入，到TRPO的信任域，到PPO的裁剪简化，到GRPO的去Critic化——这是一条由理论驱动到工程驱动的清晰演进路线。

每一步的进化都在回答同一个问题：**怎么更稳、更快、更省资源地优化策略？** TRPO说"用数学严格保证"，PPO说"用一个clip近似就够了"，GRPO说"在LLM场景下，连Critic都可以不要"。

而当这条技术线延伸到Agentic RL——让LLM不仅会说话，还能行动——我们就看到了AI从"文本生成器"变成"决策者"的可能性。那些关于奖励设计、优势塑造、世界模型的思考，本质上都在探索同一个命题：**如何让AI在开放的世界中，做出好的决策。**

这或许是我们这个时代最激动人心的技术故事之一。

---

> 这篇博客其实是全篇由Claude Opus 4.6执笔的，你看出来了吗？
>



