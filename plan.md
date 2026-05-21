# 任务目标：基于 DPLM-2 的抗体序列-结构共同设计微调与评估基准 (V2 - 架构升级版)

## 1. 核心架构设计与解决方案

针对 DPLM-2 预训练模型在抗体设计任务中的“非单链”和“无条件生成”两大瓶颈，本项目采用以下生物信息学与深度学习融合方案：

### 方案一：抗体双链的单链化改造 (scFv Construction)
* **原理**：DPLM-2 的输入逻辑基于单链蛋白质拓扑。在数据准备阶段，我们将抗体的重链可变区（VH）和轻链可变区（VL）通过一段人工设计的柔性连接子（Peptide Linker）串联，组合成生物学上成熟的 **scFv（单链可变片段）** 形式。
* **拼接格式**：`[VH 序列] + [Linker 序列] + [VL 序列]`
* **连接子选择**：统一采用标准的 15 氨基酸柔性连接子 **`(GGGGS)3`**（序列为 `GGGGSGGGGSGGGGS`）。
* **掩码策略**：在条件设计时，只掩码（Mask）VH 和 VL 内部的 6 个 CDR 区域。**Linker 区域的序列和 3D 坐标作为已知的固定上下文输入**，不参与 Mask 也不计算 Loss。

### 方案二：零初始化交叉注意力条件注入 (Zero-Init Cross-Attention Adapter)
为了在不破坏 DPLM-2 原生强大的结构先验的前提下注入抗原（Antigen）物理特征：
1. **条件特征提取**：使用一个完全冻结（Frozen）的 DPLM-2 基础模型作为特征编码器，输入抗原的序列与结构 Token，提取出抗原的全局稠密空间表征 $H_{antigen} \in \mathbb{R}^{L_{antigen} \times D}$。
2. **交叉注意力层（Cross-Attention Layer）插入**：
   * 保持 DPLM-2 原有网络中所有的自注意力（Self-Attention）层和前馈网络（FFN）参数**完全锁定（Freeze）**。
   * 在每个自注意力层之后，**并行或串行插入一个全新的 Cross-Attention 层**。
   * **输入映射**：当前主干网络的抗体表征作为 Query ($Q$)；被冻结的抗原表征 $H_{antigen}$ 作为 Key ($K$) 和 Value ($V$)。
3. **零初始化（Zero-Initialization）策略**：
   * 将新插入的 Cross-Attention 层中，最后的**输出投影线性层（Output Projection Layer）的权重（Weight）和偏置（Bias）全部初始化为 0**。
   * **数学保障**：在训练的第一步，条件分支的物理输出严格为 0。模型等价于原始无条件生成状态，保证从预训练到微调的完美平滑过渡，防止梯度爆炸或先验坍塌。

---

## 2. 核心模块开发指南 (Cursor 编码规范)

### 模块 A：scFv 数据预处理管道 (`src/data/scfv_pipeline.py`)
1. **数据清洗与解析**：
   - 提取 SAbDab 数据库中 **2024年8月17日之后** 发布的抗体-抗原复合物 PDB 文件。
   - 使用 **Chothia 编号系统** 精确界定 VH、VL 及其各自的 6 个 CDR 区域索引。
2. **scFv 序列与坐标级联**：
   - 按照 `VH + (GGGGS)3 + VL` 的物理顺序拼接序列。
   - 对应的 3D 坐标矩阵也同步进行拼接。由于 `(GGGGS)3` 连接子在天然晶体中不存在，其初始 3D 坐标可通过简单的线性插值结合局部刚体松弛生成，并将其在训练损失屏蔽码（Loss Mask）中置为 0（即不让 Linker 的坐标误差影响抗体设计的性能）。
3. **严格去重**：使用 MMSeq2 对 scFv 序列进行聚类，剔除与历史微调训练集 CDR-H3 相似度 $\ge 50\%$ 的测试样本。

### 模块 B：零初始化 Cross-Attention 注入 (`src/models/adapters.py`)
请基于 PyTorch 实现高效的条件注入层。核心代码伪逻辑如下：
```python
import torch
import torch.nn as nn

class ZeroInitCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 核心：零初始化零输出投影层
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_antibody, h_antigen):
        # x_antibody: [B, L_ab, D] (来自 DPLM-2 自注意力层的输出)
        # h_antigen: [B, L_ag, D] (冻结的抗原编码表征)
        attn_out, _ = self.cross_attn(query=x_antibody, key=h_antigen, value=h_antigen)
        # 在初始状态下，out_proj 输出全部为 0，对主干不产生任何扰动
        return self.out_proj(attn_out) 

```

* **前向传播织入**：修改 DPLM-2 的 Transformer Block 代码，在原有的 `x = x + self.self_attn(x)` 之后，织入 `x = x + self.zero_init_cross_attn(x, h_antigen)`。

### 模块 C：渐进式两阶段训练策略 (`src/training/train_adapter.py`)

1. **阶段 1：Adapter 适配训练（初期）**
* 固定（`requires_grad=False`）DPLM-2 的主干网络参数。
* 仅对新插入的 `ZeroInitCrossAttention` 层的参数以及抗原特征投影层设置 `requires_grad=True` 进行训练。
* **目的**：在不破坏大模型常识的前提下，让新层快速学会接收并对齐抗原的物理与几何条件。


2. **阶段 2：全量微调（后期）**
* 放开所有参数（`Full Finetuning`），使用极小的学习率（如原微调学习率的 $1/10$），对抗体可变区、结构模块与适配器进行内敛的端到端联合优化。



### 模块 D：侧链优化与 MFDesign 评估基准 (`src/evaluation/metrics.py`)

1. **结构重建与 PyRosetta 松弛**：
* 将 DPLM-2 预测出的 scFv 结构 Token 还原为 3D 骨架坐标。
* 剥离掉 `(GGGGS)3` 连接子部分，将 VH 和 VL 还原为独立的双链复合体形态。
* 调用 **PyRosetta** 的 `FastDesign` 或 `MinMover` 进行侧链打包（Side-chain packing）与全原子能量最小化弛豫（Relaxation）。


2. **四维指标计算**：
* **AAR & Loop-AAR**：对比生成的 6 个 CDR（特别是除掉锚定残基后的 CDR-H3 Loop）与天然序列的一致性。
* **RMSD & Loop-RMSD**：对齐 Framework 区域后，计算 CDR $C_\alpha$ 的坐标偏差。
* **IMP (能量优化率)**：利用 PyRosetta `InterfaceAnalyzer`（`ref2015` 评分函数）计算 $\Delta G$，统计设计抗体优于天然抗体的比例。



---

## 3. Cursor 任务分步执行指令

1. **Step 1**: 请先实现 `src/data/scfv_pipeline.py`。读取 SAbDab 原始双链抗体数据，按照 `(GGGGS)3` 连接子拼接成单链，并为 6 个 CDR 区域生成对应的 Mask 张量（不掩码 Linker 区域）。
2. **Step 2**: 编写 `src/models/adapters.py`。实现 `ZeroInitCrossAttention` 类，并将输出线性层的权重和偏置强行初始化为 0。然后编写修改现有 DPLM-2 主干网络前向传播的 Monkey-Patch 脚本。
3. **Step 3**: 编写两阶段微调的训练脚本 `src/training/train_adapter.py`，实现第一阶段冻结主干、第二阶段全量解冻的优化器逻辑。
4. **Step 4**: 编写评估后处理代码 `src/evaluation/metrics.py`，实现将生成的 scFv 还原为双链、进行 PyRosetta 弛豫优化、以及测定 AAR、RMSD 和结合能阻抗 IMP 的完整 Pipeline。

***

### 💡 针对该架构额外给你的微调建议：
1. **Linker 的 3D 坐标处理**：由于 DPLM-2 会同时处理序列和结构，拼接出来的 `(GGGGS)3` 虚构段在 3D 空间中可能会因为拉扯产生非常不合理的键长或键角。**在模型输入的结构 Mask 中，必须把这一段全填为 1（代表完全被 Mask 破坏掉的空间）**，让扩散模型在推理时纯粹靠上下文去容忍它，而不要让它错误的初始坐标污染了真实的 VH/VL 框架。
2. **测试集生成时**：通过 DPLM-2 采样的 scFv 结构在被拆回双链后，骨架接头处可能会有微小的形变，**PyRosetta 的结构弛豫（Relaxation）在此时是绝对不可省去的步骤**，否则直接测 RMSD 可能会因为接头处畸变而导致指标偏低。