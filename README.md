# very-very-basic-knowledge-about-cs
## LLM推理流程： Prompt->第一个token
### 输入与预处理
1. Tokenization
将prompt文本切分为token序列
$tokens =[t_0, t_1,...,t_(n-1)]$
把连续字符串转换成可计算的离散符号
让模型/词表附庸同一套编码规则
2. Vocabulary Lookup
将token映射为token_ids
$token_ids = [id_0,id_1,...,id_(n-1)]$
把符号映射成整数索引，便于embedding table以O(1)查表
3. Padding/Batching
用于batch推理时对齐长度
生成attention_mask标记有效位置
目的1：并行计算多个样本，提升效率
目的2：防止padding位置参与注意力计算，避免分布损坏
### Embedding与位置信息
1. Token embedding 词嵌入
形状：[n, d_model](单样本)或[b,n,d_model](批处理）
目的：把离散id映射到连续向量空间，让后续注意力/MLP能做连续变换与相似性计算
2. 位置处理
方式A：绝对/可学习位置嵌入
X=X+PosEmbed(positions)
直接将位置信息注入到每个token的表里
PostEmbed可以是固定的正弦余弦编码，也可以是可学习的参数矩阵
方式B：RoPE/ALBi
不改变X， 在注意力内部操作，对Q/K施加位置相关变换（RoPE rotation position embedding旋转）或对注意力分数施加偏置(ALiBi)
目的：让注意力对相对距离更敏感，能更好地外推到更长上下文
### Transformer前向

### 得到Logits
### 解码决策
