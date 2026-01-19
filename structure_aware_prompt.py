import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Fragments


class StructureAwarePromptGenerator(nn.Module):
    """
    结构感知提示生成器 (Structure-Aware Prompt Generator)
    基于分子功能团生成结构相关的提示
    """

    def __init__(self, prompt_dim, kg_embed_dim=128, num_heads=4, dropout=0.1):
        """
        Args:
            prompt_dim (int): 提示向量维度
            kg_embed_dim (int): 知识图谱嵌入维度
            num_heads (int): 多头注意力头数
            dropout (float): Dropout 概率
        """
        super(StructureAwarePromptGenerator, self).__init__()

        self.prompt_dim = prompt_dim
        self.kg_embed_dim = kg_embed_dim

        # 1. 功能团嵌入投影层（从 KG 嵌入到提示空间）
        self.kg_projection = nn.Linear(kg_embed_dim, prompt_dim)

        # 2. 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=prompt_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prompt_dim * 4, prompt_dim)
        )

        # 4. Layer Normalization
        self.norm1 = nn.LayerNorm(prompt_dim)
        self.norm2 = nn.LayerNorm(prompt_dim)

        self.dropout = nn.Dropout(dropout)

    def detect_functional_groups(self, smiles_list):
        """
        检测分子中的功能团

        Args:
            smiles_list (list): SMILES 字符串列表

        Returns:
            functional_groups (list): 每个分子的功能团列表
        """
        all_functional_groups = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                all_functional_groups.append([])
                continue

            fg_list = []

            # 使用 RDKit Fragments 模块检测常见功能团
            # 返回功能团名称
            if Fragments.fr_Al_OH(mol) > 0:
                fg_list.append('alcohol')
            if Fragments.fr_ketone(mol) > 0:
                fg_list.append('ketone')
            if Fragments.fr_aldehyde(mol) > 0:
                fg_list.append('aldehyde')
            if Fragments.fr_ester(mol) > 0:
                fg_list.append('ester')
            if Fragments.fr_amide(mol) > 0:
                fg_list.append('amide')
            if Fragments.fr_aniline(mol) > 0:
                fg_list.append('aniline')
            if Fragments.fr_nitro(mol) > 0:
                fg_list.append('nitro')
            if Fragments.fr_benzene(mol) > 0:
                fg_list.append('benzene')
            if Fragments.fr_C_O(mol) > 0:
                fg_list.append('ether')
            if Fragments.fr_halogen(mol) > 0:
                fg_list.append('halogen')

            # 如果没有检测到功能团，添加默认值
            if len(fg_list) == 0:
                fg_list.append('unknown')

            all_functional_groups.append(fg_list)

        return all_functional_groups

    def forward(self, functional_group_embeddings, mask=None):
        """
        Args:
            functional_group_embeddings (torch.Tensor):
                功能团的知识图谱嵌入，shape: [batch_size, num_groups, kg_embed_dim]
            mask (torch.Tensor, optional):
                掩码矩阵，shape: [batch_size, num_groups]

        Returns:
            struct_prompt (torch.Tensor): 结构感知提示，shape: [batch_size, prompt_dim]
        """
        batch_size, num_groups, _ = functional_group_embeddings.shape

        # Step 1: 将 KG 嵌入投影到提示空间
        # [B, N, kg_dim] -> [B, N, prompt_dim]
        fg_prompts = self.kg_projection(functional_group_embeddings)

        # Step 2: 多头自注意力融合功能团信息
        # x_struct = MultiHeadAttn([x_{g_j}]_{j=1}^m)
        attn_output, attn_weights = self.multihead_attn(
            query=fg_prompts,
            key=fg_prompts,
            value=fg_prompts,
            key_padding_mask=mask  # [B, N], True 表示该位置被忽略
        )

        # 残差连接 + Layer Norm
        fg_prompts = self.norm1(fg_prompts + self.dropout(attn_output))

        # Step 3: 前馈网络
        ffn_output = self.ffn(fg_prompts)
        fg_prompts = self.norm2(fg_prompts + self.dropout(ffn_output))

        # Step 4: 聚合为单一结构提示（平均池化或加权求和）
        if mask is not None:
            # 使用掩码进行加权平均
            mask_expanded = (~mask).unsqueeze(-1).float()  # [B, N, 1]
            struct_prompt = (fg_prompts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # 简单平均
            struct_prompt = fg_prompts.mean(dim=1)  # [B, prompt_dim]

        return struct_prompt, attn_weights


# === 使用示例 ===
if __name__ == "__main__":
    # 配置
    batch_size = 4
    num_groups = 5  # 最多检测到的功能团数量
    kg_embed_dim = 128
    prompt_dim = 128

    # 初始化模块
    spg = StructureAwarePromptGenerator(
        prompt_dim=prompt_dim,
        kg_embed_dim=kg_embed_dim,
        num_heads=4
    )

    # 模拟功能团的 KG 嵌入（实际应从预训练的 ElementKG 中获取）
    fg_embeddings = torch.randn(batch_size, num_groups, kg_embed_dim)

    # 创建掩码（某些分子功能团数量少于 num_groups）
    mask = torch.zeros(batch_size, num_groups, dtype=torch.bool)
    mask[0, 4] = True  # 第一个分子只有 4 个功能团
    mask[1, 3:] = True  # 第二个分子只有 3 个功能团

    # 前向传播
    struct_prompt, attn = spg(fg_embeddings, mask)

    print(f"结构提示形状: {struct_prompt.shape}")  # [4, 128]
    print(f"注意力权重形状: {attn.shape}")  # [4, 5, 5]
