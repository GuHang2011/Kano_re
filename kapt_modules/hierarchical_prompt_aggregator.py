import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPromptAggregator(nn.Module):
    """
    层次提示聚合器 (Hierarchical Prompt Aggregator)
    自适应融合任务级和结构级提示
    """

    def __init__(self, prompt_dim, task_embed_dim=128):
        """
        Args:
            prompt_dim (int): 提示向量维度
            task_embed_dim (int): 任务嵌入维度
        """
        super(HierarchicalPromptAggregator, self).__init__()

        self.prompt_dim = prompt_dim

        # 融合权重计算网络
        # W_fuse: 将任务嵌入映射到 2 维权重（任务权重 + 结构权重）
        self.W_fuse = nn.Sequential(
            nn.Linear(task_embed_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, 2),  # 输出 2 个权重
            nn.Softmax(dim=-1)  # 归一化
        )

        # 可选：门控机制增强融合
        self.gate = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.Sigmoid()
        )

    def forward(self, task_prompt, struct_prompt, task_embed):
        """
        Args:
            task_prompt (torch.Tensor): 任务提示，shape: [batch_size, prompt_dim]
            struct_prompt (torch.Tensor): 结构提示，shape: [batch_size, prompt_dim]
            task_embed (torch.Tensor): 任务嵌入，shape: [batch_size, task_embed_dim]

        Returns:
            fused_prompt (torch.Tensor): 融合后的提示，shape: [batch_size, prompt_dim]
            fusion_weights (torch.Tensor): 融合权重，shape: [batch_size, 2]
        """
        batch_size = task_prompt.shape[0]

        # Step 1: 计算自适应融合权重
        # β_task, β_struct = softmax(W_fuse * t_k)
        fusion_weights = self.W_fuse(task_embed)  # [B, 2]

        beta_task = fusion_weights[:, 0].unsqueeze(-1)  # [B, 1]
        beta_struct = fusion_weights[:, 1].unsqueeze(-1)  # [B, 1]

        # Step 2: 加权融合
        # x_prompt^(k) = β_task · p_k^task + β_struct · x_struct
        fused_prompt = beta_task * task_prompt + beta_struct * struct_prompt  # [B, D]

        # === 可选：门控增强融合 ===
        # 将两个提示拼接后通过门控网络，进一步调整融合结果
        concatenated = torch.cat([task_prompt, struct_prompt], dim=-1)  # [B, 2D]
        gate_weight = self.gate(concatenated)  # [B, D]

        # 门控调制
        fused_prompt = fused_prompt * gate_weight

        return fused_prompt, fusion_weights


# === 使用示例 ===
if __name__ == "__main__":
    # 配置
    batch_size = 4
    prompt_dim = 128
    task_embed_dim = 128

    # 初始化模块
    hpa = HierarchicalPromptAggregator(
        prompt_dim=prompt_dim,
        task_embed_dim=task_embed_dim
    )

    # 模拟输入
    task_prompt = torch.randn(batch_size, prompt_dim)
    struct_prompt = torch.randn(batch_size, prompt_dim)
    task_embed = torch.randn(batch_size, task_embed_dim)

    # 前向传播
    fused_prompt, weights = hpa(task_prompt, struct_prompt, task_embed)

    print(f"融合提示形状: {fused_prompt.shape}")  # [4, 128]
    print(f"融合权重形状: {weights.shape}")  # [4, 2]
    print(f"融合权重示例:\n{weights[:2]}")
    print(f"权重和: {weights.sum(dim=1)}")  # 应该全为 1.0
