import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeLevelPromptRefiner(nn.Module):
    """
    节点级提示细化器 (Node-Level Prompt Refiner)
    将结构感知信息分配到每个原子节点，增强空间敏感性
    """

    def __init__(self, node_dim, prompt_dim, kg_embed_dim=128):
        """
        Args:
            node_dim (int): 节点特征维度
            prompt_dim (int): 提示向量维度
            kg_embed_dim (int): 功能团知识图谱嵌入维度
        """
        super(NodeLevelPromptRefiner, self).__init__()

        self.node_dim = node_dim
        self.prompt_dim = prompt_dim

        # 1. 节点投影矩阵：用于计算节点与功能团的相似度
        self.W_node = nn.Linear(node_dim, kg_embed_dim, bias=False)

        # 2. 功能团嵌入投影（从 KG 嵌入到节点空间）
        self.kg_to_node = nn.Linear(kg_embed_dim, node_dim)

        # 3. 全局提示投影（从提示空间到节点空间）
        self.prompt_to_node = nn.Linear(prompt_dim, node_dim)

        # 4. 可学习的缩放系数 λ
        self.lambda_scale = nn.Parameter(torch.tensor(0.5))

        # 初始化
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.kg_to_node.weight)
        nn.init.xavier_uniform_(self.prompt_to_node.weight)

    def forward(self, node_features, fg_embeddings, global_prompt, fg_mask=None):
        """
        Args:
            node_features (torch.Tensor):
                节点特征，shape: [batch_size, num_nodes, node_dim]
            fg_embeddings (torch.Tensor):
                功能团 KG 嵌入，shape: [batch_size, num_groups, kg_embed_dim]
            global_prompt (torch.Tensor):
                全局融合提示，shape: [batch_size, prompt_dim]
            fg_mask (torch.Tensor, optional):
                功能团掩码，shape: [batch_size, num_groups]

        Returns:
            refined_node_features (torch.Tensor):
                增强后的节点特征，shape: [batch_size, num_nodes, node_dim]
            node_fg_weights (torch.Tensor):
                节点-功能团相似度权重，shape: [batch_size, num_nodes, num_groups]
        """
        batch_size, num_nodes, _ = node_features.shape
        _, num_groups, _ = fg_embeddings.shape

        # === Step 1: 计算节点与功能团的相似度分数 ===
        # γ_{v,j} = exp(x_v^T W_node x_{g_j}) / Σ_l exp(x_v^T W_node x_{g_l})

        # 投影节点特征: [B, N, node_dim] -> [B, N, kg_embed_dim]
        node_proj = self.W_node(node_features)

        # 计算相似度分数: [B, N, kg_dim] × [B, kg_dim, G] = [B, N, G]
        similarity_scores = torch.matmul(
            node_proj,  # [B, N, kg_dim]
            fg_embeddings.transpose(1, 2)  # [B, kg_dim, G]
        )  # [B, N, G]

        # 应用掩码（将无效功能团的分数设为 -inf）
        if fg_mask is not None:
            # fg_mask: [B, G], True 表示该功能团无效
            mask_expanded = fg_mask.unsqueeze(1).expand(-1, num_nodes, -1)  # [B, N, G]
            similarity_scores = similarity_scores.masked_fill(mask_expanded, float('-inf'))

        # Softmax 归一化得到分布权重
        node_fg_weights = F.softmax(similarity_scores, dim=-1)  # [B, N, G]

        # === Step 2: 将功能团信息注入到节点 ===
        # x_v^prompt = x_v + Σ_{j=1}^m γ_{v,j} * x_{g_j}

        # 将功能团嵌入投影到节点空间: [B, G, kg_dim] -> [B, G, node_dim]
        fg_node_space = self.kg_to_node(fg_embeddings)  # [B, G, node_dim]

        # 加权求和: [B, N, G] × [B, G, node_dim] = [B, N, node_dim]
        fg_contribution = torch.matmul(
            node_fg_weights,  # [B, N, G]
            fg_node_space  # [B, G, node_dim]
        )  # [B, N, node_dim]

        # 注入到节点特征
        node_features_with_fg = node_features + fg_contribution

        # === Step 3: 加入全局融合提示 ===
        # x_v^final = x_v^prompt + λ · x_prompt^(k)

        # 投影全局提示: [B, prompt_dim] -> [B, node_dim]
        global_prompt_proj = self.prompt_to_node(global_prompt)  # [B, node_dim]

        # 扩展到所有节点: [B, node_dim] -> [B, N, node_dim]
        global_prompt_expanded = global_prompt_proj.unsqueeze(1).expand(-1, num_nodes, -1)

        # 加权加入（使用可学习的 λ）
        refined_node_features = (
                node_features_with_fg +
                self.lambda_scale * global_prompt_expanded
        )

        return refined_node_features, node_fg_weights


# === 使用示例 ===
if __name__ == "__main__":
    # 配置
    batch_size = 4
    num_nodes = 20  # 每个分子的原子数
    num_groups = 5  # 功能团数量
    node_dim = 128
    prompt_dim = 128
    kg_embed_dim = 128

    # 初始化模块
    nlpr = NodeLevelPromptRefiner(
        node_dim=node_dim,
        prompt_dim=prompt_dim,
        kg_embed_dim=kg_embed_dim
    )

    # 模拟输入
    node_features = torch.randn(batch_size, num_nodes, node_dim)
    fg_embeddings = torch.randn(batch_size, num_groups, kg_embed_dim)
    global_prompt = torch.randn(batch_size, prompt_dim)

    # 创建掩码
    fg_mask = torch.zeros(batch_size, num_groups, dtype=torch.bool)
    fg_mask[0, 4] = True  # 第一个分子只有 4 个功能团

    # 前向传播
    refined_features, weights = nlpr(
        node_features, fg_embeddings, global_prompt, fg_mask
    )

    print(f"增强节点特征形状: {refined_features.shape}")  # [4, 20, 128]
    print(f"节点-功能团权重形状: {weights.shape}")  # [4, 20, 5]
    print(f"λ 缩放系数: {nlpr.lambda_scale.item():.4f}")
