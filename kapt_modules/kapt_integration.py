import torch
import torch.nn as nn

from kapt_modules.dynamic_prompt_pool import DynamicPromptPool
from kapt_modules.hierarchical_prompt_aggregator import HierarchicalPromptAggregator
from kapt_modules.node_level_prompt_refiner import NodeLevelPromptRefiner
from kapt_modules.structure_aware_prompt import StructureAwarePromptGenerator


class KAPTPromptModule(nn.Module):
    """
    KAPT 完整提示模块
    整合 DPP, SPG, HPA, NLPR 四个子模块
    """

    def __init__(
            self,
            num_tasks,
            node_dim,
            prompt_dim=128,
            kg_embed_dim=128,
            num_prompts_per_task=10,
            num_heads=4,
            dropout=0.1
    ):
        super(KAPTPromptModule, self).__init__()

        self.prompt_dim = prompt_dim

        # 四大核心组件
        self.dpp = DynamicPromptPool(
            num_tasks=num_tasks,
            prompt_dim=prompt_dim,
            num_prompts_per_task=num_prompts_per_task
        )

        self.spg = StructureAwarePromptGenerator(
            prompt_dim=prompt_dim,
            kg_embed_dim=kg_embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.hpa = HierarchicalPromptAggregator(
            prompt_dim=prompt_dim,
            task_embed_dim=prompt_dim  # 任务嵌入与提示维度一致
        )

        self.nlpr = NodeLevelPromptRefiner(
            node_dim=node_dim,
            prompt_dim=prompt_dim,
            kg_embed_dim=kg_embed_dim
        )

    def forward(self, task_id, node_features, fg_embeddings, fg_mask=None):
        """
        完整的提示增强流程

        Args:
            task_id (int or torch.Tensor): 任务ID
            node_features (torch.Tensor): 节点特征 [B, N, node_dim]
            fg_embeddings (torch.Tensor): 功能团KG嵌入 [B, G, kg_embed_dim]
            fg_mask (torch.Tensor, optional): 功能团掩码 [B, G]

        Returns:
            refined_features (torch.Tensor): 增强后的节点特征 [B, N, node_dim]
            prompt_info (dict): 包含各阶段的中间结果和权重
        """
        # ===== Step 1: 动态任务提示池 =====
        task_prompt, task_attn_weights = self.dpp(task_id)  # [B, prompt_dim]

        # 获取任务嵌入（用于 HPA）
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor([task_id], device=node_features.device)
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)

        task_embed = self.dpp.task_embedding(task_id)  # [B, prompt_dim]

        # ===== Step 2: 结构感知提示生成 =====
        struct_prompt, struct_attn_weights = self.spg(
            fg_embeddings, mask=fg_mask
        )  # [B, prompt_dim]

        # ===== Step 3: 层次提示聚合 =====
        fused_prompt, fusion_weights = self.hpa(
            task_prompt, struct_prompt, task_embed
        )  # [B, prompt_dim]

        # ===== Step 4: 节点级提示细化 =====
        refined_features, node_fg_weights = self.nlpr(
            node_features, fg_embeddings, fused_prompt, fg_mask
        )  # [B, N, node_dim]

        # 收集所有中间信息
        prompt_info = {
            'task_prompt': task_prompt,
            'struct_prompt': struct_prompt,
            'fused_prompt': fused_prompt,
            'task_attention': task_attn_weights,
            'struct_attention': struct_attn_weights,
            'fusion_weights': fusion_weights,
            'node_fg_weights': node_fg_weights,
            'lambda_scale': self.nlpr.lambda_scale.item()
        }

        return refined_features, prompt_info


# === 使用示例 ===
if __name__ == "__main__":
    # 配置参数
    num_tasks = 14  # MoleculeNet 14 个数据集
    batch_size = 8
    num_nodes = 25  # 平均原子数
    num_groups = 6  # 平均功能团数
    node_dim = 300  # GNN 节点特征维度
    prompt_dim = 128
    kg_embed_dim = 128

    # 初始化完整模块
    kapt_prompt = KAPTPromptModule(
        num_tasks=num_tasks,
        node_dim=node_dim,
        prompt_dim=prompt_dim,
        kg_embed_dim=kg_embed_dim,
        num_prompts_per_task=10,
        num_heads=4,
        dropout=0.1
    )

    # 模拟输入数据
    task_id = torch.tensor([3, 3, 5, 5, 7, 7, 2, 2])  # batch 中的任务ID
    node_features = torch.randn(batch_size, num_nodes, node_dim)
    fg_embeddings = torch.randn(batch_size, num_groups, kg_embed_dim)

    # 创建功能团掩码（某些分子功能团数量不足）
    fg_mask = torch.zeros(batch_size, num_groups, dtype=torch.bool)
    fg_mask[0, 5:] = True  # 第一个分子只有 5 个功能团
    fg_mask[3, 4:] = True  # 第四个分子只有 4 个功能团

    # 前向传播
    refined_features, info = kapt_prompt(
        task_id, node_features, fg_embeddings, fg_mask
    )

    print("=" * 60)
    print("KAPT 提示模块输出结果")
    print("=" * 60)
    print(f"增强后节点特征: {refined_features.shape}")  # [8, 25, 300]
    print(f"\n任务提示: {info['task_prompt'].shape}")  # [8, 128]
    print(f"结构提示: {info['struct_prompt'].shape}")  # [8, 128]
    print(f"融合提示: {info['fused_prompt'].shape}")  # [8, 128]
    print(f"\n融合权重 (任务 vs 结构):\n{info['fusion_weights'][:3]}")
    print(f"\nλ 缩放系数: {info['lambda_scale']:.4f}")
    print("=" * 60)
