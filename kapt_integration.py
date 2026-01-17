import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.models import MoleculeModel
# 修复：替换不存在的 MoleculeDataLoader，改用 PyTorch 原生 DataLoader
from chemprop.data.data import MoleculeDataset
from torch.utils.data import DataLoader  # 核心修改：导入原生 DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np


class KAPTPromptModule(nn.Module):
    """KAPT 提示模块：结构感知的动态提示生成器"""

    def __init__(
            self,
            num_tasks: int,
            node_dim: int,
            prompt_dim: int = 128,
            kg_embed_dim: int = 128,
            num_prompts_per_task: int = 10,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.kg_embed_dim = kg_embed_dim
        self.num_prompts_per_task = num_prompts_per_task

        # 任务特定提示池
        self.task_prompt_pool = nn.Parameter(
            torch.randn(num_tasks, num_prompts_per_task, prompt_dim)
        )

        # 功能团-提示注意力融合层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=prompt_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 节点特征投影层（对齐节点维度与提示维度）
        self.node_projection = nn.Linear(node_dim, prompt_dim)
        # 输出投影层（融合后投影回节点维度）
        self.output_projection = nn.Linear(prompt_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(prompt_dim)

    def forward(
            self,
            task_id: int,
            node_features: torch.Tensor,
            fg_embeddings: torch.Tensor,
            fg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        生成动态提示并增强节点特征
        Args:
            task_id: 当前任务ID
            node_features: 原始节点特征 [B, N, node_dim]
            fg_embeddings: 功能团知识图谱嵌入 [B, G, kg_embed_dim]
            fg_mask: 功能团掩码 [B, G]（标记有效功能团）
        Returns:
            enhanced_node_features: 增强后的节点特征 [B, N, node_dim]
            prompt_info: 提示模块中间信息（用于调试）
        """
        B, N, _ = node_features.shape

        # 1. 获取当前任务的提示候选集 [num_prompts_per_task, prompt_dim]
        task_prompts = self.task_prompt_pool[task_id]  # [P, D]
        P = task_prompts.shape[0]

        # 2. 功能团嵌入投影到提示维度 [B, G, D]
        fg_proj = F.linear(fg_embeddings, torch.randn(self.kg_embed_dim, self.prompt_dim, device=fg_embeddings.device))

        # 3. 交叉注意力：功能团引导的提示选择与融合
        # 调整维度：task_prompts -> [1, P, D]（batch维度扩展）
        task_prompts_expanded = task_prompts.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        attn_output, attn_weights = self.cross_attention(
            query=task_prompts_expanded,
            key=fg_proj,
            value=fg_proj,
            key_padding_mask=fg_mask
        )

        # 4. 提示精炼：层归一化 + 残差连接
        refined_prompts = self.layer_norm(attn_output + task_prompts_expanded)
        refined_prompts = self.dropout(refined_prompts)  # [B, P, D]

        # 5. 节点特征与提示融合
        node_features_proj = self.node_projection(node_features)  # [B, N, D]
        # 提示池与节点特征的注意力融合
        node_prompt_attn = torch.matmul(node_features_proj, refined_prompts.transpose(1, 2))  # [B, N, P]
        node_prompt_attn = F.softmax(node_prompt_attn, dim=-1)  # [B, N, P]

        # 动态提示生成：节点特定提示 [B, N, D]
        node_specific_prompts = torch.matmul(node_prompt_attn, refined_prompts)

        # 6. 增强节点特征
        enhanced_node_features = self.output_projection(
            self.dropout(node_features_proj + node_specific_prompts)
        )  # [B, N, node_dim]

        prompt_info = {
            'task_prompts': task_prompts,
            'attn_weights': attn_weights,
            'node_prompt_attn': node_prompt_attn
        }

        return enhanced_node_features, prompt_info


class KAPTEnhancedModel(nn.Module):
    """集成 KAPT 提示模块的增强型分子属性预测模型"""

    def __init__(
            self,
            kano_model: MoleculeModel,
            num_tasks: int,
            node_dim: int,
            prompt_dim: int = 128,
            kg_embed_dim: int = 128,
            use_prompt: bool = True,
            **prompt_kwargs
    ):
        super().__init__()

        self.kano_model = kano_model
        self.use_prompt = use_prompt
        self.node_dim = node_dim
        self.kg_embed_dim = kg_embed_dim

        # 初始化 KAPT 提示模块
        if self.use_prompt:
            self.prompt_module = KAPTPromptModule(
                num_tasks=num_tasks,
                node_dim=node_dim,
                prompt_dim=prompt_dim,
                kg_embed_dim=kg_embed_dim, **prompt_kwargs
            )
        else:
            self.prompt_module = None

    def forward(
            self,
            batch,
            task_id: int,
            fg_embeddings: Optional[torch.Tensor] = None,
            fg_mask: Optional[torch.Tensor] = None,
            return_prompt_info: bool = False
    ):
        """
        前向传播：先增强节点特征，再通过 KANO 模型预测
        Args:
            batch: DataLoader 的 batch 数据（原 MoleculeDataLoader 替换为原生 DataLoader）
            task_id: 当前任务ID
            fg_embeddings: 功能团知识图谱嵌入 [B, G, kg_embed_dim]
            fg_mask: 功能团掩码 [B, G]
            return_prompt_info: 是否返回提示信息
        Returns:
            predictions: 预测结果
            prompt_info (optional): 提示模块中间信息
        """
        # 1. 获取 KANO 的原始节点特征（GNN 编码前）
        node_features = self.kano_model.encoder.get_node_features(batch)  # [B, N, node_dim]

        # 2. KAPT 提示增强
        prompt_info = None
        if self.use_prompt and self.prompt_module is not None:
            if fg_embeddings is None:
                fg_embeddings, fg_mask = self._extract_functional_groups(batch)

            node_features, prompt_info = self.prompt_module(
                task_id=task_id,
                node_features=node_features,
                fg_embeddings=fg_embeddings,
                fg_mask=fg_mask
            )

        # 3. 替换 KANO 的节点特征，执行后续预测
        batch.node_features = node_features
        predictions = self.kano_model(batch)

        if return_prompt_info:
            return predictions, prompt_info
        return predictions

    def _extract_functional_groups(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 batch 中提取功能团嵌入和掩码（对接 KANO 的 ElementKG）
        Args:
            batch: DataLoader 输出的 batch 数据
        Returns:
            fg_embeddings: [B, G, kg_embed_dim] 功能团嵌入
            fg_mask: [B, G] 掩码（True 表示无效功能团）
        """
        batch_size = len(batch)
        max_num_groups = 6  # KANO 中默认最大功能团数量

        # 从 ElementKG 加载预训练功能团嵌入（占位符，需替换为真实逻辑）
        fg_embeddings = torch.randn(
            batch_size, max_num_groups, self.kg_embed_dim,
            device=batch.x.device
        )

        # 生成功能团掩码（模拟部分分子功能团数量不足）
        fg_mask = torch.rand(batch_size, max_num_groups, device=batch.x.device) > 0.8

        return fg_embeddings, fg_mask


# 核心修改：函数参数中的 MoleculeDataLoader 替换为 DataLoader
def train_with_kapt(
        model: KAPTEnhancedModel,
        data_loader: DataLoader,  # 替换 MoleculeDataLoader -> DataLoader
        optimizer: torch.optim.Optimizer,
        task_id: int,
        device: torch.device,
        loss_func: nn.Module
) -> float:
    """使用 KAPT 提示模块的训练函数"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 前向传播（自动提取功能团嵌入）
        predictions = model(batch=batch, task_id=task_id)

        # 计算损失（适配 KANO 原有损失函数）
        loss = loss_func(predictions, batch.targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# 核心修改：函数参数中的 MoleculeDataLoader 替换为 DataLoader
def evaluate_with_kapt(
        model: KAPTEnhancedModel,
        data_loader: DataLoader,  # 替换 MoleculeDataLoader -> DataLoader
        task_id: int,
        device: torch.device,
        metric_func
) -> float:
    """使用 KAPT 提示模块的评估函数"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predictions = model(batch=batch, task_id=task_id)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch.targets.cpu().numpy())

    # 合并结果并计算指标
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return metric_func(all_targets, all_preds)