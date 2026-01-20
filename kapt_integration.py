# KANO-main/kapt_integration.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.models import MoleculeModel
from chemprop.data.data import MoleculeDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
from rdkit.Chem import Descriptors
import math
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import ortho_group

# ========== 1. 正确导入kapt_modules下的核心子模块 ==========
from kapt_modules.dynamic_prompt_pool import DynamicPromptPool
from kapt_modules.hierarchical_prompt_aggregator import HierarchicalPromptAggregator
from kapt_modules.node_level_prompt_refiner import NodeLevelPromptRefiner
from kapt_modules.structure_aware_prompt import StructureAwarePromptGenerator

# ========== 2. 保留AUC优化的Focal Loss（性能核心：解决类别不平衡） ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ========== 3. 整合KAPT核心模块（复用kapt_modules子模块） ==========
class KAPTPromptModule(nn.Module):
    """整合kapt_modules子模块 + AUC优化"""
    def __init__(
            self,
            num_tasks: int,
            node_dim: int,
            prompt_dim: int = 512,
            kg_embed_dim: int = 128,
            num_prompts_per_task: int = 40,
            num_heads: int = 16,
            dropout: float = 0.02,
            num_layers: int = 4,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.kg_embed_dim = kg_embed_dim
        self.num_prompts_per_task = num_prompts_per_task

        # ===== 直接调用kapt_modules下的4个子模块 =====
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
            task_embed_dim=prompt_dim
        )
        self.nlpr = NodeLevelPromptRefiner(
            node_dim=node_dim,
            prompt_dim=prompt_dim,
            kg_embed_dim=kg_embed_dim
        )

        # 保留AUC优化的投影/融合层（提升分类区分度）
        self.gate_fusion = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.Sigmoid()
        )
        self.output_projection = nn.Sequential(
            nn.Linear(prompt_dim, node_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, node_dim),
            nn.LayerNorm(node_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(prompt_dim)

    def forward(
            self,
            task_id: int,
            node_features: torch.Tensor,
            fg_embeddings: torch.Tensor,
            fg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        B, N, _ = node_features.shape

        # Step 1: 调用DPP（动态提示池）
        task_prompt, task_attn_weights = self.dpp(task_id)
        task_prompts_expanded = task_prompt.unsqueeze(0).expand(B, -1, -1)

        # Step 2: 调用SPG（结构感知提示生成）
        struct_prompt, struct_attn_weights = self.spg(fg_embeddings, mask=fg_mask)
        struct_prompt_expanded = struct_prompt.unsqueeze(0).expand(B, -1, -1)

        # Step 3: 调用HPA（层次提示聚合）
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor([task_id], device=node_features.device)
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)
        task_embed = self.dpp.task_embedding(task_id)
        fused_prompt, fusion_weights = self.hpa(task_prompts_expanded, struct_prompt_expanded, task_embed)

        # Step 4: 调用NLPR（节点级提示细化）
        refined_features, node_fg_weights = self.nlpr(
            node_features, fg_embeddings, fused_prompt, fg_mask
        )

        # 保留AUC优化的门控融合（提升特征区分度）
        node_features_proj = self.nlpr.node_projection(node_features)
        concat_feat = torch.cat([node_features_proj, refined_features], dim=-1)
        gate_weight = self.gate_fusion(concat_feat)
        temperature = 0.3  # 放大分类特征差异
        gate_weight = torch.sigmoid(gate_weight / temperature)
        fused_feat = gate_weight * refined_features + (1 - gate_weight) * node_features_proj
        enhanced_node_features = self.output_projection(self.dropout(fused_feat))

        # 收集中间信息（便于调试/监控）
        prompt_info = {
            'task_prompt': task_prompt,
            'struct_prompt': struct_prompt,
            'fused_prompt': fused_prompt,
            'task_attention': task_attn_weights,
            'struct_attention': struct_attn_weights,
            'fusion_weights': fusion_weights,
            'node_fg_weights': node_fg_weights,
            'gate_weight': gate_weight
        }

        return enhanced_node_features, prompt_info

# ========== 4. 整合KAPT增强模型（性能核心：参数冻结+梯度检查点） ==========
class KAPTEnhancedModel(nn.Module):
    def __init__(
            self,
            kano_model: MoleculeModel,
            num_tasks: int,
            node_dim: int,
            prompt_dim: int = 512,
            kg_embed_dim: int = 128,
            kg_embed_path: str = "initial/elementkgontology.embeddings.txt",
            use_prompt: bool = True,
            use_gradient_checkpointing: bool = True,
            pos_weight: float = 2.0,
            **prompt_kwargs
    ):
        super().__init__()

        self.kano_model = kano_model
        self.use_prompt = use_prompt
        self.node_dim = node_dim
        self.pos_weight = pos_weight  # 正负样本权重

        # 性能优化1：冻结主干模型（仅微调输出层+KAPT）
        for name, param in self.kano_model.named_parameters():
            if 'ffn' not in name and 'output' not in name:
                param.requires_grad = False

        # 加载功能团嵌入（归一化，提升稳定性）
        self.kg_embedding_dict = self._load_elementkg_embeddings(kg_embed_path)
        self.default_fg_embed = F.normalize(torch.randn(kg_embed_dim), p=2, dim=0)

        # 初始化KAPT提示模块（调用kapt_modules）
        self.prompt_module = None
        if self.use_prompt:
            self.prompt_module = KAPTPromptModule(
                num_tasks=num_tasks,
                node_dim=node_dim,
                prompt_dim=prompt_dim,
                kg_embed_dim=kg_embed_dim,** prompt_kwargs
            )
            # 性能优化2：梯度检查点（节省显存，提升大批次训练能力）
            if use_gradient_checkpointing and self.prompt_module is not None:
                self.prompt_module.spg.attention_layer = torch.utils.checkpoint.checkpoint_sequential(
                    [self.prompt_module.spg.attention_layer], 1, use_reentrant=False
                )
                self.prompt_module.attention_layers = torch.utils.checkpoint.checkpoint_sequential(
                    list(self.prompt_module.attention_layers), 1, use_reentrant=False
                )

        # 性能优化3：分类任务专用输出头（AUC优化）
        self.task_output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.LayerNorm(node_dim),
                nn.Linear(node_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)  # 解决类别不平衡

    def _load_elementkg_embeddings(self, kg_embed_path: str) -> Dict[str, torch.Tensor]:
        """加载功能团嵌入（归一化，提升稳定性）"""
        kg_embeddings = {}
        try:
            with open(kg_embed_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) < self.kg_embed_dim + 1:
                        continue
                    fg_name = parts[0].lower()
                    embed_vec = torch.tensor([float(x) for x in parts[1:1 + self.kg_embed_dim]], dtype=torch.float32)
                    embed_vec = F.normalize(embed_vec, p=2, dim=0)
                    kg_embeddings[fg_name] = embed_vec
            print(f"成功加载ElementKG嵌入：共{len(kg_embeddings)}个功能团")
        except FileNotFoundError:
            raise ValueError(f"ElementKG嵌入文件未找到：{kg_embed_path}")
        return kg_embeddings

    def forward(self, batch, task_id: int = 0) -> Tuple[torch.Tensor, Dict]:
        """适配chemprop训练流程的前向传播"""
        # 1. 提取基础模型输入
        node_features = batch.node_features
        fg_embeddings = batch.fg_embeddings  # 需确保数据加载时包含功能团嵌入
        fg_mask = batch.fg_mask if hasattr(batch, 'fg_mask') else None

        # 2. 调用KAPT提示模块增强节点特征
        if self.use_prompt and self.prompt_module is not None:
            enhanced_node_features, prompt_info = self.prompt_module(
                task_id=task_id,
                node_features=node_features,
                fg_embeddings=fg_embeddings,
                fg_mask=fg_mask
            )
        else:
            enhanced_node_features = node_features
            prompt_info = {}

        # 3. 基础模型前向传播
        batch.node_features = enhanced_node_features
        base_output = self.kano_model(batch)

        # 4. 分类任务输出（AUC优化）
        task_output = self.task_output_heads[task_id](base_output)
        return task_output, prompt_info

    def calculate_loss(self, outputs, targets) -> torch.Tensor:
        """性能优化4：使用Focal Loss提升AUC"""
        targets = targets.float()
        # 标签平滑（降低过拟合）
        targets = (1 - self.label_smoothing) * targets + self.label_smoothing * 0.5
        return self.focal_loss(outputs.squeeze(), targets)