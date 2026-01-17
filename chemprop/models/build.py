import logging
from argparse import Namespace

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# 基础CMPNN编码器（KANO核心编码器）
class CMPNNEncoder(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.hidden_size = getattr(args, 'hidden_size', 300)
        self.ffn_hidden_size = getattr(args, 'ffn_hidden_size', 300)
        self.layers = getattr(args, 'layers', 3)
        self.dropout = getattr(args, 'dropout', 0.1)

        # 占位：实际CMPNN逻辑可根据KANO源码补充
        self.embedding = nn.Embedding(100, self.hidden_size)  # 原子类型嵌入
        self.fc = nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, step: str, augment: bool, smiles: list, features: torch.Tensor = None):
        """
        前向传播
        :param step: 训练阶段（pretrain/finetune）
        :param augment: 是否数据增强
        :param smiles: SMILES列表
        :param features: 分子特征
        :return: 分子嵌入向量
        """
        # 占位逻辑：返回随机嵌入（实际需替换为CMPNN逻辑）
        batch_size = len(smiles)
        return torch.randn(batch_size, self.hidden_size).to(self.embedding.weight.device)


# KANO模型（带Prompt）
class KANOModel(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = CMPNNEncoder(args)
        self.prompt = None  # Prompt模块占位
        hidden_size = args.get('hidden_size', 300) if isinstance(args, dict) else getattr(args, 'hidden_size', 300)
        num_tasks = args['num_tasks'] if isinstance(args, dict) else args.num_tasks
        self.task_head = nn.Linear(hidden_size, num_tasks)

    def forward(self, smiles: list, features: torch.Tensor = None):
        embedding = self.encoder('finetune', False, smiles, features)
        if self.prompt is not None:
            embedding = self.prompt(embedding)
        return self.task_head(embedding)


def build_model(args: Namespace, encoder_name: str = 'CMPNN') -> nn.Module:
    """
    构建模型
    :param args: 参数命名空间
    :param encoder_name: 编码器名称
    :return: 模型实例
    """
    if encoder_name.lower() == 'cmpnn':
        model = KANOModel(args)
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    # 移至GPU
    if getattr(args, 'cuda', True) and torch.cuda.is_available():
        model = model.cuda()

    logger.debug(f"Built {encoder_name} model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def build_pretrain_model(args: Namespace, encoder_name: str = 'CMPNN') -> nn.Module:
    """
    构建预训练模型
    :param args: 参数命名空间
    :param encoder_name: 编码器名称
    :return: 预训练模型实例
    """
    return build_model(args, encoder_name)


def add_functional_prompt(model: nn.Module, args: Namespace):
    """
    为模型添加Functional Prompt（KANO核心）
    :param model: 基础模型
    :param args: 参数命名空间
    """
    # 占位：实际Prompt逻辑可根据KANO源码补充
    model.prompt = nn.Linear(model.encoder.hidden_size, model.encoder.hidden_size)
    logger.debug("Added functional prompt to KANO model")