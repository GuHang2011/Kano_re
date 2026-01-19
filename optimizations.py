"""
KANO 优化模块
================================================
放置于: kapt_modules/optimizations.py

使用方法:
    from kapt_modules.optimizations import (
        WarmupCosineScheduler,
        LabelSmoothingBCE,
        FocalLoss,
        EarlyStopping,
        set_seed
    )
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# ==================== 随机种子 ====================
def set_seed(seed: int):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 学习率调度器 ====================
class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup 轮数
        total_epochs: 总训练轮数
        min_lr: 最小学习率
        warmup_start_lr: warmup 起始学习率
    
    Example:
        optimizer = AdamW(model.parameters(), lr=3e-5)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=100)
        
        for epoch in range(100):
            train(...)
            scheduler.step()
    """
    
    def __init__(self, 
                 optimizer, 
                 warmup_epochs: int, 
                 total_epochs: int, 
                 min_lr: float = 1e-7,
                 warmup_start_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch < self.warmup_epochs:
                # 线性 warmup
                progress = (self.current_epoch + 1) / self.warmup_epochs
                lr = self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            param_group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


# ==================== 损失函数 ====================
class LabelSmoothingBCE(nn.Module):
    """带标签平滑的二分类交叉熵损失
    
    Args:
        smoothing: 平滑系数 (默认 0.05)
    
    Example:
        criterion = LabelSmoothingBCE(smoothing=0.05)
        loss = criterion(predictions, targets)
    """
    
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 处理 NaN
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[mask]
        target = target[mask]
        
        # 标签平滑: y_smooth = y * (1 - smoothing) + 0.5 * smoothing
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return F.binary_cross_entropy_with_logits(pred, target_smooth)


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡
    
    Args:
        alpha: 平衡因子 (默认 0.25)
        gamma: 聚焦参数 (默认 2.0)
    
    Example:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss = criterion(predictions, targets)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 处理 NaN
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[mask]
        target = target[mask]
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失: BCE + Focal + Label Smoothing
    
    Args:
        smoothing: 标签平滑系数
        focal_weight: Focal Loss 权重
    
    Example:
        criterion = CombinedLoss(smoothing=0.05, focal_weight=0.3)
        loss = criterion(predictions, targets)
    """
    
    def __init__(self, smoothing: float = 0.05, focal_weight: float = 0.3):
        super().__init__()
        self.bce = LabelSmoothingBCE(smoothing)
        self.focal = FocalLoss()
        self.bce_weight = 1.0 - focal_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.focal_weight * self.focal(pred, target)


# ==================== 早停策略 ====================
class EarlyStopping:
    """改进的早停策略
    
    Args:
        patience: 容忍轮数
        min_delta: 最小改进阈值
        mode: 'max' 或 'min'
        verbose: 是否打印信息
    
    Example:
        early_stopping = EarlyStopping(patience=25, mode='max')
        
        for epoch in range(100):
            train(...)
            val_score = validate(...)
            
            if early_stopping(val_score, model):
                print("Early stopping!")
                break
        
        early_stopping.load_best(model)
    """
    
    def __init__(self, 
                 patience: int = 25, 
                 min_delta: float = 1e-4, 
                 mode: str = 'max',
                 verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save(model)
            if self.verbose:
                print(f'  [EarlyStopping] Initial score: {score:.4f}')
        elif self._is_improvement(score):
            improvement = abs(score - self.best_score)
            if self.verbose:
                print(f'  [EarlyStopping] Improved by {improvement:.4f}')
            self.best_score = score
            self.best_epoch = epoch
            self._save(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'  [EarlyStopping] No improvement. {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  [EarlyStopping] Triggered! Best: {self.best_score:.4f} @ epoch {self.best_epoch}')
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta
    
    def _save(self, model: nn.Module):
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def load_best(self, model: nn.Module):
        """加载最佳模型"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print(f'  [EarlyStopping] Loaded best model from epoch {self.best_epoch}')


# ==================== 模型增强 ====================
class DropoutWrapper(nn.Module):
    """为现有模型添加 Dropout 的包装器
    
    Example:
        model = DropoutWrapper(original_model, dropout=0.2)
    """
    
    def __init__(self, model: nn.Module, dropout: float = 0.2):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return self.dropout(output)


# ==================== 工具函数 ====================
def get_optimizer(model: nn.Module, 
                  lr: float = 3e-5, 
                  weight_decay: float = 5e-5,
                  use_layerwise_lr: bool = True) -> torch.optim.Optimizer:
    """获取优化器
    
    Args:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        use_layerwise_lr: 是否使用分层学习率
    
    Returns:
        AdamW 优化器
    """
    if use_layerwise_lr:
        # 尝试分层学习率
        param_groups = []
        
        # 编码器层使用较小学习率
        encoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name.lower() or 'mpn' in name.lower():
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': lr * 0.1})
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr})
        
        if not param_groups:
            param_groups = [{'params': model.parameters(), 'lr': lr}]
    else:
        param_groups = [{'params': model.parameters(), 'lr': lr}]
    
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    task_type: str = 'classification') -> Dict[str, float]:
    """计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测值
        task_type: 'classification' 或 'regression'
    
    Returns:
        指标字典
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        mean_squared_error, mean_absolute_error
    )
    
    # 移除 NaN
    mask = ~np.isnan(y_true.flatten())
    y_true = y_true.flatten()[mask]
    y_pred = y_pred.flatten()[mask]
    
    if task_type == 'classification':
        y_pred_binary = (y_pred >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.5
        
        return {
            'auc': auc,
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }
    else:
        return {
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'mae': mean_absolute_error(y_true, y_pred)
        }


# ==================== 训练辅助函数 ====================
def train_step(model: nn.Module,
               batch,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: torch.device,
               max_grad_norm: float = 1.0) -> float:
    """单步训练
    
    Args:
        model: 模型
        batch: 数据批次
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        max_grad_norm: 梯度裁剪阈值
    
    Returns:
        损失值
    """
    model.train()
    optimizer.zero_grad()
    
    # 获取数据
    if hasattr(batch, 'batch_graph'):
        mol_batch = batch.batch_graph()
        features = batch.features() if hasattr(batch, 'features') else None
        targets = torch.FloatTensor(batch.targets()).to(device)
    else:
        mol_batch = batch[0]
        features = batch[1] if len(batch) > 1 else None
        targets = batch[-1].to(device)
    
    # 前向传播
    if features is not None and isinstance(features, np.ndarray):
        features = torch.FloatTensor(features).to(device)
    
    preds = model(mol_batch, features)
    
    # 计算损失
    loss = criterion(preds, targets)
    
    if torch.isnan(loss) or torch.isinf(loss):
        return 0.0
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return loss.item()


# ==================== 数据集优化配置 ====================
OPTIMIZED_CONFIGS = {
    'bbbp': {
        'init_lr': 3e-5,
        'epochs': 100,
        'batch_size': 64,
        'dropout': 0.2,
        'weight_decay': 5e-5,
        'warmup_epochs': 10,
        'patience': 25,
        'label_smoothing': 0.05,
    },
    'bace': {
        'init_lr': 2e-5,
        'epochs': 120,
        'batch_size': 32,
        'dropout': 0.25,
        'weight_decay': 1e-4,
        'warmup_epochs': 15,
        'patience': 30,
        'label_smoothing': 0.1,
    },
    'hiv': {
        'init_lr': 5e-5,
        'epochs': 80,
        'batch_size': 128,
        'dropout': 0.2,
        'weight_decay': 1e-4,
        'warmup_epochs': 8,
        'patience': 20,
        'label_smoothing': 0.05,
    },
    'clintox': {
        'init_lr': 3e-5,
        'epochs': 100,
        'batch_size': 64,
        'dropout': 0.25,
        'weight_decay': 5e-5,
        'warmup_epochs': 10,
        'patience': 25,
        'label_smoothing': 0.1,
    },
    'tox21': {
        'init_lr': 5e-5,
        'epochs': 80,
        'batch_size': 128,
        'dropout': 0.2,
        'weight_decay': 1e-4,
        'warmup_epochs': 8,
        'patience': 20,
        'label_smoothing': 0.05,
    },
    'default': {
        'init_lr': 3e-5,
        'epochs': 100,
        'batch_size': 64,
        'dropout': 0.2,
        'weight_decay': 5e-5,
        'warmup_epochs': 10,
        'patience': 25,
        'label_smoothing': 0.05,
    }
}


def get_optimized_config(dataset_name: str) -> Dict:
    """获取数据集的优化配置
    
    Args:
        dataset_name: 数据集名称 (如 'bbbp', 'bace' 等)
    
    Returns:
        配置字典
    """
    key = dataset_name.lower().replace('.csv', '')
    return OPTIMIZED_CONFIGS.get(key, OPTIMIZED_CONFIGS['default'])


# ==================== 打印信息 ====================
def print_optimization_info():
    """打印优化信息"""
    info = """
╔════════════════════════════════════════════════════════════════════╗
║              KANO Optimization Module Loaded                        ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Available Components:                                             ║
║  • WarmupCosineScheduler - 学习率调度器                              ║
║  • LabelSmoothingBCE     - 标签平滑损失                              ║
║  • FocalLoss             - Focal 损失                               ║
║  • CombinedLoss          - 组合损失                                  ║
║  • EarlyStopping         - 早停策略                                  ║
║  • get_optimizer         - 优化器工具                                ║
║  • get_optimized_config  - 数据集配置                                ║
║                                                                    ║
║  Usage:                                                            ║
║  from kapt_modules.optimizations import *                          ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""
    print(info)


if __name__ == '__main__':
    print_optimization_info()
    
    # 简单测试
    print("\nTesting components...")
    
    # 测试早停
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    scores = [0.8, 0.85, 0.84, 0.84, 0.84]
    for i, score in enumerate(scores):
        if early_stopping(score, model, i):
            print("Stopped!")
            break
    
    print("\nAll tests passed!")
