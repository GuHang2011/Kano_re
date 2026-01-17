import warnings

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from argparse import Namespace
import logging
from logging import Logger
import os
import numpy as np
import torch
from typing import Tuple

# 项目核心模块导入
from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import (
    makedirs, save_checkpoint, load_checkpoint, load_scalers  # 新增load_scalers导入
)
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def run_kapt_training(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    适配KANO框架的KAPT训练逻辑，复用项目核心训练流程
    """
    # 仅在当前函数内定义info，作用域限定
    info = logger.info if logger is not None else print
    # 初始化基础变量
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)
    info(f"KAPT Training | Step: {args.step}")  # 移除task_id（命令行不支持该参数）

    # 多轮训练（兼容num_runs参数）
    all_scores = []
    for run_num in range(args.num_runs):
        info(f'===== Run {run_num + 1}/{args.num_runs} (Seed: {init_seed + run_num}) =====')
        args.seed = init_seed + run_num
        args.save_dir = os.path.join(save_dir, f'run_{run_num}')
        makedirs(args.save_dir)  # 复用utils的makedirs，无需额外判断

        # 执行核心训练（复用KANO的run_training，prompt=True对应KAPT逻辑）
        model_scores = run_training(args, prompt=True, logger=logger)
        all_scores.append(model_scores)

    # 计算多轮训练结果
    all_scores = np.array(all_scores)
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    # 打印结果
    info(f'\n===== KAPT Training Results =====')
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
    for task_num, task_name in enumerate(task_names):
        info(f'Overall test {task_name} {args.metric} = '
             f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    # 保存最佳模型到指定路径
    if getattr(args, 'save_model_path', None) is not None:
        makedirs(args.save_model_path, isfile=True)  # 自动创建模型保存目录（兼容文件路径）
        # 加载最后一轮的最佳模型并重保存
        last_run_dir = os.path.join(save_dir, f'run_{args.num_runs - 1}')
        best_model_path = os.path.join(last_run_dir, 'model_0', 'model.pt')  # 单模型ensemble
        if os.path.exists(best_model_path):
            # 修复：拆分load_checkpoint和load_scalers（解包错误核心修复）
            model = load_checkpoint(best_model_path, current_args=args, cuda=args.cuda, logger=logger)
            scaler, features_scaler = load_scalers(best_model_path)  # 单独加载scaler
            # 保存模型到指定路径
            save_checkpoint(args.save_model_path, model, scaler, features_scaler, args)
            info(f'Best KAPT model saved to: {args.save_model_path}')
        else:
            info(f'Warning: Best model checkpoint not found at {best_model_path}, skip save')

    return mean_score, std_score


def main():
    # 1. 解析命令行参数（复用KANO的parse_train_args，保证参数兼容）
    args = parse_train_args()

    # 2. 补充KAPT专属参数（兼容命令行传入/默认值，仅内部使用，不暴露给命令行）
    args.use_kapt = getattr(args, 'use_kapt', True)  # 是否启用KAPT（prompt=True）
    args.task_id = getattr(args, 'task_id', 0)  # 内部兼容参数，不影响命令行
    args.save_model_path = getattr(args, 'save_model_path', None)  # 模型保存路径

    # 3. 设备配置（映射--gpu到cuda参数）
    args.cuda = args.gpu is not None and torch.cuda.is_available()

    # 4. 验证/修正核心参数（复用KANO的modify_train_args）
    modify_train_args(args)

    # 5. 初始化日志（必须在参数修正后执行，兼容exp_name/exp_id）
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))

    # 设备日志输出
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU: {args.gpu}")
    else:
        logger.info("Using CPU (GPU not specified or unavailable)")

    logger.info(
        f"Initialized KAPT Training | "
        f"GPU: {args.gpu} | "
        f"Epochs: {args.epochs} | "
        f"Batch Size: {args.batch_size} | "
        f"Step: {args.step} | "
        f"Dataset Type: {args.dataset_type}"
    )

    # 6. 执行KAPT训练
    mean_score, std_score = run_kapt_training(args, logger)

    # 7. 打印最终结果
    logger.info(f'\nKAPT Training Completed | Mean {args.metric}: {mean_score:.5f} +/- {std_score:.5f}')
    print(f'\nKAPT Training Completed | Mean {args.metric}: {mean_score:.5f} +/- {std_score:.5f}')


if __name__ == '__main__':
    main()