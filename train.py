import warnings

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np
import torch

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp

# 新增：导入KAPT增强模型（兼容导入失败）
try:
    from chemprop.models import KAPTEnhancedModel
except ImportError:
    KAPTEnhancedModel = None


# 新增：包装KANO模型为KAPT增强模型
def wrap_kano_with_kapt(base_model, args):
    if KAPTEnhancedModel is None:
        raise ImportError("KAPT modules not found! Please check kapt_modules/ directory.")
    return KAPTEnhancedModel(base_model, args)


def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-time independent runs with KANO/KAPT support"""
    info = logger.info if logger is not None else print

    # 初始化相关变量
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # 校验KAPT参数
    if getattr(args, 'use_kapt_prompt', False) and KAPTEnhancedModel is None:
        raise RuntimeError("use_kapt_prompt=True but KAPT modules are not available!")

    # 多轮训练
    all_scores = []
    for run_num in range(args.num_runs):
        info(f'Run {run_num + 1}/{args.num_runs}')
        args.seed = init_seed + run_num
        args.save_dir = os.path.join(save_dir, f'run_{run_num}')
        makedirs(args.save_dir)

        # 临时替换模型构建函数（兼容KAPT）
        import chemprop.models as models_module
        original_build = models_module.build_model

        def custom_build_model(args_build, encoder_name):
            base_model = original_build(args_build, encoder_name)
            if getattr(args, 'use_kapt_prompt', False):
                info(f"[Run {run_num}] Wrapping KANO model with KAPT prompt module")
                base_model = wrap_kano_with_kapt(base_model, args_build)
            return base_model

        models_module.build_model = custom_build_model

        # 执行训练（修复所有参数缺失问题）
        try:
            model_scores = run_training(args, prompt=False, logger=logger)
        finally:
            # 恢复原函数，避免污染后续运行
            models_module.build_model = original_build

        all_scores.append(model_scores)

    all_scores = np.array(all_scores)

    # 结果报告
    info(f'\n{args.num_runs}-time runs completed')
    for run_num, scores in enumerate(all_scores):
        avg_score = np.nanmean(scores)
        info(f'Seed {init_seed + run_num} ==> test {args.metric} = {avg_score:.6f}')

        if getattr(args, 'show_individual_scores', False):
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + run_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # 计算均值和标准差
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score = np.nanmean(avg_scores)
    std_score = np.nanstd(avg_scores)

    info(f'\n========== Final Results ==========')
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    # 单个任务的统计
    if getattr(args, 'show_individual_scores', False):
        for task_num, task_name in enumerate(task_names):
            task_mean = np.nanmean(all_scores[:, task_num])
            task_std = np.nanstd(all_scores[:, task_num])
            info(f'Overall test {task_name} {args.metric} = {task_mean:.6f} +/- {task_std:.6f}')

    # 保存结果到文件
    result_path = os.path.join(args.save_dir, 'final_results.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment: {args.exp_name}-{args.exp_id}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"Mean score: {mean_score:.6f}\n")
        f.write(f"Std score: {std_score:.6f}\n")
        f.write(f"All runs scores: {[f'{s:.6f}' for s in avg_scores]}\n")

    return mean_score, std_score


if __name__ == '__main__':
    # 解析并修正参数
    args = parse_train_args()
    args = modify_train_args(args)

    # 核心修复：兜底dump_path参数（防止未定义）
    if not hasattr(args, 'dump_path'):
        args.dump_path = './dumped'
        os.makedirs(args.dump_path, exist_ok=True)

    # 初始化日志和实验目录
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))

    # 执行多轮训练
    try:
        mean_auc_score, std_auc_score = run_stat(args, logger)
        print(f'\nTraining completed! Final results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise