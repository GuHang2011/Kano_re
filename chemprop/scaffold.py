import random
import logging
from typing import List, Tuple, Dict, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .data import MoleculeDataset, MoleculeDatapoint

logger = logging.getLogger(__name__)


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    生成分子的Murcko scaffold（骨架）SMILES字符串
    :param smiles: 分子的SMILES字符串
    :param include_chirality: 是否包含手性信息
    :return: 骨架SMILES字符串
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(data: MoleculeDataset, use_indices: bool = False) -> Dict[str, List[int]]:
    """
    将数据集按骨架分组，返回{骨架SMILES: 分子索引列表}的字典
    :param data: MoleculeDataset数据集
    :param use_indices: 是否使用索引（固定为True）
    :return: 骨架到分子索引的映射字典
    """
    scaffolds = {}
    for i, dp in enumerate(data):
        smile = dp.smiles
        scaffold = generate_scaffold(smile)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)
    return scaffolds


def log_scaffold_stats(data: MoleculeDataset, fold_indices: List[List[int]], logger: logging.Logger = None):
    """
    打印骨架拆分的统计信息（如每个折的骨架数量）
    :param data: MoleculeDataset数据集
    :param fold_indices: 各折的索引列表
    :param logger: 日志器
    """
    debug = logger.debug if logger is not None else print
    scaffolds = scaffold_to_smiles(data)
    scaffold_sets = [set() for _ in fold_indices]
    for i, indices in enumerate(fold_indices):
        for idx in indices:
            for scaffold, scaffold_indices in scaffolds.items():
                if idx in scaffold_indices:
                    scaffold_sets[i].add(scaffold)
                    break
    debug(f"Number of scaffolds in each fold: {[len(s) for s in scaffold_sets]}")
    debug(f"Total scaffolds: {len(scaffolds)}")


def scaffold_split(
    data: MoleculeDataset,
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    balanced: bool = False,
    seed: int = 0,
    logger: logging.Logger = None
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    按分子骨架拆分数据集（保证同一骨架的分子只出现在一个拆分集中）
    :param data: MoleculeDataset数据集
    :param sizes: 训练/验证/测试集比例
    :param balanced: 是否平衡各集的骨架数量
    :param seed: 随机种子
    :param logger: 日志器
    :return: 训练集、验证集、测试集
    """
    debug = logger.debug if logger is not None else print
    assert sum(sizes) == 1.0

    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 按骨架分组
    scaffold_dict = scaffold_to_smiles(data)
    scaffold_list = list(scaffold_dict.items())
    random.shuffle(scaffold_list)

    # 拆分逻辑
    train_size, val_size, test_size = sizes
    train_count = 0
    val_count = 0
    test_count = 0
    train_indices = []
    val_indices = []
    test_indices = []

    if balanced:
        # 平衡模式：按骨架大小排序，优先拆分大骨架
        scaffold_list.sort(key=lambda x: len(x[1]), reverse=True)
        total_size = len(data)
        train_target = train_size * total_size
        val_target = val_size * total_size
        test_target = test_size * total_size

        for scaffold, indices in scaffold_list:
            if train_count + len(indices) <= train_target:
                train_indices.extend(indices)
                train_count += len(indices)
            elif val_count + len(indices) <= val_target:
                val_indices.extend(indices)
                val_count += len(indices)
            else:
                test_indices.extend(indices)
                test_count += len(indices)
    else:
        # 普通模式：按比例拆分骨架
        total_scaffolds = len(scaffold_list)
        train_scaffolds = int(train_size * total_scaffolds)
        val_scaffolds = int(val_size * total_scaffolds)

        for i, (scaffold, indices) in enumerate(scaffold_list):
            if i < train_scaffolds:
                train_indices.extend(indices)
            elif i < train_scaffolds + val_scaffolds:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)

    # 生成拆分后的数据集
    train_data = MoleculeDataset([data[i] for i in train_indices])
    val_data = MoleculeDataset([data[i] for i in val_indices])
    test_data = MoleculeDataset([data[i] for i in test_indices])

    debug(f"Scaffold split results: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def cluster_split(
    data: MoleculeDataset,
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    balanced: bool = False,
    seed: int = 0,
    logger: logging.Logger = None
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    聚类拆分（兼容接口，暂时复用scaffold_split逻辑）
    :param data: MoleculeDataset数据集
    :param sizes: 拆分比例
    :param balanced: 是否平衡
    :param seed: 随机种子
    :param logger: 日志器
    :return: 训练/验证/测试集
    """
    debug = logger.debug if logger is not None else print
    debug("cluster_split is using scaffold_split logic (compatible mode)")
    return scaffold_split(data, sizes, balanced, seed, logger)