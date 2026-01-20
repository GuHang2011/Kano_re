"""
诊断脚本：检查 chemprop 的 build_model 是否能构建正确的 CMPN 模型

运行方式:
python diagnose_model_build.py --data_path data/bbbp.csv --dataset_type classification
"""

import os
import sys
import argparse

# 添加项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def check_chemprop_model():
    """检查 chemprop 的模型结构"""

    print("=" * 60)
    print("检查 chemprop 模型构建")
    print("=" * 60)

    try:
        from chemprop.models import build_model, MoleculeModel
        from chemprop.nn_utils import param_count
        print("✓ 成功导入 chemprop.models")

        # 检查 MoleculeModel 的结构
        import inspect
        print(f"\nMoleculeModel 定义位置: {inspect.getfile(MoleculeModel)}")

        # 查看 build_model 函数
        print(f"build_model 定义位置: {inspect.getfile(build_model)}")

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return

    # 创建测试 args
    class Args:
        pass

    args = Args()

    # 基本参数
    args.dataset_type = 'classification'
    args.num_tasks = 1
    args.task_names = ['task1']
    args.features_size = 0
    args.features_path = None
    args.features_generator = None
    args.no_features_generator = False

    # 模型参数
    args.hidden_size = 300
    args.depth = 3
    args.dropout = 0.1
    args.activation = 'ReLU'
    args.ffn_num_layers = 2
    args.ffn_hidden_size = None  # 默认等于 hidden_size
    args.bias = True
    args.aggregation = 'mean'
    args.aggregation_norm = 100

    # 可能需要的额外参数
    args.atom_messages = False
    args.undirected = False
    args.class_balance = False
    args.checkpoint_frzn = None
    args.frzn_ffn_layers = 0
    args.freeze_first_only = False

    # CMPN/KANO 特定参数 (这些可能是关键!)
    args.encoder_type = 'CMPN'  # 或 'DMPNN', 'GCN' 等

    print("\n" + "=" * 60)
    print("尝试构建模型...")
    print("=" * 60)

    try:
        model = build_model(args)
        print(f"\n✓ 模型构建成功!")
        print(f"  参数数量: {param_count(model):,}")
        print(f"  模型类型: {type(model)}")

        print("\n模型结构:")
        print(model)

        print("\n模型参数键名:")
        state_dict = model.state_dict()
        for i, (key, tensor) in enumerate(state_dict.items()):
            print(f"  {i + 1}. {key} -> shape: {tuple(tensor.shape)}")

    except Exception as e:
        print(f"\n✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 60)
        print("可能的解决方案:")
        print("=" * 60)
        print("""
1. 检查你的 chemprop 版本是否包含 CMPN 编码器
   - KANO 项目可能修改了 chemprop 来添加 CMPN 支持
   - 检查 chemprop/models/ 目录下是否有 cmpn.py 或类似文件

2. 检查是否需要特殊的命令行参数来启用 CMPN:
   --encoder_type CMPN
   或
   --model_type CMPN

3. 查看 KANO 项目的原始训练脚本，看看它是如何构建模型的
""")


def check_encoder_types():
    """检查 chemprop 支持的编码器类型"""
    print("\n" + "=" * 60)
    print("检查可用的编码器类型")
    print("=" * 60)

    try:
        # 检查 chemprop 的模型模块
        import chemprop.models as models_module
        print(f"\nchemprop.models 包含:")
        for name in dir(models_module):
            if not name.startswith('_'):
                obj = getattr(models_module, name)
                if isinstance(obj, type):
                    print(f"  - {name} (class)")
                elif callable(obj):
                    print(f"  - {name} (function)")
    except Exception as e:
        print(f"检查失败: {e}")

    # 检查是否有 CMPN 相关文件
    print("\n检查 CMPN 相关模块:")
    try:
        from chemprop.models.cmpn import CMPN
        print("  ✓ 找到 chemprop.models.cmpn.CMPN")
    except ImportError:
        print("  ✗ 未找到 chemprop.models.cmpn")

    try:
        from chemprop.models.mpn import CMPN
        print("  ✓ 找到 chemprop.models.mpn.CMPN")
    except ImportError:
        print("  ✗ 未找到 chemprop.models.mpn.CMPN")


def list_chemprop_files():
    """列出 chemprop 的文件结构"""
    print("\n" + "=" * 60)
    print("chemprop 模块结构")
    print("=" * 60)

    try:
        import chemprop
        chemprop_path = os.path.dirname(chemprop.__file__)
        print(f"chemprop 路径: {chemprop_path}")

        for root, dirs, files in os.walk(chemprop_path):
            level = root.replace(chemprop_path, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = '  ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    print(f"{sub_indent}{file}")
    except Exception as e:
        print(f"检查失败: {e}")


if __name__ == "__main__":
    check_chemprop_model()
    check_encoder_types()
    list_chemprop_files()

    print("\n" + "=" * 60)
    print("下一步")
    print("=" * 60)
    print("""
请将此脚本的输出发给我，我需要看到：
1. chemprop 的模型目录结构
2. 是否存在 CMPN 编码器
3. build_model 构建的模型结构

这样我才能帮你修复模型加载问题。
""")