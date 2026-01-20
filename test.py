"""
诊断脚本：检查预训练模型和当前模型的键名
"""
import os
import sys
import torch

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def check_checkpoint(checkpoint_path):
    print("=" * 60)
    print("预训练模型分析")
    print("=" * 60)

    state = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n文件类型: {type(state)}")

    if isinstance(state, dict):
        print(f"字典键: {list(state.keys())}")

        # 尝试找到 state_dict
        if 'state_dict' in state:
            state_dict = state['state_dict']
        elif 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        elif 'model' in state:
            state_dict = state['model']
        else:
            state_dict = state

        print(f"\n参数数量: {len(state_dict)}")
        print("\n所有参数键名:")
        for i, (k, v) in enumerate(state_dict.items()):
            print(f"  {i + 1}. {k} -> shape: {v.shape}")
    else:
        print(f"不是字典类型: {type(state)}")


def check_model():
    print("\n" + "=" * 60)
    print("当前模型分析")
    print("=" * 60)

    from chemprop.models import build_model
    import argparse

    # 创建最小 args
    args = argparse.Namespace(
        hidden_size=300,
        depth=3,
        dropout=0.2,
        activation='ReLU',
        ffn_hidden_size=300,
        ffn_num_layers=2,
        atom_messages=False,
        bias=False,
        undirected=False,
        features_only=False,
        features_size=None,
        num_tasks=1,
        dataset_type='classification',
        use_input_features=False,
        aggregation='mean',
        aggregation_norm=100,
        step='functional_prompt',
        cuda=False,
        features_generator=None,
    )

    model = build_model(args)
    state_dict = model.state_dict()

    print(f"\n参数数量: {len(state_dict)}")
    print("\n所有参数键名:")
    for i, (k, v) in enumerate(state_dict.items()):
        print(f"  {i + 1}. {k} -> shape: {v.shape}")


if __name__ == '__main__':
    import sys

    checkpoint_path = "./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    print(f"检查预训练模型: {checkpoint_path}")

    try:
        check_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"加载预训练模型失败: {e}")

    try:
        check_model()
    except Exception as e:
        print(f"构建模型失败: {e}")

    print("\n" + "=" * 60)
    print("请将上述输出发给我，我会帮你修复键名映射！")
    print("=" * 60)