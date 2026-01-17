import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPromptPool(nn.Module):
    """
    动态提示池 (Dynamic Prompt Pool)
    根据任务语义动态选择和聚合提示向量
    """

    def __init__(self, num_tasks, prompt_dim, num_prompts_per_task=10):
        """
        Args:
            num_tasks (int): 任务总数
            prompt_dim (int): 提示向量维度
            num_prompts_per_task (int): 每个任务的提示候选数量
        """
        super(DynamicPromptPool, self).__init__()

        self.num_tasks = num_tasks
        self.prompt_dim = prompt_dim
        self.num_prompts_per_task = num_prompts_per_task

        # 1. 任务嵌入层：将任务ID映射到连续向量空间
        self.task_embedding = nn.Embedding(num_tasks, prompt_dim)

        # 2. 可学习的提示候选池：每个任务有 Np 个提示向量
        # Shape: [num_tasks, num_prompts_per_task, prompt_dim]
        self.prompt_pool = nn.Parameter(
            torch.randn(num_tasks, num_prompts_per_task, prompt_dim)
        )

        # 3. 选择矩阵：用于计算任务嵌入与提示的相关性
        self.W_sel = nn.Linear(prompt_dim, prompt_dim, bias=False)

        # 初始化
        nn.init.xavier_uniform_(self.prompt_pool)
        nn.init.xavier_uniform_(self.W_sel.weight)

    def forward(self, task_id):
        """
        Args:
            task_id (torch.Tensor): 任务ID，shape: [batch_size] 或标量

        Returns:
            task_prompt (torch.Tensor): 任务自适应提示，shape: [batch_size, prompt_dim]
        """
        # 处理标量输入
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor([task_id], device=self.prompt_pool.device)

        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)

        batch_size = task_id.shape[0]

        # Step 1: 获取任务嵌入 t_k ∈ ℝ^d
        # Shape: [batch_size, prompt_dim]
        task_embed = self.task_embedding(task_id)

        # Step 2: 获取对应任务的提示候选
        # Shape: [batch_size, num_prompts_per_task, prompt_dim]
        prompts = self.prompt_pool[task_id]

        # Step 3: 计算注意力权重
        # α_{k,i} = exp(t_k^T W_sel p_k^(i)) / Σ_j exp(t_k^T W_sel p_k^(j))

        # 投影任务嵌入: [batch_size, prompt_dim]
        task_proj = self.W_sel(task_embed)

        # 计算相似度分数: [batch_size, num_prompts_per_task]
        # task_proj: [B, D] -> [B, 1, D]
        # prompts: [B, Np, D]
        # scores: [B, Np]
        scores = torch.matmul(
            task_proj.unsqueeze(1),  # [B, 1, D]
            prompts.transpose(1, 2)  # [B, D, Np]
        ).squeeze(1)  # [B, Np]

        # Softmax 归一化得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [B, Np]

        # Step 4: 加权聚合提示
        # p_k^task = Σ_{i=1}^Np α_{k,i} * p_k^(i)
        # [B, Np, 1] * [B, Np, D] -> [B, Np, D] -> [B, D]
        task_prompt = torch.sum(
            attention_weights.unsqueeze(-1) * prompts,
            dim=1
        )

        return task_prompt, attention_weights


# === 使用示例 ===
if __name__ == "__main__":
    # 配置
    num_tasks = 5
    prompt_dim = 128
    num_prompts_per_task = 10
    batch_size = 4

    # 初始化模块
    dpp = DynamicPromptPool(
        num_tasks=num_tasks,
        prompt_dim=prompt_dim,
        num_prompts_per_task=num_prompts_per_task
    )

    # 测试
    task_ids = torch.tensor([0, 1, 2, 0])  # batch 中的任务ID
    task_prompts, attn_weights = dpp(task_ids)

    print(f"任务提示形状: {task_prompts.shape}")  # [4, 128]
    print(f"注意力权重形状: {attn_weights.shape}")  # [4, 10]
    print(f"注意力权重和: {attn_weights.sum(dim=1)}")  # 应该全为 1.0
