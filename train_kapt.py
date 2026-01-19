"""
KANO Training Script - train_kapt.py (v17 - Complete Fix)
==========================================================
Based on the official KANO implementation from HICAI-ZJU/KANO

This script properly handles:
1. CMPN model building with correct architecture
2. Pretrained checkpoint loading
3. Functional prompt integration

Usage:
python train_kapt.py \
    --data_path data/bbbp.csv \
    --dataset_type classification \
    --gpu 0 \
    --step functional_prompt \
    --checkpoint_path "./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"
"""

import os
import sys
import math
import random
import logging
import argparse
from datetime import datetime
from typing import List, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ==================== Import chemprop ====================
from chemprop.data.utils import get_data, get_task_names, split_data
from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.data.scaler import StandardScaler
from chemprop.train import evaluate, evaluate_predictions
from chemprop.nn_utils import param_count, get_activation_function

# Import CMPN model
from chemprop.models.cmpn import CMPN, CMPNEncoder
from chemprop.features import get_atom_fdim, get_bond_fdim

print("[OK] Successfully imported chemprop modules")


# ==================== Prompt Generator Module ====================
class PromptGenerator(nn.Module):
    """
    Functional Prompt Generator for KANO.

    This module generates prompts based on functional group features from ElementKG.
    It integrates functional group knowledge into atom representations during fine-tuning.
    """
    def __init__(self, input_size: int = 300, output_size: int = 300, fg_size: int = 133):
        super(PromptGenerator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fg_size = fg_size

        # Functional group feature transformation
        self.fg_transform = nn.Sequential(
            nn.Linear(fg_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

        # Attention mechanism for combining atom features with FG features
        self.attention_layer = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.Tanh(),
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.Sigmoid()
        )

    def forward(self, atom_hiddens, fg_features, atom_num, fg_indices):
        """
        Forward pass of the prompt generator.

        Args:
            atom_hiddens: Hidden representations of atoms [num_atoms, hidden_size]
            fg_features: Functional group features [num_fg * num_mols, fg_size]
            atom_num: List of number of atoms per molecule
            fg_indices: Indices for functional groups per molecule

        Returns:
            Prompted atom features [num_atoms, hidden_size]
        """
        device = atom_hiddens.device
        batch_size = len(atom_num)

        # Transform functional group features
        fg_transformed = self.fg_transform(fg_features)  # [num_fg * num_mols, output_size]

        # Number of functional groups per molecule (typically 13 in KANO)
        num_fg_per_mol = fg_features.shape[0] // batch_size if batch_size > 0 else 13

        # Aggregate FG features per molecule using mean pooling
        fg_per_mol = fg_transformed.view(batch_size, num_fg_per_mol, -1).mean(dim=1)  # [batch_size, output_size]

        # Expand FG features to match atom dimensions
        # Each atom in a molecule gets the same FG representation
        fg_expanded = torch.repeat_interleave(
            fg_per_mol,
            torch.tensor(atom_num, device=device),
            dim=0
        )  # [total_atoms, output_size]

        # Handle dummy atom at position 0 (padding)
        if atom_hiddens.shape[0] > fg_expanded.shape[0]:
            padding = torch.zeros(1, self.output_size, device=device)
            fg_expanded = torch.cat([padding, fg_expanded], dim=0)

        # Ensure shapes match
        if fg_expanded.shape[0] != atom_hiddens.shape[0]:
            # Adjust if needed
            diff = atom_hiddens.shape[0] - fg_expanded.shape[0]
            if diff > 0:
                padding = torch.zeros(diff, self.output_size, device=device)
                fg_expanded = torch.cat([fg_expanded, padding], dim=0)
            else:
                fg_expanded = fg_expanded[:atom_hiddens.shape[0]]

        # Compute attention-weighted combination
        combined = torch.cat([atom_hiddens, fg_expanded], dim=1)
        gate_values = self.gate(combined)

        # Apply gating to combine original and prompted features
        prompted_hiddens = atom_hiddens + gate_values * fg_expanded

        return prompted_hiddens


def create_prompt_generator(hidden_size: int = 300):
    """Create and initialize a prompt generator."""
    return PromptGenerator(
        input_size=hidden_size,
        output_size=hidden_size,
        fg_size=133  # Standard functional group feature size in KANO
    )


# ==================== Modified CMPN Model with Prompt Support ====================
class MoleculeModel(nn.Module):
    """
    Complete Molecule Model for KANO with optional functional prompt support.

    Architecture:
    1. CMPN Encoder (Message Passing Network with GRU)
    2. Optional Prompt Generator (for functional_prompt mode)
    3. FFN Head (for classification/regression)
    """

    def __init__(self, args):
        super(MoleculeModel, self).__init__()
        self.args = args
        self.step = args.step

        # CMPN Encoder
        self.encoder = CMPN(args)

        # Create and attach prompt generator if using functional_prompt
        if self.step == 'functional_prompt':
            self.prompt_generator = create_prompt_generator(args.hidden_size)
            # Attach to encoder's W_i_atom layer
            self.encoder.encoder.W_i_atom.prompt_generator = self.prompt_generator

        # FFN dimensions
        self.hidden_size = args.hidden_size
        first_linear_dim = self.hidden_size

        # Add features dimension if available
        if hasattr(args, 'features_size') and args.features_size is not None and args.features_size > 0:
            first_linear_dim += args.features_size

        # Build FFN layers
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, args.num_tasks)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, self.hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(self.hidden_size, self.hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(self.hidden_size, args.num_tasks)])

        self.ffn = nn.Sequential(*ffn)

    def forward(self, smiles_batch: List[str], features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            smiles_batch: List of SMILES strings
            features_batch: Optional additional molecular features

        Returns:
            Predictions tensor [batch_size, num_tasks]
        """
        # Encode molecules using CMPN
        # The encoder handles prompt generation internally based on args.step
        mol_encodings = self.encoder(
            step=self.args.step,
            prompt=(self.args.step == 'functional_prompt'),
            batch=smiles_batch,
            features_batch=features_batch
        )

        # Concatenate additional features if provided
        if features_batch is not None and len(features_batch) > 0 and features_batch[0] is not None:
            features = torch.from_numpy(np.array(features_batch)).float()
            if next(self.parameters()).is_cuda:
                features = features.cuda()
            mol_encodings = torch.cat([mol_encodings, features], dim=1)

        # FFN prediction
        output = self.ffn(mol_encodings)
        return output


def build_model(args):
    """Build the complete model."""
    return MoleculeModel(args)


# ==================== Pretrained Checkpoint Loading ====================
def load_pretrained_checkpoint(model, checkpoint_path, cuda=False, logger=None):
    """
    Load pretrained CMPN encoder weights.

    The pretrained checkpoint contains encoder weights with keys like:
    - encoder.W_i_atom.weight
    - encoder.W_i_bond.weight
    - encoder.W_h_atom.weight
    - encoder.W_h_0.weight, encoder.W_h_1.weight
    - encoder.W_o.weight, encoder.W_o.bias
    - encoder.gru.bias, encoder.gru.gru.*
    - encoder.lr.weight

    These need to be mapped to our model structure:
    - encoder.encoder.W_i_atom.weight (etc.)
    """
    if logger:
        logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")

    # Load checkpoint
    if cuda:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if 'state_dict' in state:
            loaded_state_dict = state['state_dict']
        elif 'model_state_dict' in state:
            loaded_state_dict = state['model_state_dict']
        elif 'model' in state:
            loaded_state_dict = state['model']
        else:
            # Check if it's a direct state dict
            first_key = list(state.keys())[0] if state else None
            if first_key and isinstance(state[first_key], torch.Tensor):
                loaded_state_dict = state
            else:
                loaded_state_dict = state
    else:
        loaded_state_dict = state

    # Get model state dict
    model_state_dict = model.state_dict()

    if logger:
        logger.info(f"  Checkpoint parameters: {len(loaded_state_dict)}")
        logger.info(f"  Model parameters: {len(model_state_dict)}")

    # Match and load parameters
    pretrained_dict = {}
    matched_keys = []
    skipped_keys = []

    for ckpt_key, ckpt_val in loaded_state_dict.items():
        matched = False

        # Try different key mapping strategies
        possible_model_keys = [
            ckpt_key,  # Direct match
            f"encoder.{ckpt_key}",  # Add encoder prefix
            ckpt_key.replace("module.", ""),  # Remove DataParallel prefix
        ]

        for model_key in possible_model_keys:
            if model_key in model_state_dict:
                if ckpt_val.shape == model_state_dict[model_key].shape:
                    pretrained_dict[model_key] = ckpt_val
                    matched_keys.append(f"{ckpt_key} -> {model_key}")
                    matched = True
                    break
                else:
                    skipped_keys.append(f"{ckpt_key} (shape: {ckpt_val.shape} vs {model_state_dict[model_key].shape})")
                    matched = True
                    break

        if not matched:
            skipped_keys.append(ckpt_key)

    if logger:
        logger.info(f"  Matched {len(pretrained_dict)}/{len(loaded_state_dict)} parameters")
        if matched_keys:
            for mk in matched_keys[:5]:
                logger.info(f"    [OK] {mk}")
            if len(matched_keys) > 5:
                logger.info(f"    ... and {len(matched_keys) - 5} more")
        if skipped_keys:
            logger.info(f"  Skipped {len(skipped_keys)} keys")

    # Update model
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict, strict=False)

    return model


# ==================== Data Loading ====================
class ManualBatchLoader:
    """Manual batch loader to avoid PyTorch DataLoader collate issues."""

    def __init__(self, dataset: MoleculeDataset, batch_size: int, shuffle: bool = False, args=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.args = args

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[j] for j in batch_indices]
            yield ManualBatch(batch_data, self.args)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ManualBatch:
    """Batch data wrapper."""

    def __init__(self, data_list: List[MoleculeDatapoint], args=None):
        self.data_list = data_list
        self.args = args

    def smiles(self) -> List[str]:
        return [d.smiles for d in self.data_list]

    def features(self) -> Optional[List]:
        if self.data_list[0].features is not None:
            return [d.features for d in self.data_list]
        return None

    def targets(self) -> List:
        return [d.targets for d in self.data_list]


# ==================== Training Utilities ====================
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch < self.warmup_epochs:
                lr = base_lr * (self.current_epoch + 1) / self.warmup_epochs
            else:
                progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            param_group['lr'] = lr

    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


class LabelSmoothingBCE(nn.Module):
    """Binary cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target_smooth, reduction='none')


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 25, min_delta: float = 1e-4,
                 mode: str = 'max', verbose: bool = True):
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
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save(model)
            if self.verbose:
                print(f'  [EarlyStopping] Initial: {score:.4f}')
        elif self._is_better(score):
            if self.verbose:
                print(f'  [EarlyStopping] Improved: {self.best_score:.4f} -> {score:.4f}')
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
        return self.early_stop

    def _is_better(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def _save(self, model: nn.Module):
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state, strict=False)


# ==================== Logger Setup ====================
def setup_logger(exp_name: str, exp_id: str) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = f"dumped/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('KANO')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler with UTF-8 encoding
    fh = logging.FileHandler(
        f"{log_dir}/{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ==================== Training Functions ====================
def train_epoch(model, data_loader, loss_func, optimizer, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(data_loader, desc='Training', leave=False):
        optimizer.zero_grad()

        smiles_batch = batch.smiles()
        features_batch = batch.features()
        target_batch = batch.targets()

        # Create mask for valid targets
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if args.cuda:
            mask, targets = mask.cuda(), targets.cuda()

        # Forward pass
        preds = model(smiles_batch, features_batch)

        # Compute loss
        loss = loss_func(preds, targets)
        loss = (loss * mask).sum() / mask.sum()

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_model(model, data_loader, num_tasks, metric_func, dataset_type, args):
    """Evaluate model on data."""
    model.eval()
    preds = []
    targets_list = []

    with torch.no_grad():
        for batch in data_loader:
            smiles_batch = batch.smiles()
            features_batch = batch.features()
            target_batch = batch.targets()

            batch_preds = model(smiles_batch, features_batch)
            batch_preds = batch_preds.data.cpu().numpy()

            if dataset_type == 'classification':
                batch_preds = 1 / (1 + np.exp(-batch_preds))

            preds.extend(batch_preds.tolist())
            targets_list.extend(target_batch)

    results = evaluate_predictions(
        preds=preds,
        targets=targets_list,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type
    )

    return results


# ==================== Main Training Loop ====================
def run_training(args, logger):
    """Main training loop."""
    # Setup device
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    args.cuda = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')

    logger.info(f"Using device: {device}")
    logger.info(f"Training step: {args.step}")

    # Get task info
    args.task_names = get_task_names(args.data_path)
    args.num_tasks = len(args.task_names)
    logger.info(f"Number of tasks: {args.num_tasks}")

    all_test_scores = []

    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        set_seed(seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_idx + 1}/{args.num_runs}, Seed: {seed}")
        logger.info(f"{'='*60}")

        # Load data
        logger.info("Loading data...")
        data = get_data(path=args.data_path, args=args)

        if args.separate_val_path:
            val_data = get_data(path=args.separate_val_path, args=args)
        if args.separate_test_path:
            test_data = get_data(path=args.separate_test_path, args=args)

        if args.separate_val_path and args.separate_test_path:
            train_data = data
        else:
            train_data, val_data, test_data = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                seed=seed,
                args=args
            )

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Create data loaders
        train_loader = ManualBatchLoader(train_data, args.batch_size, shuffle=True, args=args)
        val_loader = ManualBatchLoader(val_data, args.batch_size, shuffle=False, args=args)
        test_loader = ManualBatchLoader(test_data, args.batch_size, shuffle=False, args=args)

        # Normalize features if needed
        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)

        args.features_size = train_data.features_size()

        # Build model
        logger.info("Building model...")
        model = build_model(args)

        # Load pretrained checkpoint
        if args.checkpoint_path is not None:
            model = load_pretrained_checkpoint(
                model,
                args.checkpoint_path,
                cuda=args.cuda,
                logger=logger
            )

        # Move to device
        if args.cuda:
            model = model.cuda()

        logger.info(f"Model parameters: {param_count(model):,}")

        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

        # Setup loss function
        if args.dataset_type == 'classification':
            loss_func = LabelSmoothingBCE(smoothing=args.label_smoothing)
        else:
            loss_func = nn.MSELoss(reduction='none')

        # Setup metric
        if args.metric == 'auc':
            metric_func = roc_auc_score
        elif args.metric == 'rmse':
            metric_func = lambda y, p: math.sqrt(mean_squared_error(y, p))
        elif args.metric == 'mae':
            metric_func = mean_absolute_error
        else:
            metric_func = roc_auc_score

        # Early stopping
        early_stopping = EarlyStopping(
            patience=args.patience,
            mode='max' if args.dataset_type == 'classification' else 'min'
        )

        logger.info("Starting training...")

        # Training loop
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, loss_func, optimizer, args)
            scheduler.step()
            current_lr = scheduler.get_lr()[0]

            val_scores = evaluate_model(model, val_loader, args.num_tasks, metric_func, args.dataset_type, args)
            val_score = np.nanmean(val_scores)

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | "
                f"Val {args.metric}: {val_score:.4f} | LR: {current_lr:.2e}"
            )

            if early_stopping(val_score, model, epoch):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model and evaluate
        early_stopping.load_best(model)

        test_scores = evaluate_model(model, test_loader, args.num_tasks, metric_func, args.dataset_type, args)
        test_score = np.nanmean(test_scores)
        all_test_scores.append(test_score)

        logger.info(f"Run {run_idx+1} - Test {args.metric}: {test_score:.4f}")

        # Save model
        if args.save_model_path:
            save_dir = os.path.dirname(args.save_model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            save_path = args.save_model_path.replace('.pt', f'_run{run_idx+1}.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    # Final results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Scores: {[f'{s:.4f}' for s in all_test_scores]}")
    logger.info(f"Mean: {np.mean(all_test_scores):.4f} +/- {np.std(all_test_scores):.4f}")
    logger.info("=" * 60)

    return all_test_scores


# ==================== Argument Parser ====================
def parse_args():
    parser = argparse.ArgumentParser(description='KANO Training')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--separate_val_path', type=str, default=None)
    parser.add_argument('--separate_test_path', type=str, default=None)
    parser.add_argument('--features_path', type=str, nargs='*', default=None)
    parser.add_argument('--features_scaling', action='store_true', default=True)
    parser.add_argument('--no_features_scaling', action='store_false', dest='features_scaling')
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--smiles_column', type=str, default=None)
    parser.add_argument('--target_columns', type=str, nargs='*', default=None)
    parser.add_argument('--ignore_columns', type=str, nargs='*', default=None)
    parser.add_argument('--use_compound_names', action='store_true', default=False)
    parser.add_argument('--folds_file', type=str, default=None)
    parser.add_argument('--val_fold_index', type=int, default=None)
    parser.add_argument('--test_fold_index', type=int, default=None)
    parser.add_argument('--crossval_index_dir', type=str, default=None)
    parser.add_argument('--crossval_index_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--save_smiles_splits', action='store_true', default=False)

    # Feature generator
    parser.add_argument('--features_generator', type=str, nargs='*', default=None)
    parser.add_argument('--no_features_generator', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ffn_num_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--bias', action='store_true', default=True)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--aggregation_norm', type=int, default=100)

    # CMPN specific
    parser.add_argument('--atom_messages', action='store_true', default=False)
    parser.add_argument('--undirected', action='store_true', default=False)
    parser.add_argument('--features_only', action='store_true', default=False)
    parser.add_argument('--use_input_features', action='store_true', default=False)

    # Training arguments
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'rmse', 'mae'])

    # Experiment arguments
    parser.add_argument('--step', type=str, default='functional_prompt',
                        choices=['pretrain', 'functional_prompt', 'finetune_add', 'finetune_concat'])
    parser.add_argument('--exp_name', type=str, default='kano')
    parser.add_argument('--exp_id', type=str, default='default')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)

    return parser.parse_args()


# ==================== Main ====================
def main():
    args = parse_args()
    logger = setup_logger(args.exp_name, args.exp_id)

    logger.info("=" * 60)
    logger.info("KANO Training Configuration")
    logger.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("=" * 60)

    run_training(args, logger)


if __name__ == "__main__":
    main()