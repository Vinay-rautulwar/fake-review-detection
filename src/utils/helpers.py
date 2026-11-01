"""
Utility functions for the FakeGuard framework.

Provides foundational helper utilities including seed configuration, device selection,
YAML configuration loading, structured logging, parameter counting, and model checkpointing.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries for deterministic reproducibility.

    Args:
        seed: Integer seed value to use across random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(config_device: str = "auto") -> torch.device:
    """
    Determine and return the appropriate torch.device based on availability and config.

    Args:
        config_device: Device preference ("auto", "cuda", "cuda:0", "cpu", "mps").

    Returns:
        torch.device: Configured PyTorch device object.
    """
    if config_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(config_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                logging.warning(
                    f"CUDA requested ('{config_device}') but not available. Falling back to CPU."
                )
                device = torch.device("cpu")
        except Exception as e:
            logging.warning(
                f"Invalid device specification '{config_device}' ({e}). Falling back to CPU."
            )
            device = torch.device("cpu")

    return device


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def setup_logging(
    log_dir: Union[str, Path] = "results/logs",
    log_name: str = "fakeguard.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure structured file and console logging for the FakeGuard project.

    Args:
        log_dir: Directory path where log files should be saved.
        log_name: Name of the log file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / log_name

    logger = logging.getLogger("fakeguard")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the total number of parameters in a PyTorch module.

    Args:
        model: PyTorch model/module.
        trainable_only: If True, counts only trainable parameters (requires_grad=True).

    Returns:
        int: Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: Union[str, Path],
    val_loss: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Serialize and save a training checkpoint to disk.

    Args:
        model: The PyTorch neural network model to checkpoint.
        optimizer: The optimizer whose state should be saved (optional).
        epoch: Current training epoch.
        path: File destination path for the checkpoint (.pt or .pth).
        val_loss: Validation loss associated with this checkpoint.
        metrics: Dictionary of validation metrics (accuracy, F1, AUC, etc.).
        scheduler: Learning rate scheduler instance (optional).
        **kwargs: Additional metadata to store in the checkpoint.

    Returns:
        str: Absolute path to the saved checkpoint file.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics or {},
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logging.getLogger("fakeguard").info(f"Checkpoint saved successfully to: {save_path.resolve()}")
    return str(save_path.resolve())


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load model weights and training state from a saved checkpoint file.

    Args:
        path: Path to the checkpoint file.
        model: PyTorch model instance into which weights will be loaded.
        optimizer: Optimizer instance to restore state into (optional).
        scheduler: Scheduler instance to restore state into (optional).
        device: Device to map the loaded tensors to.

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary containing metadata.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path.resolve()}")

    if device is None:
        map_location = None
    else:
        map_location = torch.device(device) if isinstance(device, str) else device

    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=map_location)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_to_load = model.module if hasattr(model, "module") else model

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_key] = v

    model_to_load.load_state_dict(cleaned_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.getLogger("fakeguard").info(
        f"Loaded checkpoint from: {checkpoint_path.resolve()} (Epoch: {checkpoint.get('epoch', 'N/A')})"
    )

    return checkpoint
