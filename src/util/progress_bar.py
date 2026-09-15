import os
import time
import re
from collections import deque

from tqdm import tqdm
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, \
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, ProgressColumn

from lightning.pytorch.callbacks import ProgressBar

import torch


def find_best_ckpt(ckpt_dir: str, metric: str = "loss"):
    """
    Find the best checkpoint file based on the metric.
    """
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        return None, 0

    best_ckpt = None
    best_metric = float("inf")
    best_step = 0
    pattern = re.compile(r"step=(\d+)_loss=([0-9.]+)\.ckpt")

    for f in ckpts:
        match = pattern.match(f)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            if loss < best_metric:
                best_metric = loss
                best_ckpt = f
                best_step = step
    
    if best_ckpt is None:
        return None, 0

    return os.path.join(ckpt_dir, best_ckpt), best_step


class LossColumn(ProgressColumn):
    """
    Custom column to display the loss value.
    """
    def __init__(self, metric: str = "loss"):
        super().__init__()
        self.metric = metric

    def render(self, task):
        loss = task.fields.get(self.metric, None)
        if loss is not None:
            return Text(f"Loss: {loss:.4e}", style="bold blue")
        else:
            return Text("Loss: ----", style="dim")


class SpeedColumn(ProgressColumn):
    """
    Custom column to display the speed of training.
    """
    def render(self, task):
        speed = task.fields.get("speed", None)
        if speed is not None and speed > 0:
            return Text(f"{speed:.2f}it/s", style="purple")
        else:
            return Text("----it/s", style="dim")
    

class InstantSpeedTracker:
    def __init__(self, window_size=10):
        self.timestamps = deque(maxlen=window_size)

    def update(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timestamps.append(time.time())
    
    def get_speed(self):
        ts = list(self.timestamps)
        if len(ts) < 2:
            return 0.0
        durations = [t1 - t0 for t0, t1 in zip(ts[:-1], ts[1:])]
        avg = sum(durations) / len(durations)
        return 1.0 / avg if avg > 0 else 0.0


class StepRichProgressBar(ProgressBar):
    """
    Custom progress bar to show step progress
    """
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self._bar = None
        self._task = None
        self._speed_tracker = InstantSpeedTracker(window_size=10)

    def on_train_start(self, trainer, pl_module):
        self._bar = Progress(
            SpinnerColumn(),
            LossColumn(),
            BarColumn(bar_width=None),
            TextColumn("[green]Steps:"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            SpeedColumn(),
        )
        self._bar.start()
        self._task = self._bar.add_task("Training", total=self.total_steps, completed=trainer.global_step)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Try to extract loss
        loss = trainer.callback_metrics.get("loss", None)
        if loss is not None:
            self._bar.update(self._task, loss=loss.item())
        # Update speed
        self._speed_tracker.update()
        speed = self._speed_tracker.get_speed()
        self._bar.update(self._task, speed=speed, completed=trainer.global_step)
    
    def on_train_end(self, trainer, pl_module):
        self._bar.stop()
    
    def disable(self):
        pass


class StepTQDMProgressBar(ProgressBar):
    """
    Custom progress bar to show step progress
    """
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self._bar = None

    def on_train_start(self, trainer, pl_module):
        self._bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._bar.update(1)
    
    def on_train_end(self, trainer, pl_module):
        self._bar.close()
    