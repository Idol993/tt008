import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class ModuleHook:
    def __init__(self, name: str):
        self.name = name
        self.output = None
        self._handle = None

    def register(self, module: nn.Module):
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach()
        else:
            self.output = output.detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class LowLevelModule(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class MidLevelModule(nn.Module):
    def __init__(self, in_channels: int = 64, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class HighLevelModule(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 10, spatial_size: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * spatial_size * spatial_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self._spatial_size = spatial_size
        self._in_channels = in_channels

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModularContinualNet(nn.Module):
    MODULE_NAMES = ["low_level", "mid_level", "high_level", "classifier"]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
    ):
        super().__init__()
        self.low_level = LowLevelModule(in_channels, 64)
        self.mid_level = MidLevelModule(64, 128)
        self.high_level = HighLevelModule(128, 256)

        spatial_after_pool = input_size // 8
        self.classifier = ClassifierHead(256, num_classes, spatial_after_pool)

        self._hooks: Dict[str, ModuleHook] = {}
        self._register_hooks()

        self._module_outputs: Dict[str, Optional[torch.Tensor]] = {}

    def _register_hooks(self):
        for name in self.MODULE_NAMES:
            module = getattr(self, name)
            hook = ModuleHook(name)
            hook.register(module)
            self._hooks[name] = hook

    def get_module_names(self) -> List[str]:
        return list(self.MODULE_NAMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.low_level(x)
        self._module_outputs["low_level"] = x
        x = self.mid_level(x)
        self._module_outputs["mid_level"] = x
        x = self.high_level(x)
        self._module_outputs["high_level"] = x
        x = self.classifier(x)
        self._module_outputs["classifier"] = x
        return x

    def forward_to_module(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        if module_name == "low_level":
            return self.low_level(x)
        elif module_name == "mid_level":
            x = self.low_level(x)
            return self.mid_level(x)
        elif module_name == "high_level":
            x = self.low_level(x)
            x = self.mid_level(x)
            return self.high_level(x)
        else:
            return self.forward()

    def get_module_output(self, module_name: str) -> Optional[torch.Tensor]:
        if module_name in self._module_outputs:
            return self._module_outputs[module_name]
        hook = self._hooks.get(module_name)
        if hook is not None:
            return hook.output
        return None

    def get_all_module_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        return dict(self._module_outputs)

    def get_module(self, module_name: str) -> Optional[nn.Module]:
        return getattr(self, module_name, None)

    def get_module_params(self, module_name: str) -> List[nn.Parameter]:
        module = self.get_module(module_name)
        if module is not None:
            return list(module.parameters())
        return []

    def adapt_classifier(self, new_num_classes: int, input_size: int = 32):
        spatial_after_pool = input_size // 8
        old_classes = self.classifier.fc2.out_features
        if new_num_classes <= old_classes:
            return
        in_channels = self.classifier._in_channels
        self.classifier = ClassifierHead(in_channels, new_num_classes, spatial_after_pool)
        device = next(self.parameters()).device
        self.classifier.to(device)
        if "classifier" in self._hooks:
            self._hooks["classifier"].remove()
        hook = ModuleHook("classifier")
        hook.register(self.classifier)
        self._hooks["classifier"] = hook
