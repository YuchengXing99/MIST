from dataclasses import dataclass, field, asdict
from typing import Union
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PretrainedConfig:
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")

    @classmethod
    def from_pretrained(cls, config_path: Union[str, os.PathLike], update_dict={}):
        config_dict = json.load(open(config_path))
        for key in update_dict:
            if key in config_dict:
                config_dict[key] = update_dict[key]
        return cls(**config_dict)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if os.path.isfile(save_directory):
            raise AssertionError(f"{save_directory} should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        self.to_json_file(os.path.join(save_directory, "config.json"))
        logger.info(f"Configuration saved in {save_directory}")


@dataclass
class ABMILConfig(PretrainedConfig):
    gate: bool = True
    in_dim: int = 768
    n_classes: int = 2
    embed_dim: int = 512
    attn_dim: int = 384
    n_fc_layers: int = 1
    dropout: float = 0.25
    # MIST options (ignored when use_mist=False)
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class TransMILConfig(PretrainedConfig):
    in_dim: int = 1024
    n_classes: int = 2
    embed_dim: int = 512
    dropout: float = 0.25
    n_heads: int = 8
    num_landmarks: int = 256
    pinv_iterations: int = 6
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class ILRAConfig(PretrainedConfig):
    num_layers: int = 2
    in_dim: int = 768
    hidden_feat: int = 512
    num_heads: int = 8
    n_classes: int = 2
    topk: int = 2
    ln: bool = True
    dropout: float = 0.25
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class MeanMILConfig(PretrainedConfig):
    in_dim: int = 1536
    n_classes: int = 2
    embed_dim: int = 512
    n_fc_layers: int = 1
    dropout: float = 0.25
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class MaxMILConfig(PretrainedConfig):
    in_dim: int = 1536
    n_classes: int = 2
    embed_dim: int = 512
    n_fc_layers: int = 1
    dropout: float = 0.25
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class CLAMConfig(PretrainedConfig):
    in_dim: int = 1536
    n_classes: int = 2
    embed_dim: int = 512
    attn_dim: int = 384
    n_fc_layers: int = 1
    dropout: float = 0.25
    k_sample: int = 8
    instance_loss_weight: float = 0.3
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class TransformerMILConfig(PretrainedConfig):
    in_dim: int = 1536
    n_classes: int = 2
    embed_dim: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.25
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16


@dataclass
class DSMILConfig(PretrainedConfig):
    in_dim: int = 1536
    n_classes: int = 2
    embed_dim: int = 512
    dropout: float = 0.25
    # MIST options
    use_mist: bool = False
    n_prototypes: int = 16
    mist_rank: int = 16
