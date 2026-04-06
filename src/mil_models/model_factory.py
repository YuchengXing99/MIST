import os
from .model_abmil import ABMIL
from .model_transmil import TransMIL
from .model_ilra import ILRA
from .model_simple_mil import MeanMIL, MaxMIL
from .model_clam import CLAM_SB
from .model_transformer_mil import TransformerMIL
from .model_dsmil import DSMIL
from .model_configs import (
    ABMILConfig, TransMILConfig, ILRAConfig,
    MeanMILConfig, MaxMILConfig, CLAMConfig,
    TransformerMILConfig, DSMILConfig,
)

device = __import__('torch').device("cuda" if __import__('torch').cuda.is_available() else "cpu")


def create_downstream_model(args, mode='classification', config_dir='./configs'):
    """
    Create a downstream model. MIST is controlled by use_mist flag in the config JSON,
    not by model_type — use --model_config ABMIL_MIST_default to enable MIST.
    """
    config_path = os.path.join(config_dir, args.model_config, 'config.json')
    assert os.path.exists(config_path), f"Config not found: {config_path}"

    update_dict = {'in_dim': args.in_dim}
    if getattr(args, 'n_proto', None):
        update_dict['n_prototypes'] = args.n_proto
    if mode == 'classification':
        update_dict['n_classes'] = args.n_classes
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict['n_classes'] = args.n_label_bins
        elif args.loss_fn in ('cox', 'rank'):
            update_dict['n_classes'] = 1
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

    model_type = args.model_type
    if model_type == 'ABMIL':
        config = ABMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ABMIL(config=config, mode=mode)
    elif model_type == 'TransMIL':
        config = TransMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = TransMIL(config=config, mode=mode)
    elif model_type == 'ILRA':
        config = ILRAConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ILRA(config=config, mode=mode)
    elif model_type == 'MeanMIL':
        config = MeanMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = MeanMIL(config=config, mode=mode)
    elif model_type == 'MaxMIL':
        config = MaxMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = MaxMIL(config=config, mode=mode)
    elif model_type == 'CLAM':
        config = CLAMConfig.from_pretrained(config_path, update_dict=update_dict)
        model = CLAM_SB(config=config, mode=mode)
    elif model_type == 'TransformerMIL':
        config = TransformerMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = TransformerMIL(config=config, mode=mode)
    elif model_type == 'DSMIL':
        config = DSMILConfig.from_pretrained(config_path, update_dict=update_dict)
        model = DSMIL(config=config, mode=mode)
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")

    return model
