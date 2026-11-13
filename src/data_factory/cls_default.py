# Classification tasks registered for main_classification.py.
#
# Each entry in `label_dicts` maps the raw value in the split CSV's `target_col`
# to an integer class index. main_classification uses len(set(values)) to set
# args.n_classes, so keep the mapping consistent for each task.
#
# --- MAMMOTH biomarker suite (arXiv:2603.22198, "Mixture of Mini Experts") -----
# 13 tasks across 4 datasets: TCGA-Lung (4), TCGA-BRCA (4), TCGA-GBMLGG (2),
# BCNB (3). All binary (AUROC) except gbmlgg_histomol which is 5-class.
# Split CSVs are expected under src/splits/classification/biomarker/<task>_k=<fold>/
# and feature extraction follows the repo default (UNI, 20x/256, dim=1536).

_BINARY_0_1 = {0: 0, 1: 1}
_BINARY_STATUS = {'negative': 0, 'positive': 1, 'Negative': 0, 'Positive': 1}
_BINARY_MUT = {'WT': 0, 'MUT': 1, 'wt': 0, 'mut': 1, 0: 0, 1: 1}

_BIOMARKER_LABEL_DICTS = {
    # TCGA-Lung mutation prediction
    'tcga_lung_tp53':       _BINARY_MUT,
    'tcga_lung_kras':       _BINARY_MUT,
    'tcga_lung_stk11':      _BINARY_MUT,
    'tcga_lung_egfr':       _BINARY_MUT,
    # TCGA-BRCA status / mutation prediction
    'tcga_brca_her2':       _BINARY_STATUS,
    'tcga_brca_er':         _BINARY_STATUS,
    'tcga_brca_pr':         _BINARY_STATUS,
    'tcga_brca_pik3ca':     _BINARY_MUT,
    # TCGA-GBMLGG molecular tasks
    'tcga_gbmlgg_idh1':     _BINARY_MUT,
    'tcga_gbmlgg_histomol': {
        'astrocytoma_idhmut':          0,
        'astrocytoma_idhwt':            1,
        'oligodendroglioma_idhmut_1p19qcodel': 2,
        'glioblastoma_idhmut':          3,
        'glioblastoma_idhwt':           4,
    },
    # BCNB biopsy receptor status
    'bcnb_er':              _BINARY_STATUS,
    'bcnb_pr':              _BINARY_STATUS,
    'bcnb_her2':            _BINARY_STATUS,
}

tasks = list(_BIOMARKER_LABEL_DICTS.keys())
label_dicts = dict(_BIOMARKER_LABEL_DICTS)
