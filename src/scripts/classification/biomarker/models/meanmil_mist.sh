#!/bin/bash
MODEL_TUPLE='MeanMIL,MIST_default'
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../_run_classification_model.sh" "$@"
