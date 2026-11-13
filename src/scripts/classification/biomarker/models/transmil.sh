#!/bin/bash
MODEL_TUPLE='TransMIL,default'
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../_run_classification_model.sh" "$@"
