#!/bin/bash
MODEL_TUPLE='CLAM,default'
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_run_survival_model.sh" "$@"
