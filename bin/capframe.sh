#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "This script must be executed directly, not sourced." >&2
  return 1
fi

require_aws_profile() {
  if [[ -z "${AWS_PROFILE:-}" ]]; then
    echo "ERROR: AWS_PROFILE is not set." >&2
    exit 1
  fi
}

COMMON_REGION_ARG=""
COMMON_REMAINING_ARGS=()
parse_region_and_remainder() {
  COMMON_REGION_ARG=""
  COMMON_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --region)
        COMMON_REGION_ARG="${2:-}"
        shift 2
        ;;
      --region=*)
        COMMON_REGION_ARG="${1#*=}"
        shift
        ;;
      *)
        COMMON_REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

apply_region() {
  local provided="$1"
  if [[ -n "$provided" ]]; then
    export AWS_REGION="$provided"
  elif [[ -n "${AWS_REGION:-}" ]]; then
    export AWS_REGION
  else
    echo "ERROR: you must set --region or AWS_REGION." >&2
    exit 1
  fi
  export DEFAULT_AWS_REGION="$AWS_REGION"
  export AWS_DEFAULT_REGION="$AWS_REGION"
}

parse_region_and_remainder "$@"
if [[ ${#COMMON_REMAINING_ARGS[@]:-0} -gt 0 ]]; then
  set -- "${COMMON_REMAINING_ARGS[@]}"
else
  set --
fi
require_aws_profile
apply_region "$COMMON_REGION_ARG"

imagesnap -w 1 frame.jpg   # single shot
python face_search.py frame.jpg
