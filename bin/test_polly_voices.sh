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
set -- "${COMMON_REMAINING_ARGS[@]}"
require_aws_profile
apply_region "$COMMON_REGION_ARG"

REGION="$AWS_REGION"
OUT_DIR="${OUT_DIR:-polly_voices}"
TEXT_TEMPLATE='Hello, I am %s, today is Thursday. Goodbye'

command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 1; }
mkdir -p "$OUT_DIR"

aws polly describe-voices --region "$REGION" --output json \
| jq -r '.Voices[] | "\(.Id)\t\((.SupportedEngines // ["standard"]) | join(","))"' \
| sort \
| while IFS="$(printf '\t')" read -r VOICE ENGINES; do
  TEXT=$(printf "$TEXT_TEMPLATE" "$VOICE")
  OUT="$OUT_DIR/$VOICE.mp3"
  echo "[$VOICE] -> $OUT"

  case ",$ENGINES," in
    *,neural,*)  # voice supports neural
      aws polly synthesize-speech \
        --region "$REGION" \
        --voice-id "$VOICE" \
        --engine neural \
        --output-format mp3 \
        --text "$TEXT" \
        "$OUT" >/dev/null
      ;;
    *)           # standard only (omit --engine for max compatibility)
      aws polly synthesize-speech \
        --region "$REGION" \
        --voice-id "$VOICE" \
        --output-format mp3 \
        --text "$TEXT" \
        "$OUT" >/dev/null
      ;;
  esac
done
