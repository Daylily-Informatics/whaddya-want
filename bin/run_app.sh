#!/usr/bin/env bash

# Launcher for the local CLI talking to a deployed whaddya-want stack.
#
# Usage examples:
#
#   # Basic: derive session name automatically (requires AWS_PROFILE and --region or AWS_REGION)
#   ./bin/run_app.sh --stack-name marpro18 --region us-west-2
#
#   # Explicit session name
#   ./bin/run_app.sh --stack-name marpro18 --region us-west-2 --session-name jem18
#
#   # Override voice + add extra CLI flags
#   ./bin/run_app.sh \
#     --stack-name marpro18 \
#     --region us-west-2 \
#     --session-name jem18 \
#     --voice Joanna \
#     --voice-mode generative \
#     --extra-cli-args "--verbose --vv"
#
#   # Just list valid Polly voices and supported voice-modes in this region
#   ./bin/run_app.sh --query-voices --region us-west-2
#
# Flags:
#   --stack-name     (required unless --query-voices) CloudFormation stack name.
#   --session-name   (optional) CLI session id. Defaults to "<stack>-<YYYYmmdd-HHMMSS>".
#   --region         (required unless AWS_REGION is already set) AWS region for all AWS calls.
#   --voice          (optional) Polly VoiceId. Default: Matthew.
#   --voice-mode     (optional) Arbitrary mode string passed to client.cli. Default: generative.
#   --extra-cli-args (optional) Single string of extra flags for client.cli.
#   --query-voices   (optional) Print Polly voices + supported voice-modes and exit.
#
# Requirements:
#   - AWS_PROFILE must be set in the environment before running this script.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "This script must be executed directly, not sourced." >&2
  return 1
fi

require_aws_profile() {
  if [[ -z "${AWS_PROFILE:-}" ]]; then
    echo "ERROR: AWS_PROFILE is not set. Please export AWS_PROFILE before running this script." >&2
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

usage() {
  cat >&2 <<EOF
Usage:
  ./bin/run_app.sh --stack-name STACK --region REGION [--session-name SESSION] \\
                   [--voice VOICE] [--voice-mode MODE] [--extra-cli-args "FLAGS"] [--query-voices]

Examples:
  ./bin/run_app.sh --stack-name marpro18 --region us-west-2
  ./bin/run_app.sh --stack-name marpro18 --region us-west-2 --session-name jem18
  ./bin/run_app.sh --stack-name marpro18 --region us-west-2 --extra-cli-args "--verbose --vv"
  ./bin/run_app.sh --query-voices --region us-west-2
EOF
}

parse_region_and_remainder "$@"
if [[ ${#COMMON_REMAINING_ARGS[@]:-0} -gt 0 ]]; then
  set -- "${COMMON_REMAINING_ARGS[@]}"
else
  set --
fi
require_aws_profile
apply_region "$COMMON_REGION_ARG"
REGION="$AWS_REGION"

# Defaults
STACK=""
SESSION=""
VOICE="Matthew"
VOICE_MODE="generative"
EXTRA_CLI_ARGS=""
QUERY_VOICES=0

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)
      STACK="${2:-}"
      shift 2
      ;;
    --session-name)
      SESSION="${2:-}"
      shift 2
      ;;
    --voice)
      VOICE="${2:-}"
      shift 2
      ;;
    --voice-mode)
      VOICE_MODE="${2:-standard}"
      shift 2
      ;;
    --extra-cli-args)
      EXTRA_CLI_ARGS="${2:-}"
      shift 2
      ;;
    --query-voices)
      QUERY_VOICES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export AGENT_ID=marvin
export REGION
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"

# If we're only querying voices, we don't need a stack at all.
if [[ "$QUERY_VOICES" -eq 1 ]]; then
  echo "AWS_PROFILE       : $AWS_PROFILE"
  echo "Region            : $AWS_REGION"
  echo
  echo "Available Polly voices in this region:"
  aws polly describe-voices \
    --region "$AWS_REGION" \
    --query 'Voices[].{Id:Id,LanguageCode:LanguageCode,Name:Name,Engines:SupportedEngines}' \
    --output table

  echo
  echo "Supported voice modes (client-side concept, not Polly-specific):"
  echo "  generative   - LLM-driven conversational agent with TTS output"
  echo "  tts-only     - (if you implement it) pure TTS playback without LLM"
  exit 0
fi

# From here on, we need a stack.
if [[ -z "$STACK" ]]; then
  echo "ERROR: --stack-name is required (unless using --query-voices)." >&2
  usage
  exit 1
fi

# Default session if not provided: STACK-YYYYmmdd-HHMMSS
if [[ -z "$SESSION" ]]; then
  SESSION="${STACK}-$(date +%Y%m%d-%H%M%S)"
fi

export STACK
export SESSION

# Resolve AgentStateTable physical name
export AGENT_STATE_TABLE="$(
  aws cloudformation describe-stack-resources \
    --stack-name "$STACK" \
    --logical-resource-id AgentStateTable \
    --region "$REGION" \
    --query 'StackResources[0].PhysicalResourceId' \
    --output text
)"

# Resolve BrokerEndpoint from stack outputs
BROKER_FROM_OUTPUT="$(
  aws cloudformation describe-stacks \
    --stack-name "$STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='BrokerEndpoint'].OutputValue" \
    --output text 2>/dev/null || echo ""
)"

if [[ "$BROKER_FROM_OUTPUT" = "None" ]]; then
  BROKER_FROM_OUTPUT=""
fi

if [[ -n "$BROKER_FROM_OUTPUT" ]]; then
  export BROKER="$BROKER_FROM_OUTPUT"
else
  # Fallback: derive from RestApiId if BrokerEndpoint output is missing
  REST_API_ID="$(
    aws cloudformation describe-stacks \
      --stack-name "$STACK" \
      --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='RestApiId'].OutputValue" \
      --output text 2>/dev/null || echo ""
  )"
  if [[ "$REST_API_ID" = "None" || -z "$REST_API_ID" ]]; then
    echo "ERROR: Could not derive BrokerEndpoint or RestApiId from stack '$STACK'." >&2
    exit 1
  fi
  export BROKER="https://${REST_API_ID}.execute-api.${REGION}.amazonaws.com/Prod/agent"
fi

# Resolve MODEL_ID from BrokerFunction's Lambda env var (unless pre-set)
if [[ -z "${MODEL_ID:-}" ]]; then
  BROKER_FN_PHYS="$(
    aws cloudformation describe-stack-resources \
      --stack-name "$STACK" \
      --logical-resource-id BrokerFunction \
      --region "$REGION" \
      --query 'StackResources[0].PhysicalResourceId' \
      --output text
  )"

  MODEL_ID_LOOKUP="$(
    aws lambda get-function-configuration \
      --function-name "$BROKER_FN_PHYS" \
      --region "$REGION" \
      --query 'Environment.Variables.MODEL_ID' \
      --output text 2>/dev/null || echo ""
  )"

  if [[ "$MODEL_ID_LOOKUP" = "None" ]]; then
    MODEL_ID_LOOKUP=""
  fi

  if [[ -n "$MODEL_ID_LOOKUP" ]]; then
    export MODEL_ID="$MODEL_ID_LOOKUP"
  else
    export MODEL_ID="meta.llama3-1-8b-instruct-v1:0"
    echo "WARNING: Could not determine MODEL_ID from stack '$STACK'. Falling back to $MODEL_ID" >&2
  fi
fi

ENABLE_OUTBOUND_SMS=1          # to actually send SMS
ENABLE_OUTBOUND_EMAIL=1        # to actually send email
ENABLE_SYSTEM_COMMANDS=1       # to allow run_command
ACTION_EMAIL_FROM=john@dyly.bio  # verified in SES

echo "AWS_PROFILE       : $AWS_PROFILE"
echo "BROKER            : $BROKER"
echo "STACK             : $STACK"
echo "SESSION           : $SESSION"
echo "AGENT_STATE_TABLE : $AGENT_STATE_TABLE"
echo "MODEL_ID          : $MODEL_ID"
echo "REGION            : $REGION"
echo "AWS_REGION        : $AWS_REGION"
echo "AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION"
echo "VOICE             : $VOICE"
echo "VOICE_MODE        : $VOICE_MODE"
echo "EXTRA_CLI_ARGS    : $EXTRA_CLI_ARGS"
echo

# Build the python command; let the shell split EXTRA_CLI_ARGS into flags.
PYTHONPATH="$PWD:$PWD/layers/shared/" python -m client.cli \
  --session "$SESSION" \
  --broker-url "$BROKER" \
  --setup-devices \
  --voice "$VOICE" \
  --voice-mode "$VOICE_MODE" \
  --self-voice-name "$VOICE" \
  ${EXTRA_CLI_ARGS}
