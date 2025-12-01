#!/usr/bin/env bash

# Usage:
#   bash bin/tail_cloud_logs.sh STACK [--region REGION]
#
# Example:
#   bash bin/tail_cloud_logs.sh marpro17 --region us-west-2
#
# Requirements:
#   - AWS_PROFILE set
#   - Region provided via --region or AWS_REGION

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

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 STACK [--region REGION]" >&2
  exit 1
fi

STACK="$1"

# Resolve the physical Lambda function name for the BrokerFunction
BROKER_FN_PHYS="$(
  aws cloudformation describe-stack-resources \
    --stack-name "$STACK" \
    --logical-resource-id BrokerFunction \
    --region "$AWS_REGION" \
    --query 'StackResources[0].PhysicalResourceId' \
    --output text
)"

if [[ -z "$BROKER_FN_PHYS" || "$BROKER_FN_PHYS" = "None" ]]; then
  echo "ERROR: Could not find BrokerFunction in stack '$STACK'." >&2
  exit 1
fi

# Resolve the current AWS account ID
ACCOUNT_ID="$(
  aws sts get-caller-identity \
    --query 'Account' \
    --output text
)"

if [[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" = "None" ]]; then
  echo "ERROR: Could not determine AWS account ID (sts get-caller-identity failed)." >&2
  exit 1
fi

LOG_GROUP_NAME="/aws/lambda/${BROKER_FN_PHYS}"
LOG_GROUP_ARN="arn:aws:logs:${AWS_REGION}:${ACCOUNT_ID}:log-group:${LOG_GROUP_NAME}"

echo "Tailing logs for stack '$STACK':"
echo "  Function    : $BROKER_FN_PHYS"
echo "  Log group   : $LOG_GROUP_NAME"
echo "  Log group ARN: $LOG_GROUP_ARN"
echo

aws logs start-live-tail \
  --log-group-identifiers "$LOG_GROUP_ARN" \
  --region "$AWS_REGION"
