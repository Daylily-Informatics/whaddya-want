#!/usr/bin/env bash

# Usage:
#   bash bin/tail_cloud_logs.sh STACK
#
# Example:
#   bash bin/tail_cloud_logs.sh marpro17
#
# Requirements:
#   - AWS_PROFILE set (or default profile configured)
#   - AWS_REGION / AWS_DEFAULT_REGION set (or we default to us-west-2)

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 STACK" >&2
  exit 1
fi

STACK="$1"

export AWS_PROFILE="${AWS_PROFILE:-daylily}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"

# Resolve the physical Lambda function name for the BrokerFunction
BROKER_FN_PHYS="$(
  aws cloudformation describe-stack-resources \
    --stack-name "$STACK" \
    --logical-resource-id BrokerFunction \
    --region "$AWS_REGION" \
    --query 'StackResources[0].PhysicalResourceId' \
    --output text
)"

if [ -z "$BROKER_FN_PHYS" ] || [ "$BROKER_FN_PHYS" = "None" ]; then
  echo "ERROR: Could not find BrokerFunction in stack '$STACK'." >&2
  exit 1
fi

# Resolve the current AWS account ID
ACCOUNT_ID="$(
  aws sts get-caller-identity \
    --query 'Account' \
    --output text
)"

if [ -z "$ACCOUNT_ID" ] || [ "$ACCOUNT_ID" = "None" ]; then
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
