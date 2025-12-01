#!/usr/bin/env bash

# Usage:
#   source bin/run_app.sh STACK SESSION [MODEL_ID]
#
# STACK    = CloudFormation stack name (e.g., marpro9)
# SESSION  = Arbitrary session id (used by the broker to group events)
# MODEL_ID = (optional) Bedrock model id; if omitted, this script will try to
#            read MODEL_ID from the BrokerFunction's Lambda configuration.
#
# The broker URL is derived from the CloudFormation stack outputs:
#   - First tries the BrokerEndpoint output.
#   - If missing, falls back to RestApiId and constructs the URL.

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 STACK SESSION [MODEL_ID]" >&2
  return 1 2>/dev/null || exit 1
fi

export STACK="$1"
export SESSION="$2"
MODEL_ID_OVERRIDE="${3:-}"

export AGENT_ID=marvin
export REGION="${AWS_REGION:-us-west-2}"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"

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

if [ "$BROKER_FROM_OUTPUT" = "None" ]; then
  BROKER_FROM_OUTPUT=""
fi

if [ -n "$BROKER_FROM_OUTPUT" ]; then
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
  if [ "$REST_API_ID" = "None" ] || [ -z "$REST_API_ID" ]; then
    echo "ERROR: Could not derive BrokerEndpoint or RestApiId from stack '$STACK'." >&2
    return 1 2>/dev/null || exit 1
  fi
  export BROKER="https://${REST_API_ID}.execute-api.${REGION}.amazonaws.com/Prod/agent"
fi

# Resolve MODEL_ID:
#  - if a 3rd arg was provided, use that
#  - otherwise, ask Lambda for BrokerFunction's environment MODEL_ID
if [ -n "$MODEL_ID_OVERRIDE" ]; then
  export MODEL_ID="$MODEL_ID_OVERRIDE"
else
  # Get the physical Lambda name for the BrokerFunction
  BROKER_FN_PHYS="$(
    aws cloudformation describe-stack-resources \
      --stack-name "$STACK" \
      --logical-resource-id BrokerFunction \
      --region "$REGION" \
      --query 'StackResources[0].PhysicalResourceId' \
      --output text
  )"

  # Ask Lambda for its MODEL_ID env var
  MODEL_ID_LOOKUP="$(
    aws lambda get-function-configuration \
      --function-name "$BROKER_FN_PHYS" \
      --region "$REGION" \
      --query 'Environment.Variables.MODEL_ID' \
      --output text 2>/dev/null || echo ""
  )"

  # Some AWS CLI versions return "None" if unset; normalize that away
  if [ "$MODEL_ID_LOOKUP" = "None" ]; then
    MODEL_ID_LOOKUP=""
  fi

  if [ -n "$MODEL_ID_LOOKUP" ]; then
    export MODEL_ID="$MODEL_ID_LOOKUP"
  else
    # Last-resort default if the stack doesn't have MODEL_ID set for some reason.
    export MODEL_ID="meta.llama3-1-8b-instruct-v1:0"
    echo "WARNING: Could not determine MODEL_ID from stack '$STACK'. Falling back to $MODEL_ID" >&2
  fi
fi

ENABLE_OUTBOUND_SMS=1          # to actually send SMS
ENABLE_OUTBOUND_EMAIL=1        # to actually send email
ENABLE_SYSTEM_COMMANDS=1       # to allow run_command
ACTION_EMAIL_FROM=john@dyly.bio  # verified in SES

echo "BROKER         : $BROKER"
echo "STACK          : $STACK"
echo "SESSION        : $SESSION"
echo "AGENT_STATE_TABLE : $AGENT_STATE_TABLE"
echo "MODEL_ID       : $MODEL_ID"
echo "REGION         : $REGION"
echo "AWS_REGION     : $AWS_REGION"
echo "AWS_DEFAULT_REGION : $AWS_DEFAULT_REGION"

PYTHONPATH="$PWD:$PWD/layers/shared/python" python -m client.cli --session "$SESSION" \
  --broker-url "$BROKER" \
  --setup-devices \
  --voice Matthew \
  --voice-mode generative \
  --verbose -vv \
  --self-voice-name Matthew 

# --enroll-ai-voice
