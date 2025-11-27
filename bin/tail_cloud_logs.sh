export AWS_PROFILE=daylily
export AWS_REGION=us-west-2
STACK=$1

LOG_GROUP_ARN="arn:aws:logs:us-west-2:670484050738:log-group:/aws/lambda/${STACK}-broker"

aws logs start-live-tail \
  --log-group-identifiers "$LOG_GROUP_ARN"
