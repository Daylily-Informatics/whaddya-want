export AWS_REGION=us-west-2

LOG_GROUP_ARN="arn:aws:logs:us-west-2:670484050738:log-group:/aws/lambda/marvin-B-broker"

aws logs start-live-tail \
  --log-group-identifiers "$LOG_GROUP_ARN"
