REGION=$1
AWS_PROFILE=$2

aws cloudformation list-stacks \
  --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE \
  --query "StackSummaries[].StackName" \
  --output text --region $REGION --profile $AWS_PROFILE
