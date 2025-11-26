AWS_REGION=us-west-2
STACK=$1

sam build

sam deploy \
  --stack-name $STACKNAME \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides LLMSecretArn=arn:aws:secretsmanager:...:secret:ai-companion \
  --resolve-s3 --region $AWS_REGION
