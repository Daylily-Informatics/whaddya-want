
STACKNAME=$1
export AWS_REGION=us-west-2        
export AWS_DEFAULT_REGION=us-west-2                                                   
export AWS_PROFILE=daylily                                                  

sam build
sam deploy \
  --stack-name $STACKNAME \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides LLMSecretArn=arn:aws:secretsmanager:...:secret:ai-companion \
  --resolve-s3 --region $AWS_REGION
