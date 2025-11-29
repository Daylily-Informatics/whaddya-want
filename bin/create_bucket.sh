ACC=$(aws sts get-caller-identity --query Account --output text)
REG=us-east-1
BUCKET="companion-${ACC}-${REG}"
aws s3api create-bucket --bucket "$BUCKET" --region $REG --create-bucket-configuration LocationConstraint=$REG
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
