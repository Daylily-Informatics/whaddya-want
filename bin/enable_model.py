# list_models.py
import boto3, json
br = boto3.client("bedrock-runtime", region_name="us-east-1")
bd = boto3.client("bedrock", region_name="us-east-1")
print(bd.list_foundation_models()["modelSummaries"])  # discover modelIds
