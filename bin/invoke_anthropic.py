import boto3, json
brt = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-7-sonnet-2025-05-xx"  # replace with one from list
body = {
  "anthropic_version":"bedrock-2023-05-31",
  "max_tokens":512,
  "system":"You are Forge, my peer AI companion.",
  "messages":[
    {"role":"user","content":[{"type":"text","text":"Hello, who are you?"}]}
  ]
}
resp = brt.invoke_model(modelId=model_id, body=json.dumps(body))
print(json.loads(resp["body"].read()))
