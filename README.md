# whaddya-want

## Cloud-Based AI Companion via Online Services (Rank #3)

This design assembles an AI companion almost entirely from fully managed cloud
services. Your local device (Mac, PC, phone) works as a thin client that
captures audio/video, forwards it to cloud APIs, and plays the synthesized
responses that the cloud services generate. All heavy computation—speech
recognition, multimodal perception, reasoning, and text-to-speech—runs on the
cloud provider.

## Bootstrapping the repository

Run the helper script to create a Python environment (either a virtual
environment or Conda) and install the runtime dependencies for the Lambda
function and optional CLI client:

```bash
export AWS_PROFILE=your-aws-profile   # optional
bin/bootstrap.sh            # creates .venv/ by default
# or
bin/bootstrap.sh --mode conda
```

After the script completes activate the environment that was created and export
`PYTHONPATH=src` when running local tools or tests. The script will also alert
you if required system dependencies such as the AWS CLI are missing.

## Other
```bash
brew tap aws/tap
brew install aws-sam-cli

brew install finch
finch vm init
finch vm start
finch version

```

### High-Level Architecture

1. **Client capture & playback**
   - Record microphone audio and (optionally) webcam snapshots/video frames.
   - Forward raw audio streams to speech-to-text and collect synthesized audio
     from text-to-speech for playback.
2. **Speech pipeline**
   - Streaming capture to Amazon Transcribe for low-latency transcription.
   - Text responses rendered to voice with Amazon Polly neural voices.
3. **Vision pipeline (optional)**
   - Periodically capture still frames and send to Amazon Rekognition for
     face/object detection and recognition.
4. **Conversation engine**
   - Collate speech/vision context and send prompts to a managed LLM API (for
     example, OpenAI GPT-4/GPT-3.5 or Anthropic Claude) via a lightweight
     integration service hosted on AWS Lambda.
   - Persist dialog state, user profiles, and recognition metadata in DynamoDB.
5. **Action execution**
   - Define callable functions (e.g., calendar updates, smart-home commands) in
     Lambda. Let the LLM invoke them through structured function-calling hooks
     (OpenAI function calling / tool use).
6. **Orchestration & observability**
   - Use Amazon EventBridge or Step Functions to coordinate workflows, and
     CloudWatch for monitoring, logging, and alerts.

```
Client (macOS/iOS/Browser)
    ├─▶ Amazon Transcribe (Streaming STT)
    ├─▶ Amazon Rekognition (Vision)
    └─▶ API Gateway → Lambda (Conversation Broker)
                                 ├─▶ OpenAI / Anthropic API (LLM)
                                 ├─▶ DynamoDB (Session & user state)
                                 ├─▶ S3 (Audio/image storage, prompts)
                                 ├─▶ Amazon Polly (TTS)
                                 └─▶ Optional integrations (SNS, smart home, etc.)
```

### Core AWS Services and Roles

| Component               | AWS Service(s)                                   | Purpose                                                      |
|-------------------------|--------------------------------------------------|--------------------------------------------------------------|
| Speech recognition      | **Amazon Transcribe** (Streaming)                | Convert live audio to text, supports word timestamps.        |
| Speech synthesis        | **Amazon Polly** (Neural voices)                 | Generate lifelike speech; cache in S3 for reuse.             |
| Vision analysis         | **Amazon Rekognition** (Detect/Index Faces)      | Detect faces, label scenes, match enrolled users.            |
| Conversation backend    | **AWS Lambda + API Gateway**                     | Stateless API layer to route inputs between services.        |
| Reasoning/LLM           | **OpenAI/Anthropic API** via outbound HTTPS      | Provide high-quality conversational intelligence.            |
| State & memory          | **Amazon DynamoDB**                              | Persist session transcripts, embeddings, preferences.        |
| File/object storage     | **Amazon S3**                                    | Store audio clips, prompt assets, Rekognition face indexes.  |
| Event orchestration     | **Amazon EventBridge / Step Functions**          | Manage multi-step workflows and scheduled jobs.              |
| Metrics & logging       | **Amazon CloudWatch**                            | Aggregated logs, latency alarms, cost visibility.            |
| Authentication          | **Amazon Cognito / IAM**                         | Secure client access tokens and service roles.               |

### Step-by-Step AWS Setup Guide

> **Prerequisites**: AWS account with administrative access, AWS CLI (v2), an
> IAM user with programmatic credentials, Node.js or Python for client
> development, and the OpenAI/Anthropic API keys stored securely in AWS Secrets
> Manager.

1. **Provision foundational infrastructure**
   1. Create an S3 bucket (e.g., `ai-companion-artifacts`) for audio caches,
      captured frames, and prompt templates.
   2. Create a DynamoDB table (`ai-companion-sessions`) with a primary key on
      `sessionId` and a sort key on `timestamp` for transcript/event storage.
   3. Store LLM API keys in AWS Secrets Manager under a secret name such as
      `prod/ai-companion/llm`.
2. **Configure IAM roles and policies**
   1. Define an IAM role `ai-companion-lambda-role` with permissions for S3,
      DynamoDB, Transcribe, Polly, Rekognition, Secrets Manager read access, and
      CloudWatch logging.
   2. If using client authentication, create a Cognito User Pool and an
      identity pool granting authenticated users permission to invoke API
      Gateway endpoints.
3. **Set up Amazon Transcribe streaming**
   1. Create a transcription **vocabulary** or **language model** if you need
      custom terminology (optional).
   2. Implement a client module that uses the AWS SDK (WebSocket or HTTP/2) to
      start a `StartStreamTranscription` session and stream microphone audio.
   3. Capture partial transcription events and forward them to the conversation
      broker (Lambda/API Gateway).
4. **Deploy the conversation broker Lambda**
   1. Create an API Gateway (HTTP API) with routes `/ingest/audio` and
      `/ingest/vision` connected to a Lambda function (`ai-companion-broker`).
   2. The Lambda function should:
      - Retrieve the LLM API key from Secrets Manager.
      - Accept STT text and optional Rekognition insights.
      - Call the chosen LLM API with conversation context (stored/fetched from
        DynamoDB) and optional function definitions.
      - Interpret tool/function calls from the LLM to trigger other Lambdas or
        publish to EventBridge/SNS.
      - Generate a response payload and request speech synthesis from Polly.
   3. Package your Lambda code (Python example) with dependencies using
      AWS SAM/Serverless Framework or a container image and deploy.
5. **Integrate Amazon Polly for text-to-speech**
   1. Within the broker Lambda, call `SynthesizeSpeech` with a neural voice
      (e.g., `Joanna`, `Matthew`, `Amy`).
   2. Stream the resulting audio back to the client or store it in S3 with a
      signed URL for client retrieval.
6. **Enable vision features with Amazon Rekognition**
   1. For face recognition, create a Rekognition collection (e.g.,
      `ai-companion-faces`) and index known users via the `IndexFaces` API.
   2. From the client, capture frames (e.g., one every 5 seconds or on-demand)
      and send them to the `/ingest/vision` endpoint.
   3. Lambda uploads the frame to S3 (if not already) and calls `DetectFaces` or
      `SearchFacesByImage`, returning labeled results for the LLM context.
7. **Persist memory and context**
   1. Use DynamoDB streams (optional) to trigger downstream analytics Lambdas.
   2. Maintain rolling conversation history and embeddings. For semantic search
      you can integrate Amazon OpenSearch Serverless or store vector embeddings
      alongside text in DynamoDB using the new vector type.
8. **Client implementation**
   1. Build a macOS/iOS app (Swift) or cross-platform Electron/React app.
   2. Use WebRTC or native audio APIs to capture audio and send to Transcribe.
   3. Display conversation text, manage wake-word detection, and play Polly
      audio responses.
   4. Secure API calls with Cognito-issued tokens or SigV4 signed requests.
9. **Monitoring & cost controls**
   1. Enable CloudWatch metrics and alarms for latency, Lambda errors, and API
      throttling.
   2. Set AWS Budgets alerts for expected monthly cost ceilings.
   3. Leverage S3 lifecycle rules to expire stale audio/image artifacts.

### Operational Tips

- **Latency optimization**: Use AWS regions closest to your users (e.g.,
  `us-east-1` or `us-west-2`). Enable Transcribe streaming partial results and
  start Polly synthesis while the LLM response streams to minimize round-trip
  time.
- **Cost optimization**: Cache frequent Polly responses in S3, adjust
  Rekognition sampling frequency based on activity, and prefer GPT-3.5 or other
  lower-cost LLMs for casual chit-chat. Evaluate usage-based savings plans for
  Lambda if invocation volumes are predictable.
- **Privacy and compliance**: Encrypt S3 buckets, enforce TLS everywhere, and
  review AWS service data retention policies. Consider on-device preprocessing
  (e.g., wake-word detection) to limit what reaches the cloud.
- **Extensibility**: Add more tools via Lambda (calendar management, IoT
  control). Incorporate Step Functions for multi-turn actions that require
  waiting on external events.

### Reference Implementation

This repository now includes a minimal reference stack illustrating how the
components wire together:

- **`template.yaml`** – AWS SAM template that provisions DynamoDB, S3, and the
  Lambda/API Gateway pair that fronts the conversation workflow.
- **`lambda/broker`** – Python Lambda function that orchestrates DynamoDB
  memory, Secrets Manager (OpenAI credentials), the OpenAI Chat Completions
  endpoint, and Amazon Polly for speech synthesis.
- **`src/companion`** – Shared runtime package imported by the Lambda handler
  containing configuration helpers, DynamoDB persistence, OpenAI wrapper, and
  Polly/Rekognition utilities.
- **`client/cli.py`** – An optional local CLI that captures microphone audio,
  streams it to Amazon Transcribe, and forwards transcripts to the Lambda
  endpoint while saving synthesized audio responses to disk.

To deploy the backend with AWS SAM:

```bash
sam build --use-container
sam deploy \
  --stack-name ai-companion \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides LLMSecretArn=arn:aws:secretsmanager:...:secret:ai-companion \
  --resolve-s3 --region us-west-2
```

After deployment copy the output `BrokerEndpoint` URL and point the CLI client
at it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r client/requirements.txt
pip install amazon-transcribe sounddevice numpy requests
pip install --upgrade awscrt
pip install --upgrade pip
pip install amazon-transcribe sounddevice numpy requests awscrt

export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2

python client/cli.py --broker-url https://xxxx.execute-api.us-east-1.amazonaws.com/ingest/audio
```

### Usage Loop

1. User speaks → audio streamed to Transcribe → transcripts forwarded to Lambda.
2. Optional camera frame captured → Rekognition identifies subjects/context.
3. Lambda collates inputs → queries LLM (OpenAI/Anthropic) → receives text and
   optional tool invocations.
4. Lambda executes tool calls, updates state, and sends final response text to
   Polly.
5. Polly audio returned to client → played for the user.
6. Conversation transcript and metadata stored in DynamoDB/S3 for future
   context.

### Pros & Cons Recap

**Pros**
- Near-zero local infrastructure; accessible from any device.
- Access to state-of-the-art managed AI models with minimal setup.
- Elastic scaling: pay only for what you use.

**Cons**
- Ongoing operational cost with variable monthly spend.
- Dependence on third-party moderation, availability, and policy limits.
- Sensitive audio/video traverses cloud networks; requires trust and strong
  security practices.
- Latency can be higher than fully local or hybrid deployments, especially when
  chaining multiple APIs.

---

For deployments that require maximum control or reduced cost over time, see the
hybrid or fully local alternatives to complement this approach.


# Other
## Cleanup

```bash
# SAM build artifacts
rm -rf .aws-sam/

# Any zipped artifacts you created manually
find . -name '*.zip' -maxdepth 3 -print -delete

# Python build junk (if any)
find . -name '__pycache__' -type d -prune -print -exec rm -rf {} +
find . -name '*.pyc' -print -delete

# Optional: wipe venv (if you want to rebuild deps)
rm -rf .venv/

finch ps -a | awk 'NR>1 {print $1}' | xargs -r finch rm -f
finch images | awk 'NR>1 && /public\.ecr\.aws\/sam\/(build-|emulation-)/ {print $3}' | xargs -r finch rmi -f
finch system prune -f
rm -f samconfig.toml
sam delete --stack-name ai-companion --region us-west-2
# It will prompt to delete the S3 artifacts bucket; say yes to fully clean.
```