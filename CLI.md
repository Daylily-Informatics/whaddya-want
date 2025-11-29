# Command-Line Interfaces

This repository ships several command-line entry points for local development, deployment, and experimentation. Use the sections below to discover what each command does and how to invoke it.

## Voice client (primary CLI)

The interactive voice client lives at `client/cli.py` and can be launched directly or via `python -m client.cli`. Core options include:

- `--broker-url` (required unless set in `client/config.yaml`): API Gateway or Lambda ingest endpoint for audio/requests.
- `--session`: Session identifier (defaults to a random UUID).
- `--region` / `--language`: AWS Transcribe region and language code.
- `--input-device` / `--setup-devices`: Select or configure microphone/speaker devices; persisted under `~/.whaddya`.
- `--identify_image`: One-shot face identification for a supplied image file, then exit.
- `--rate` / `--channels`: Transcribe sampling rate (Hz) and channel count.
- `--id-threshold` / `--id-window`: Speaker-identification confidence and rolling window tuning.
- `--auto-register-name` / `--force-enroll`: Automatically enroll unknown speakers with a name or force re-enrollment.
- `--voice` / `--voice-mode`: Choose Polly voice and synthesis engine (`standard`, `neural`, or `generative`).
- `--text-only`: Request text responses instead of synthesized audio.
- `--save-audio`: Persist broker audio replies to disk.
- `--verbose` / `-vv`: Increase logging detail; `-vv` enables debug logging.
- `--enroll-ai-voice` / `--self-voice-name`: Enroll or suppress the AI's own voice to avoid echo/self-talk.

The vision/audio monitor launches automatically with the client; use the `kill monitor` voice command if you need to pause it during a session.

Example:

```bash
PYTHONPATH=. python -m client.cli \
  --broker-url https://example.execute-api.us-west-2.amazonaws.com/ingest/audio \
  --session my-session-123 \
  --setup-devices --voice Joanna --voice-mode neural --verbose
```

## Build and deployment make targets

The `Makefile` wraps common AWS SAM actions:

- `make build` — Run `sam build` with containerization and caching.
- `make deploy` — Build then deploy the stack (non-interactive, resolves S3 packaging).
- `make sync` — Watch local changes and sync them live to the stack.
- `make sync-once` — One-shot fast sync without watching.
- `make logs FN=<LogicalFunctionName>` — Tail Lambda logs for the given function.
- `make status` — Show CloudFormation stack status and outputs via `aws cloudformation describe-stacks`.
- `make delete` — Delete the deployed stack and wait for completion.
- `make clean` — Remove SAM build artifacts and Python bytecode.
- `make clean-docker` / `make clean-finch` — Prune container artifacts for Docker or Finch.
- `make clean-hard` — Run `clean` and `clean-docker`, then remove the virtualenv and `samconfig.toml`.

## Bootstrap and environment helpers (bin/)

- `bin/bootstrap.sh [options]` — Create a Python 3.11 virtualenv or Conda env, verify system dependencies, and install Lambda/client requirements.
- `bin/install.sh` / `bin/install.sh2` — Conda environment creation and macOS/Homebrew setup helpers for local tooling.
- `bin/mon_setup.sh` — Minimal setup script for local monitoring dependencies (Ultralytics, Vosk, etc.).

## Deployment utilities (bin/)

- `bin/build_local.sh <stack>` / `bin/update_stack.sh <stack>` — Build and deploy the SAM stack to the configured AWS account/region.
- `bin/create_bucket.sh` — Create and encrypt the S3 bucket used for artifacts based on the current AWS account/region.
- `bin/rekognition.sh` — Create a Rekognition collection and upload face images from `./faces/`.
- `bin/print_stack_names.sh <region> <profile>` — List CloudFormation stack names for a region/profile.
- `bin/tail_cloud_logs.sh <stack>` — Live-tail the broker Lambda CloudWatch log group for a given stack name.
- `bin/update_stack.sh <stack>` — Rebuild and redeploy the stack with fixed parameters.

## AI/ML experiment helpers (bin/)

- `bin/aai.py` — End-to-end MVP loop: capture audio, run STT, query Bedrock (Anthropic model), and synthesize a Polly reply.
- `bin/install_trascribe.py` — Minimal streaming STT example using Amazon Transcribe.
- `bin/invoke_anthropic.py` — Invoke an Anthropic model via Bedrock with a sample prompt.
- `bin/enable_model.py` — List available Bedrock foundation models.
- `bin/polly.py` — Generate a sample Polly MP3 (Joanna voice) locally.
- `bin/test_polly_voices.sh` — Iterate through available Polly voices and synthesize sample audio files.
- `bin/runtime_search.py` — Search a Rekognition collection for the best face match in a local image file.
- `bin/entroll_faces.py` — Index sample faces from S3 into the `companion-people` Rekognition collection.
- `bin/capframe.sh` — Capture a single webcam frame (`imagesnap`) and run face search.

## Identity and memory utilities (bin/)

- `bin/dump_memory.py [--target short|long|both]` — Dump DynamoDB conversation and AIS memory tables to JSON via DynamoDB Scan.
- `bin/dump_face_profiles.py` — Export face-profile metadata and embeddings recorded locally.
- `bin/dump_voice_profiles.py` — Export voice-profile metadata and embeddings recorded locally.
- `bin/unenroll_profiles.py --name <profile> [--type voice|face|both]` — Remove enrolled voice/face profiles locally and optionally purge Rekognition entries.
- `bin/exec_on_my_behalf.py` — Execute limited local actions (e.g., `open_url`) based on JSON tool calls piped to stdin.
- `bin/run_app.sh <broker> <session>` — Launch the voice client with preconfigured feature flags for outbound actions and self-voice suppression.

## Contributor reporting (bin/)

- `bin/check_contribs_global2.sh [--start ISO8601] [--end ISO8601] ORG [REPO_FILTER]` — Aggregate GitHub contributions (commits/issues/PRs) across one or more repositories in an organization.

> Many scripts assume environment variables such as `AWS_PROFILE`, `AWS_REGION`, `BUCKET`, or Rekognition collection IDs are already exported. Review each script before use to confirm prerequisites.
