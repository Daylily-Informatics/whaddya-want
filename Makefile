# ========= AWS SAM build/deploy/sync =========
# Usage:
#   make build              # sam build (containerized, cached, parallel)
#   make deploy             # build + deploy (no interactive prompts)
#   make sync               # live-redeploy on file changes (ctrl-c to stop)
#   make sync-once          # one-shot fast sync (no watch)
#   make logs FN=MyFunction # tail logs for function logical id
#   make status             # cloudformation stack status/outputs
#   make delete             # delete the stack

# --- Config (override in env or .env) ---
-include .env
STACK        ?= whaddya-want
REGION       ?= us-west-2
PROFILE      ?= default
CAPS         ?= CAPABILITY_IAM CAPABILITY_NAMED_IAM
ENGINE       ?= docker             # or: finch
USE_CONTAINER?= --use-container    # remove to build on host
S3_PREFIX    ?= $(STACK)

AWS?=aws
SAM?=sam

AWS_FLAGS    ?= --region $(REGION) --profile $(PROFILE)
SAM_COMMON   ?= --region $(REGION) --profile $(PROFILE)

# When using finch, tell sam which engine to drive
ifdef ENGINE
  SAM_ENGINE = --container-engine $(ENGINE)
endif

# ========= Build / Deploy / Sync =========
.PHONY: build
build:
	$(SAM) build $(USE_CONTAINER) --cached --parallel $(SAM_ENGINE)

.PHONY: deploy
deploy: build
	$(SAM) deploy $(SAM_COMMON) \
	  --stack-name $(STACK) \
	  --resolve-s3 --s3-prefix $(S3_PREFIX) \
	  --capabilities $(CAPS) \
	  --no-confirm-changeset \
	  --no-fail-on-empty-changeset

# Fast code/config push without a full package/deploy
.PHONY: sync
sync:
	$(SAM) sync $(SAM_COMMON) \
	  --stack-name $(STACK) \
	  --capabilities $(CAPS) \
	  --watch

.PHONY: sync-once
sync-once:
	$(SAM) sync $(SAM_COMMON) \
	  --stack-name $(STACK) \
	  --capabilities $(CAPS)

# ========= Observability / Lifecycle =========
.PHONY: logs
logs:
ifndef FN
	$(error Provide FN=<LogicalFunctionName> (from template.yml))
endif
	$(SAM) logs $(SAM_COMMON) --stack-name $(STACK) -n $(FN) --tail

.PHONY: status
status:
	$(AWS) cloudformation describe-stacks $(AWS_FLAGS) --stack-name $(STACK) \
	| jq -r '.Stacks[0] | {StackName,StackStatus,Outputs}'

.PHONY: delete
delete:
	$(AWS) cloudformation delete-stack $(AWS_FLAGS) --stack-name $(STACK)
	@echo "Waiting for stack delete..."
	$(AWS) cloudformation wait stack-delete-complete $(AWS_FLAGS) --stack-name $(STACK) || true
	@echo "Delete complete (or already absent)."

# ========= Existing cleans (kept as-is) =========
.PHONY: clean clean-docker clean-finch clean-hard
clean:
	rm -rf .aws-sam
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
clean-docker:
	-docker ps -a | awk '/sam|rapid/ {print $$1}' | xargs -r docker rm -f
	-docker images | awk '/public\.ecr\.aws\/sam\/(build-|emulation-)/ {print $$3}' | xargs -r docker rmi -f
	-docker system prune -f
clean-finch:
	-finch ps -a | awk 'NR>1 {print $$1}' | xargs -r finch rm -f
	-finch images | awk 'NR>1 && /public\.ecr\.aws\/sam\/(build-|emulation-)/ {print $$3}' | xargs -r finch rmi -f
	-finch system prune -f
clean-hard: clean clean-docker
	rm -f samconfig.toml
	rm -rf .venv
