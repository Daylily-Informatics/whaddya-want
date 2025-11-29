# whaddya-want – Rank‑4 Agent Skeleton

This is a **reference implementation** of a Rank‑4 agent core for the
`whaddya-want` system.

It adds:

- A unified, typed memory model (`agent_core`).
- Explicit memory kinds: FACT, SPECULATION, AI_INSIGHT, ACTION, META.
- An agent broker Lambda that:
  - logs every incoming event,
  - retrieves recent memories,
  - calls an LLM with tool specs,
  - interprets tool calls into persistent memories and actions.
- A heartbeat Lambda that can self‑trigger the agent on a schedule.

## Important

This is a **drop‑in skeleton**, not a 1:1 refactor of your existing repo.
You will need to:

1. Merge `layers/shared/python/agent_core` into your shared layer.
2. Wire `lambda/broker/app.py` into your existing AWS SAM/CloudFormation
   template, or adapt these handler functions into your own broker Lambda.
3. Wire `lambda/agent_heartbeat/handler.py` into an EventBridge rule.
4. Replace the stubbed `llm_client` with your actual LLM backend.
5. Replace the stubbed `dispatch_background_actions` in `actions.py` with
   real integrations (SNS, email, lights, etc.).

The core ideas are fully implemented and ready to be adapted.
