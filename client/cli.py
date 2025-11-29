import argparse
import json
import logging
import os
from datetime import datetime, timezone

from agent_core.aws_model_client import AwsModelClient
from agent_core.schema import Event
from agent_core import memory_store
from agent_core.planner import handle_llm_result
from agent_core import tools as agent_tools, llm_client
from agent_core.actions import dispatch_background_actions
from agent_core.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI demo for rank-4 agent core.")
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("AGENT_ID", "marvin"),
        help="Agent identifier used for storage and routing.",
    )
    parser.add_argument(
        "--session",
        default="cli-session",
        help="Session identifier to group events.",
    )
    parser.add_argument(
        "--personality-prompt",
        default="You are Marvin, running in a local CLI demo.",
        help="Custom personality prompt to send to the LLM.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity; use -vv for detailed debug logs.",
    )

    # Compatibility placeholders for existing shell scripts; they currently do not
    # change behavior in the CLI demo but are accepted to avoid failures.
    parser.add_argument("--broker-url", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--setup-devices", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--voice", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--voice-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--self-voice-name", default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    logger.info("Starting CLI with agent_id=%s session=%s", args.agent_id, args.session)

    agent_id = args.agent_id
    session_id = args.session

    print("CLI demo for rank-4 agent core. Type Ctrl-D or Ctrl-C to exit.")
    model_client = AwsModelClient.from_env()

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue

        logger.debug("User input received: %s", text)
        evt = Event(
            agent_id=agent_id,
            session_id=session_id,
            source="user",
            channel="cli",
            ts=datetime.now(timezone.utc),
            payload={"transcript": text},
        )
        memory_store.put_event(evt)

        memories = memory_store.recent_memories(agent_id=agent_id, limit=20)
        system_prompt = llm_client.build_system_prompt(args.personality_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "system", "content": "Recent memories: " + json.dumps(memories, default=str)},
        ]

        logger.debug("Sending %s messages to llm_client", len(messages))
        resp = llm_client.chat_with_tools(model_client, messages, agent_tools.TOOLS_SPEC)
        logger.debug("LLM response keys: %s", list(resp.keys()))
        actions, new_mems, reply_text = handle_llm_result(resp, evt)
        for m in new_mems:
            memory_store.put_memory(m)
            logger.debug("Persisted CLI memory: %s", m.summary)
        dispatch_background_actions(actions)
        print(reply_text)


if __name__ == "__main__":
    main()
