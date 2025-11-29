#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Tuple

from agent_core.aws_model_client import AwsModelClient
from agent_core.logging_utils import setup_logging
from agent_core.schema import Event
from agent_core import memory_store
from agent_core.planner import handle_llm_result
from agent_core import tools as agent_tools, llm_client
from agent_core.actions import dispatch_background_actions


logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="CLI demo for the Rank-4 agent core")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-vv for debug logging)",
    )
    parser.add_argument(
        "--session",
        default="cli-session",
        help="Session identifier used when persisting events/memories",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("AGENT_ID", "marvin"),
        help="Agent identifier used when writing to storage",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None):
    args, unknown = _parse_args(argv)
    level = setup_logging(args.verbose)
    if unknown:
        logger.debug("Ignoring unknown CLI arguments: %s", unknown)

    logger.info("Starting CLI session", extra={"session": args.session, "agent": args.agent_id})
    logger.debug("Configured logging level: %s", logging.getLevelName(level))
    print("CLI demo for rank-4 agent core. Type Ctrl-D or Ctrl-C to exit.")

    model_client = AwsModelClient.from_env()
    logger.debug("Instantiated AwsModelClient using environment configuration")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            logger.info("CLI session terminated by user")
            break
        if not text:
            logger.debug("Received empty input; prompting again")
            continue

        logger.debug("Creating event for transcript: %s", text)
        evt = Event(
            agent_id=args.agent_id,
            session_id=args.session,
            source="user",
            channel="cli",
            ts=datetime.now(timezone.utc),
            payload={"transcript": text},
        )
        memory_store.put_event(evt)

        memories = memory_store.recent_memories(agent_id=args.agent_id, limit=20)
        logger.debug("Loaded %d recent memories", len(memories))
        system_prompt = llm_client.build_system_prompt(
            "You are Marvin, running in a local CLI demo."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "system", "content": "Recent memories: " + json.dumps(memories, default=str)},
        ]

        logger.debug("Sending %d messages to LLM client", len(messages))
        resp = llm_client.chat_with_tools(model_client, messages, agent_tools.TOOLS_SPEC)
        actions, new_mems, reply_text = handle_llm_result(resp, evt)
        logger.debug("Planner produced %d actions and %d new memories", len(actions), len(new_mems))
        for m in new_mems:
            memory_store.put_memory(m)
        dispatch_background_actions(actions)
        print(reply_text)


if __name__ == "__main__":
    main(sys.argv[1:])