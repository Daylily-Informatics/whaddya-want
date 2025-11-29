#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from agent_core.aws_model_client import AwsModelClient

from agent_core.schema import Event
from agent_core import memory_store
from agent_core.planner import handle_llm_result
from agent_core import tools as agent_tools, llm_client
from agent_core.actions import dispatch_background_actions




def main():
    agent_id = os.environ.get("AGENT_ID", "marvin")
    session_id = "cli-session"

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
        system_prompt = llm_client.build_system_prompt(
            "You are Marvin, running in a local CLI demo."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "system", "content": "Recent memories: " + json.dumps(memories, default=str)},
        ]

        resp = llm_client.chat_with_tools(model_client, messages, agent_tools.TOOLS_SPEC)
        actions, new_mems, reply_text = handle_llm_result(resp, evt)
        for m in new_mems:
            memory_store.put_memory(m)
        dispatch_background_actions(actions)
        print(reply_text)


if __name__ == "__main__":
    main()