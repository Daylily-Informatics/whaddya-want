# tool_runner.py - called when reply contains {"action":...}
import json, subprocess, sys
SAFE_ACTIONS={"open_url": lambda url: subprocess.Popen(["open", url])}

if __name__=="__main__":
    data=json.loads(sys.stdin.read())
    act=data.get("action")
    if act in SAFE_ACTIONS: SAFE_ACTIONS[act](**{k:v for k,v in data.items() if k!="action"})
