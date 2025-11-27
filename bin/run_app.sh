BROKER=$1

PYTHONPATH=. python -m client.cli \
  --broker-url $BROKER   \
  --setup-devices --voice Matthew --voice-mode generative --verbose -vv --enroll-ai-voice --self-voice-name Matthew

