BROKER=$1

PYTHONPATH=. python -m client.cli \
  --broker-url $BROKER   \
  --setup-devices --voice Matthew --voice-mode generative --verbose -vv --self-voice-name marvin

