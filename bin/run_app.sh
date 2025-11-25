BROKER=$1

PYTHONPATH=. python -m client.cli \
  --broker-url $BROKER   \
  --setup-devices --voice Amy --voice-mode neural --verbose
