BROKER=$1

PYTHONPATH=. python -m client.cli \
  --broker-url $BROKER   \
  --id-window 4.0  --setup-devices --voice Amy --voice-mode neural --verbose
