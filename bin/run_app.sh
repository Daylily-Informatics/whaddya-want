BROKER=$1

ENABLE_OUTBOUND_SMS=1          # to actually send SMS
ENABLE_OUTBOUND_EMAIL=1        # to actually send email
ENABLE_SYSTEM_COMMANDS=1       # to allow run_command
ACTION_EMAIL_FROM=john@dyly.bio  # verified in SES

PYTHONPATH=. python -m client.cli \
  --broker-url $BROKER   \
  --setup-devices --voice Matthew --voice-mode generative --verbose -vv --self-voice-name Matthew

# --enroll-ai-voice
