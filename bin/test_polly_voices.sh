#!/usr/bin/env sh
set -eu

REGION="${REGION:-us-west-2}"
OUT_DIR="${OUT_DIR:-polly_voices}"
TEXT_TEMPLATE='Hello, I am %s, today is Thursday. Goodbye'

command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 1; }
mkdir -p "$OUT_DIR"

aws polly describe-voices --region "$REGION" --output json \
| jq -r '.Voices[] | "\(.Id)\t\((.SupportedEngines // ["standard"]) | join(","))"' \
| sort \
| while IFS="$(printf '\t')" read -r VOICE ENGINES; do
  TEXT=$(printf "$TEXT_TEMPLATE" "$VOICE")
  OUT="$OUT_DIR/$VOICE.mp3"
  echo "[$VOICE] -> $OUT"

  case ",$ENGINES," in
    *,neural,*)  # voice supports neural
      aws polly synthesize-speech \
        --region "$REGION" \
        --voice-id "$VOICE" \
        --engine neural \
        --output-format mp3 \
        --text "$TEXT" \
        "$OUT" >/dev/null
      ;;
    *)           # standard only (omit --engine for max compatibility)
      aws polly synthesize-speech \
        --region "$REGION" \
        --voice-id "$VOICE" \
        --output-format mp3 \
        --text "$TEXT" \
        "$OUT" >/dev/null
      ;;
  esac
done
