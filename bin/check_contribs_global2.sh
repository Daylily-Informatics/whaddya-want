#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./check_contribs_global2.sh [--start ISO8601] [--end ISO8601] ORG
#   ./check_contribs_global2.sh [--start ISO8601] [--end ISO8601] ORG repo1
#   ./check_contribs_global2.sh [--start ISO8601] [--end ISO8601] ORG "repo1,repo2,repo3"
#   ./check_contribs_global2.sh [--start ISO8601] [--end ISO8601] ORG repos.csv
#
# Examples:
#   ./check_contribs_global2.sh Daylily-Informatics
#   ./check_contribs_global2.sh --start 2025-03-01T00:00:00Z --end 2025-04-01T00:00:00Z Daylily-Informatics daylily-ephemeral-cluster

START_DATETIME=""
END_DATETIME=""

# ---- parse flags ----
positional=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --start=*)
      START_DATETIME="${1#*=}"
      shift
      ;;
    --end=*)
      END_DATETIME="${1#*=}"
      shift
      ;;
    --start)
      START_DATETIME="${2:-}"
      shift 2
      ;;
    --end)
      END_DATETIME="${2:-}"
      shift 2
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done
set -- "${positional[@]}"

# ---- positional args: ORG [REPO_FILTER] ----
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 [--start ISO8601] [--end ISO8601] ORG [REPO_CSV_OR_LIST]" >&2
  exit 1
fi

ORG="$1"
REPO_FILTER="${2:-}"

# Always define the array so set -u doesn't freak out
REPO_LIST=()

if [ -n "$REPO_FILTER" ]; then
  if [ -f "$REPO_FILTER" ]; then
    # -------- REPOS FROM CSV FILE (first column) --------
    while IFS=, read -r col1 _; do
      name="$col1"
      name="${name##*/}"              # strip org/ if present
      name="$(echo "$name" | xargs)"  # trim whitespace
      case "$name" in
        ""|"repo"|"name") continue ;;
      esac
      REPO_LIST+=("$name")
    done < "$REPO_FILTER"
  else
    # -------- REPOS FROM COMMA-LIST OR SINGLE NAME --------
    IFS=',' read -ra raw_repos <<< "$REPO_FILTER"
    for r in "${raw_repos[@]}"; do
      name="${r##*/}"
      name="$(echo "$name" | xargs)"
      [ -z "$name" ] && continue
      REPO_LIST+=("$name")
    done
  fi
else
  # -------- NO FILTER: ALL REPOS IN ORG --------
  mapfile -t REPO_LIST < <(
    gh repo list "$ORG" --limit 1000 --json name -q '.[].name'
  )
fi

if [ "${#REPO_LIST[@]}" -eq 0 ]; then
  echo "No repositories found for '$ORG' with filter '$REPO_FILTER'" >&2
  exit 1
fi

# Header
echo -e "type\trepo\tid\tdate\tlogin\tauthor_name\tauthor_email\ttitle_or_message\turl"

# -------- COMMITS FOR SELECTED REPOS --------
for REPO in "${REPO_LIST[@]}"; do
  gh api "/repos/$ORG/$REPO/commits?per_page=100" \
    --paginate \
    --jq '.[]' 2>/dev/null \
  | jq -r \
      --arg repo "$ORG/$REPO" \
      --arg start "$START_DATETIME" \
      --arg end "$END_DATETIME" '
      select(.sha != null)
      | select(
          ($start == "" or .commit.author.date >= $start)
          and
          ($end   == "" or .commit.author.date <= $end)
        )
      | [
          "commit",
          $repo,
          .sha,
          .commit.author.date,
          (.author.login // ""),
          (.commit.author.name // ""),
          (.commit.author.email // ""),
          (.commit.message // "" | gsub("\n"; " ") | .[0:200]),
          .html_url
        ] | @tsv
    ' || true
done

# -------- PRs FOR SELECTED REPOS --------
for REPO in "${REPO_LIST[@]}"; do
  gh api /search/issues \
    --paginate \
    -F q="repo:$ORG/$REPO type:pr" \
    --jq '.items[]' 2>/dev/null \
  | jq -r \
      --arg start "$START_DATETIME" \
      --arg end "$END_DATETIME" '
      select(
        ($start == "" or .created_at >= $start)
        and
        ($end   == "" or .created_at <= $end)
      )
      | [
          "pr",
          .repository.full_name,
          (.number|tostring),
          .created_at,
          (.user.login // ""),
          "",
          "",
          (.title // "" | gsub("\n"; " ") | .[0:200]),
          .html_url
        ] | @tsv
    ' || true
done
