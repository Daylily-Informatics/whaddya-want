# GNU make 3.82+ supports this: lets us use '|' instead of a TAB
.RECIPEPREFIX := |

# Target name MUST match build-<LogicalId> from template.yaml
.PHONY: build-BrokerFunction
build-BrokerFunction:
| python -V
| pip --version
| echo "ARTIFACTS_DIR=$${ARTIFACTS_DIR}"
|
| # 1) Lambda deps into the artifact dir
| pip install --no-cache-dir -r requirements.txt -t "$${ARTIFACTS_DIR}"
|
| # 2) Copy handler
| cp app.py "$${ARTIFACTS_DIR}/"
|
| # 3) Vendor shared code (your repo has src/ two levels up)
| mkdir -p "$${ARTIFACTS_DIR}/companion"
| cp -a ../../src/companion "$${ARTIFACTS_DIR}/"
|
| # 4) Sanity check
| python - <<'PY'
| import os, pathlib
| ad = os.environ["ARTIFACTS_DIR"]
| p  = pathlib.Path(ad) / "companion" / "__init__.py"
| print("companion exists:", p.exists(), "->", p)
| PY
