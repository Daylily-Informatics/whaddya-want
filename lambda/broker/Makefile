# Target name MUST match build-<LogicalId> (BrokerFunction here)
build-BrokerFunction:
	# 1) Install Lambda deps into the artifact dir
	pip install --no-cache-dir -r requirements.txt -t "$(ARTIFACTS_DIR)"
	# 2) Copy the handler
	cp app.py "$(ARTIFACTS_DIR)/"
	# 3) Vendor shared code (your repo has src/ at ../../src)
	rsync -a --delete ../../src/ "$(ARTIFACTS_DIR)/"

	# 4) Sanity check
	python - <<'PY'
import os
ad = os.environ["ARTIFACTS_DIR"]
print("ARTIFACTS_DIR:", ad)
p = os.path.join(ad, "companion", "__init__.py")
print("companion exists:", os.path.exists(p))
PY
