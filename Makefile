.PHONY: clean clean-docker clean-finch clean-hard
clean:
	rm -rf .aws-sam
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
clean-docker:
	-docker ps -a | awk '/sam|rapid/ {print $$1}' | xargs -r docker rm -f
	-docker images | awk '/public\.ecr\.aws\/sam\/(build-|emulation-)/ {print $$3}' | xargs -r docker rmi -f
	-docker system prune -f
clean-finch:
	-finch ps -a | awk 'NR>1 {print $$1}' | xargs -r finch rm -f
	-finch images | awk 'NR>1 && /public\.ecr\.aws\/sam\/(build-|emulation-)/ {print $$3}' | xargs -r finch rmi -f
	-finch system prune -f
clean-hard: clean clean-docker
	rm -f samconfig.toml
	rm -rf .venv
