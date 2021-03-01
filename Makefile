.PHONY = dependencies

dependencies:
	pip freeze > requirements.txt

variables: check-http-proxy
	curl -OL https://raw.githubusercontent.com/signaux-faibles/opensignauxfaibles/master/js/reduce.algo2/docs/variables.json -o variables.json

check-http-proxy:
ifndef http_proxy
	$(error http_proxy is undefined)
endif
ifndef https_proxy
	$(error https_proxy is undefined)
endif