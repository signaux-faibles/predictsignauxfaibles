.PHONY = dependencies

dependencies:
	pip freeze > requirements.txt

cd notebooks
curl --proxy socks5h://localhost:<PORT_INTERNET> -OL https://raw.githubusercontent.com/signaux-faibles/opensignauxfaibles/master/js/reduce.algo2/docs/variables.json -o variables.json