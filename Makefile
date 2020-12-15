.PHONY = dependencies

dependencies:
	pip freeze | grep -Ev -- "pytest|githooks|pylint|black" > requirements.txt