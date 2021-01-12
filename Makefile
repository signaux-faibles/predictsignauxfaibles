.PHONY = dependencies

dependencies:
	pip freeze | grep -Ev -- "pytest|githooks|pylint|black|matplotlib" > requirements.txt