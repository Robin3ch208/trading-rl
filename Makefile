lint:
	ruff check .
	mypy src/

test: 
	pytest

check-secrets:
	@grep -rnw . -e "CIR" && echo "ERREUR : Références privées trouvées !" && exit 1 || echo "Aucun mot privé trouvé."

check: lint test
