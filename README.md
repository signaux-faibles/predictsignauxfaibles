# predictsignauxfaibles
Dépôt du code python permettant la production de liste de prédiction Signaux Faibles.

## Dépendances / pré-requis
- python 3.8
- Docker (:construction_worker:)

## Installation pour un développeur/data scientist :

### Cloner et naviguer dans le repo
```
git clone git@github.com:signaux-faibles/predictsignauxfaibles.git
cd predictsignauxfaibles
```

### créer un environnement virtuel python (recommandé)
exemple avec [pyenv](https://github.com/pyenv/pyenv):
```
pyenv install 3.8.0
pyenv virtualenv 3.8.0 sf
pyenv local sf
```

### installer les dépendences du projet
```
pip install -r requirements-dev.txt
```

### activer les githooks
```
python -m python_githooks
```

Commencer à travailler !


## Structure du Dépot
(librement inspiré du [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science))

- `lib` contient l'essentiel du code nécéssaire à la production de listes signaux faibles
- `bin` contient des scripts utiles au projet
- `models` contient les artefacts de modèle (entrainés et serialisés), ses prédictions, et son évaluation
- `notebooks` contient les notebooks Jupyter (exploration, documentation intéractive, tutoriels, ...)
- `tests` contient les tests du projet : tests unitaires, d'intégration, et "end-to-end" (e2e). Le module python utilisé pour les tests est `pytest`.
- `Makefile` contient les commandes make pour l'execution de taches communes (`make train`, `make predict`, etc.)
- `config.py` est module de gestion de configuration du projet
- `requirements.txt` liste les dépendences (et leurs versions) nécessaires à la production d'une liste signaux faibles, `requirements-dev.txt` y rajoute les dépendences optionnelles qui ne concernent que les développeurs (pour les tests, le linting, etc.)
- `.githooks.ini` et `.pylintrc` dont des fichiers de configuration pour les githooks et le linter.

## Documentation

Un notebook jupyter interactif de démo est disponible [ici](./notebooks/00-get_started.ipynb).