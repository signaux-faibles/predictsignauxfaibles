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

## Exemple d'utilisation (vision cible)

Instantier un nouveau modèle

```python
from lib.models import SFModelGAM

features = [
    "dette_urssaf",
    "ca_2020",
    "exports"
]

model = SFModelGAM(
    id = "modele_gam_sf_janvier_2020",
    target = "outcome",
    features = features
)
```

Il est aussi possible d'initialiser un modèle via un fichier de configuration (recommandé pour la prod)

```python
# the default prod model
model = SFModelGAM.from_config_file("prod")

# or any specific model used in the past
model = SFModelGAM.from_config_file("models/1.2.15/config.yaml")
```

Entainer le modèle, l'évaluer, le sauvegarder pour plus tard, générer des prédictions

```python

# Populate model object with data for some SIRENs only
model.get_data(
    sirens = [123456789, 123456788, 123456777],
    date_min = "2015-12-01",
    date_max = "2018-09-31",
    sample_size = 1_000_000
)

# train model
model.train()

# evaluate model's performance
model.evaluate(metric = "GINI")

# export a model summary (to be version-controlled)
model.export_summary("models/1.2.15/performance.json")

# serialise the model and save it to disk (e.g. via joblib)
model.save("models/1.2.15/model.sav")

# make prediction on new data

sf_dataset = SFDataset()
sf_dataset.get_data(
    sirens = [123456789, 123456788, 123456777],
    features = features,
    date_min = "2020-11-30",
    date_max = "2020-12-31"
)

predictions = model.predict(sf_dataset.data)
```