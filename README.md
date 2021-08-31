# predictsignauxfaibles

Dépôt du code python permettant la production de liste de prédiction Signaux Faibles.

## Dépendances / pré-requis

- python 3.6.4
- un accès à la base de données du projet

## Installation pour un développeur/data scientist :

### Cloner et naviguer dans le repo
```
git clone git@github.com:signaux-faibles/predictsignauxfaibles.git

```
### Optionnel : Pour les personnes qui travaillent sur le serveur labtenant avec un proxy

Pour pouvoir télécharger les packages, leurs dépendances ainsi que la documentation de données de opensignauxfaibles, il est nécessaire de prendre en compte le proxy pour les personnes qui travaillent sur le serveur. 

Par exemple, pour installer un package : 

```
pip install --proxy socks5h://localhost:<PORT_INTERNET> <MON_PACKAGE>
```
Pour éviter de fournir l'option --proxy à chaque fois, vous pouvez créer un fichier ~/.conf/pip/pip.conf 

```
mkdir -p ~/.config/pip
nano pip.conf
```
Et y ajouter la configuration suivante :

```
[global]
proxy = socks5h://localhost:<PORT_INTERNET>
```
Cette configuration est cruciale pour l'installation automatique des packages indiqué dans requirements.  

### créer un environnement virtuel python (recommandé)

Exemple avec [pyenv](https://github.com/pyenv/pyenv) :

```
pyenv install 3.6.4
pyenv virtualenv 3.6.4 sf
pyenv local sf
```

### installer les dépendences du projet

```
pip install -r requirements.txt
```

Note: la procédure sur le serveur est légèrement différente.

### activer les githooks

```
python -m python_githooks
```

Commencer à travailler !

## Faire tourner un modèle

Les modèles sont configurés dans un fichier de configuration en python dans `models/<model_name>/model_conf.py`. Certaines valeurs peuvent être changées via le CLI (`python -m predictsignauxfaibles --help`).

Chaque run de modèle produit 2 fichiers dans `./model_runs/<model_id>` **qui n'est pas commité sur Git** :
- les prédictions du modèle
- des statistiques de performance du modèle

La variable d'environnement `ENV` permet de faire tourner le modèle en mode `develop` (utilisant moins de données, le défaut) ou bien en `prod`:

```sh
export ENV=prod
python -m predictsignauxfaibles
```

## Structure du Dépot
(librement inspiré du [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science))

- `lib/` contient l'essentiel du code nécéssaire à la production de listes signaux faibles.
- `bin/` contient des scripts utiles au projet.
- `models/` contient les artefacts de modèle (entrainés et serialisés), ses prédictions, et son évaluation.
- `model_run/` contient les prédictions, et les métriques d'évaluation des modèles exécutés.
- `notebooks/` contient les notebooks Jupyter (exploration, documentation intéractive, tutoriels…)
- `tests/` contient les tests du projet : tests unitaires, d'intégration, et "end-to-end" (e2e). Le module python utilisé pour les tests est `pytest`.
- `Makefile` peut être utilisé pour l'execution de taches communes (`make train`, `make predict`, etc.) _Rq_ : Ceci n'est pas implémenté car déjà couvert par l'utilitaire en ligne de commandes.
- `setup.cfg` est un fichier contenant les métaonnées du projet utilisées par `setuptools` lors de l'installation du projet.
- `requirements.txt` liste les dépendences (et leurs versions) nécessaires à la production d'une liste signaux faibles.
- `.githooks.ini` et `.pylintrc` sont les fichiers de configuration pour les githooks et le linter.

## Documentation

Un notebook jupyter interactif de démo est disponible [ici](./notebooks/00-get_started.ipynb).

Il est également possible de télécharger la liste des features disponibles pour le modèle Signaux Faibles, fichier `json` présent dans le dépôt Github opensignauxfaibles. Ce listing est utilisé dans le notebook de démo. 

```
cd notebooks
curl --proxy socks5h://localhost:<PORT_INTERNET> -OL https://raw.githubusercontent.com/signaux-faibles/opensignauxfaibles/master/js/reduce.algo2/docs/variables.json -o variables.json
cd ..
```

## Gestion des fonctions aléatoires

Les fonctions aléatoires doivent être décorées avec le décorateur `is_random` (dans `lib.decorators`).

Dès lors, la variable d'environnement `RANDOM_SEED` permet aux fonctions aléatoires d'être déterministes. Par défaut, cela n'est pas le cas. Cette variable doit être un entier.

Au début d'un notebook, vous pouvez créer cette variable d'environnement de la manière suivante :

```python
import os
os.environ["RANDOM_SEED"]="42"
```

Ou depuis le terminal :
```sh
export RANDOM_SEED=42
```
