# predictsignauxfaibles
Dépôt du code python permettant la production de liste de prédiction Signaux Faibles.

## Structure du Dépot

- `lib` contient l'essentiel du code nécéssaire à la production de listes signaux faibles
- `bin` contient des scripts utiles au projet
- `notebooks` contient les notebooks Jupyter (exploration, documentation intéractive, tutoriels, ...)
- `tests` contient les tests du projet : tests unitaires, d'intégration, et "end-to-end" (e2e). Le module python utilisé pour les tests est `pytest`.
- `config.py` est module de gestion de configuration du projet
- `requirements.txt` liste les dépendences (et leurs versions) nécessaires à la production d'une liste signaux faibles, `requirements-dev.txt` y rajoute les dépendences optionnelles qui ne concernent que les développeurs (pour les tests, le linting, etc.)