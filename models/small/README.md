# Modèle "small"

Ce modèle est celui exécuté par par la commande `python -m predictsignauxfaibles --model_name small`. Il est utilisé pour les établissements pour lesquels nous ne possédons pas de données financières ou de comportement de paiement.

## Données

Ce modèle utilise 2 types de données :
- des données d'activité partielle
- des données d'endettement auprès de l'URSSAF

## Modèle

Ce modèle est une régression linéaire multivariée.
