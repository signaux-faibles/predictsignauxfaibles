FEATURES_LIST = [
    "_id.siret",
    "_id.periode",
    "value.code_commune",
    "value.departement",
    "value.region",
    "value.age",
    "value.effectif",
    "value.effectif_entreprise",
    "value.time_til_failure",
    "value.outcome",
    "value.time_til_outcome",
    # insee
    "value.code_naf",
    "value.libelle_naf",
    "value.code_ape_niveau2",
    "value.libelle_ape2",
    "value.code_ape_niveau3",
    "value.libelle_ape3",
    "value.code_ape",
    # urssaf
    "value.montant_part_ouvriere",
    "value.ratio_dette",
    "value.cotisation",
    "value.cotisation_moy12m",
    "value.montant_part_ouvriere",
    "value.montant_part_patronale",
    "value.apart_heures_autorisees",
    "value.apart_heures_consommees",
    "value.apart_heures_consommees_cumulees",
    "value.apart_entreprise",
    "value.paydex_nb_jours",  # paydex
    "value.dette_fiscale",  # bdf
    "value.endettement",  # diane
    "value.taux_endettement",  # diane
]

LIBELLE_NAF = {
    "A": "Agriculture, sylviculture et pêche",
    "B": "Industries extractives",
    "C": "Industrie manufacturière",
    "D": "Production et distribution d'électricité, de gaz, de vapeur et d'air conditionné",
    "E": "Production et distribution d'eau ; assainissement, gestion des déchets et dépollution",
    "F": "Construction",
    "G": "Commerce ; réparation d'automobiles et de motocycles",
    "H": "Transports et entreposage",
    "I": "Hébergement et restauration",
    "J": "Information et communication",
    "K": "Activités financières et d'assurance",
    "L": "Activités immobilières",
    "M": "Activités spécialisées, scientifiques et techniques",
    "N": "Activités de services administratifs et de soutien",
    "O": "Administration publique",
    "P": "Enseignement",
    "Q": "Santé humaine et action sociale",
    "R": "Arts, spectacles et activités récréatives",
    "S": "Autres activités de services",
    "T": "Activités des ménages en tant qu'employeurs;\
        activités indifférenciées des ménages en tant que\
        producteurs de biens et services pour usage propre",
    "U": "Activités extra-territoriales",
}

NAN_RPL = {
    "montant_part_ouvriere": 0.0,
    "montant_part_patronale": 0.0,
    "apart_heures_autorisees": 0.0,
    "apart_heures_consommees": 0.0,
    "apart_heures_consommees_cumulees": 0.0,
    "dette_fiscale": 0.0,
    "endettement": 0.0,
    "taux_endettement": 0.0,
}

CODES_REGION = {
    "Guadeloupe": 1,
    "Martinique": 2,
    "Guyane": 3,
    "La Réunion": 4,
    "Mayotte": 6,
    "Île-de-France": 11,
    "Centre-Val de Loire": 24,
    "Bourgogne-Franche-Comté": 27,
    "Normandie": 28,
    "Hauts-de-France": 32,
    "Grand Est": 44,
    "Pays de la Loire": 52,
    "Bretagne": 53,
    "Nouvelle-Aquitaine": 75,
    "Occitanie": 76,
    "Auvergne-Rhône-Alpes": 84,
    "Provence-Alpes-Côte d'Azur": 93,
    "Corse": 94,
}

NAF_INDUSTRY = ["B", "C", "D", "E"]
