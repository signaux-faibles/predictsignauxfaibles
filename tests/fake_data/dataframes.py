import pandas as pd

df_test_acoss = pd.DataFrame(
    {
        "effectif": [10, 10, 10_000],
        # part sociale
        "montant_part_patronale": [1_000, 1_200, 0],
        "montant_part_patronale_past_3": [1_000, 0, 1_200],
        # part ouvriere
        "montant_part_ouvriere": [0, 300, 1_200],
        "montant_part_ouvriere_past_3": [0, 0, 1_200],
    }
)

df_test_paydex = pd.DataFrame(
    {
        "paydex_nb_jours": [10, 30, 10_000],
        "paydex_nb_jours_past_12": [91, 30, 15],
    }
)

df_test_code_naf = pd.DataFrame(
    {
        "code_naf": ["A", "B", "E", "O", "P"],
    }
)

df_test_full = pd.concat((df_test_code_naf, df_test_acoss, df_test_paydex), axis=1)

df_test_redressement_urssaf = pd.DataFrame(
    {
        "montant_part_ouvriere_latest": [50, 60, 100],
        "montant_part_patronale_latest": [75, 50, 100],
        "montant_part_ouvriere_july2020": [50, 50, 50],
        "montant_part_patronale_july2020": [50, 50, 0],
        "cotisation_moy12m_latest": [100, 100, 75],
        "group_final": ["vert", "vert", "rouge"],
    }
)

## df for testing explain functions
df_test_explain_prod = pd.DataFrame(
    {
        "equilibre_financier": [-0.05, 0.01, 0.05],
        "paydex_yoy": [1, 1, -0.5],
        "paydex_other_feat": [0.1, 0, -0.05],
        "ratio_dette": [0, 10, -5],
    }
)
