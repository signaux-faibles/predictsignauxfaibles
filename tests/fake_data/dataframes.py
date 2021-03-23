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
