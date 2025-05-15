import os
import sys

from evaluation import evaluate_distribution
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("..")
from src.datasets.nie_wager_synthetic import (
    simulate_easy_propensity_difficult_baseline,
    simulate_nuisance_and_easy_treatment,
    simulate_randomized_trial,
    simulate_unrelated_treatment_control,
)

NUM_COV = 5
NUM_EXPS = 100


def main(setup_name, heteroscedastic_epsilon=False, sim_number=None):
    """
    Run the evaluation for the synthetic dataset
    :param setup_name: str, either "A" or "B" or "C" or "D"
    :param heteroscedastic_epsilon: bool, whether to use heteroscedastic noise
    :param sim_number: int, the simulation number to run
    """
    if setup_name not in ["A", "B", "C", "D"]:
        raise ValueError("setup_name must be either 'A' or 'B' or 'C' or 'D'")

    if setup_name == "A":
        simulate_function = simulate_nuisance_and_easy_treatment
    elif setup_name == "B":
        simulate_function = simulate_randomized_trial
    elif setup_name == "C":
        simulate_function = simulate_easy_propensity_difficult_baseline
    elif setup_name == "D":
        simulate_function = simulate_unrelated_treatment_control

    range_start = 0 if sim_number is None else sim_number - 1
    range_end = NUM_EXPS if sim_number is None else sim_number
    for n in tqdm(range(range_start, range_end)):
        df = simulate_function(n=5000, c=0.0, heteroscedastic=heteroscedastic_epsilon)
        df_train, df_test = train_test_split(df)
        # save the data
        if heteroscedastic_epsilon:
            # check if the directory exists
            if not os.path.exists(
                f"../results/outputs/nie_wager/setup{setup_name}/data/heteroscedastic"
            ):
                os.makedirs(f"../results/outputs/nie_wager/setup{setup_name}/data/heteroscedastic")
            df_train.to_csv(
                f"../results/outputs/nie_wager/setup{setup_name}/data/heteroscedastic/simulations_setup{setup_name}_{str(n+1)}_train.csv",
                index=False,
            )
            df_test.to_csv(
                f"../results/outputs/nie_wager/setup{setup_name}/data/heteroscedastic/simulations_setup{setup_name}_{str(n+1)}_test.csv",
                index=False,
            )
            print(
                f"Saved synthetic data to ../results/outputs/nie_wager/setup{setup_name}/data/heteroscedastic"
            )
        else:
            if not os.path.exists(f"../results/outputs/nie_wager/setup{setup_name}/data"):
                os.makedirs(f"../results/outputs/nie_wager/setup{setup_name}/data")
            df_train.to_csv(
                f"../results/outputs/nie_wager/setup{setup_name}/data/simulations_setup{setup_name}_{n+1}_train.csv",
                index=False,
            )
            df_test.to_csv(
                f"../results/outputs/nie_wager/setup{setup_name}/data/simulations_setup{setup_name}_{n+1}_test.csv",
                index=False,
            )
            print(f"Saved synthetic data to ../results/outputs/nie_wager/setup{setup_name}/data")

        if heteroscedastic_epsilon:
            output_path = f"../results/outputs/nie_wager/setup{setup_name}/heteroscedastic/simulations_setup{setup_name}_{n+1}_results.csv"
        else:
            output_path = f"../results/outputs/nie_wager/setup{setup_name}/simulations_setup{setup_name}_{n+1}_results.csv"
        evaluate_distribution(
            df_train=df_train,
            df_test=df_test,
            num_cov=NUM_COV,
            output_path=output_path,
            dataset_name="nie_wager_synthetic",
            alpha=0.1,
            verbose=True,
            store_p_values=False,
        )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], bool(sys.argv[2]))
    elif len(sys.argv) == 4:
        assert sys.argv[2] in ["True", "False"], "heteroscedastic must be either True or False"
        hetroscedastic = False if sys.argv[2] == "False" else True
        main(sys.argv[1], hetroscedastic, int(sys.argv[3]))
    else:
        print(
            "Usage: python nie_wager_synthetic.py [A|B|C|D] [heteroscedastic] or python nie_wager_synthetic.py [A|B|C|D] [heteroscedastic] [sim_number]"
        )
        sys.exit(1)
