import os
import sys

from evaluation import evaluate_distribution
from tqdm import tqdm

sys.path.append("..")

import pandas as pd

from src.datasets.alaa_synthetic import get_data

NUM_COV = 10
NUM_EXPS = 100
setup_A = {"n": 5000, "d": 10, "gamma": 1, "alpha": 0.1}
setup_B = {"n": 5000, "d": 10, "gamma": 0, "alpha": 0.1}


def main(setup_name, sim_number=None):
    """
    Run the evaluation for the synthetic dataset
    :param setup_name: str, either "A" or "B"
    :param sim_number: int, the simulation number to run
    """
    if setup_name not in ["A", "B"]:
        raise ValueError("setup_name must be either 'A' or 'B'")

    range_start = 0 if sim_number is None else sim_number - 1
    range_end = NUM_EXPS if sim_number is None else sim_number
    for n in tqdm(range(range_start, range_end)):
        print(f"Running experiment {n+1}")
        df_train, df_test = get_data(**setup_A) if setup_name == "A" else get_data(**setup_B)
        evaluate_distribution(
            df_train=df_train,
            df_test=df_test,
            num_cov=NUM_COV,
            output_path=f"../results/outputs/alaa/setup{setup_name}/eval_dist/simulations_setup{setup_name}_{str(n+1)}_results.csv",
            dataset_name="alaa_synthetic",
            alpha=0.1,
            verbose=True,
            store_p_values=True,
        )
        # save the data
        # check if the directory exists
        if not os.path.exists(f"../results/outputs/alaa/setup{setup_name}/data"):
            os.makedirs(f"../results/outputs/alaa/setup{setup_name}/data")
        df_train.to_csv(
            f"../results/outputs/alaa/setup{setup_name}/data/simulations_setup{setup_name}_{str(n+1)}_train.csv",
            index=False,
        )
        df_test.to_csv(
            f"../results/outputs/alaa/setup{setup_name}/data/simulations_setup{setup_name}_{str(n+1)}_test.csv",
            index=False,
        )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print(
            "Usage: python alaa_synthetic.py [A|B] or python alaa_synthetic.py [A|B] [sim_number]"
        )
        sys.exit(1)
