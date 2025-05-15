import os
import sys

from evaluation import evaluate_distribution
from tqdm import tqdm

sys.path.append("..")

import pandas as pd

from src.datasets.semi_synthetic import get_edu

NUM_COV = 32
NUM_EXPS = 100


def main(sim_number=None):
    range_start = 0 if sim_number is None else sim_number - 1
    range_end = NUM_EXPS if sim_number is None else sim_number
    for n in tqdm(range(range_start, range_end)):
        df_train, df_test = get_edu(data_dir="../", seed=n + 1)
        # save the data
        # check if the directory exists
        if not os.path.exists(f"../results/outputs/edu/data"):
            os.makedirs(f"../results/outputs/edu/data")
        df_train.to_csv(
            f"../results/outputs/edu/data/simulations_edu_{n+1}_train.csv",
            index=False,
        )
        df_test.to_csv(
            f"../results/outputs/edu/data/simulations_edu_{n+1}_test.csv",
            index=False,
        )
        evaluate_distribution(
            df_train=df_train,
            df_test=df_test,
            num_cov=NUM_COV,
            output_path=f"../results/outputs/edu/eval_dist/simulations_edu_{n+1}_results.csv",
            dataset_name="edu",
            alpha=0.1,
            verbose=True,
            store_p_values=True,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Usage: python edu.py or python edu.py [sim_number]")
        sys.exit(1)
