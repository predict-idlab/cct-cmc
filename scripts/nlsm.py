import sys

from evaluation import evaluate_distribution
from tqdm import tqdm

sys.path.append("..")
import pandas as pd

from src.datasets.semi_synthetic import get_nlsm_sim_data

NUM_COV = 11
NUM_EXPS = 100


def main(sim_number=None):
    range_start = 0 if sim_number is None else sim_number - 1
    range_end = NUM_EXPS if sim_number is None else sim_number
    for n in tqdm(range(range_start, range_end)):
        df_train, df_test = get_nlsm_sim_data(data_dir="../")
        evaluate_distribution(
            df_train=df_train,
            df_test=df_test,
            num_cov=NUM_COV,
            output_path=f"../results/outputs/nlsm/eval_dist/simulations_nlsm_{str(n+1)}_results.csv",
            dataset_name="nlsm",
            alpha=0.1,
            verbose=True,
            store_p_values=False,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Usage: python nlsm.py or python nlsm.py [sim_number]")
        sys.exit(1)
