import sys

from evaluation import evaluate_distribution
from tqdm import tqdm

sys.path.append("..")
import pandas as pd

from src.datasets.semi_synthetic import get_acic_2016

NUM_COV = 58
NUM_EXPS = 10


def main(setting, sim_number=None):
    """
    Run the evaluation for the acic2016 dataset
    :param setting: int, the setting number to run
    :param sim_number: int, the simulation number to run
    """
    print(f"Running setup {setting}")
    if setting < 1 and setting > 77:
        raise ValueError("setting must be between 1 and 77 inclusive")

    range_start = 0 if sim_number is None else sim_number - 1
    range_end = NUM_EXPS if sim_number is None else sim_number
    for n in tqdm(range(range_start, range_end)):
        df_train, df_test = get_acic_2016(data_dir="../", setting=setting, sim_nb=n)
        evaluate_distribution(
            df_train=df_train,
            df_test=df_test,
            num_cov=NUM_COV,
            output_path=f"../results/outputs/acic2016/setup{setting}/eval_dist/simulations_setup{setting}_{str(n+1)}_results.csv",
            dataset_name="acic2016",
            alpha=0.1,
            verbose=True,
            store_p_values=False,
        )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Usage: python acic2016.py [setting] or python acic2016.py [setting] [sim_number]")
        sys.exit(1)
