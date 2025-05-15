import json
import logging
import os

import pytensor

pytensor.config.optimizer = "fast_compile"  # or 'None' for no optimizations
pytensor.config.exception_verbosity = "high"

import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append("..")

import warnings

from src.benchmarks.bart import BART
from src.benchmarks.cevae import CEVAE
from src.benchmarks.cmgp import CMGP
from src.benchmarks.diffpo.main_model import DiffPOITE
from src.benchmarks.dklite import DKLITE
from src.benchmarks.fccn import FCCN
from src.benchmarks.ganite.ganite import GANITE
from src.benchmarks.noflite.noflite import NOFLITE
from src.cmc_metalearners.cmc_metalearners import (
    CCT_Learner,
    CCT_Learner_FCCN,
    CCT_Learner_NOFLITE,
    CMC_S_Learner,
    CMC_T_Learner,
    CMC_X_Learner,
    OracleCPS,
)
from src.utils import suppress_output

warnings.filterwarnings("ignore")


def evaluate_distribution(
    df_train,
    df_test,
    num_cov,
    output_path,
    dataset_name,
    learner=RandomForestRegressor,
    alpha=0.1,
    store_p_values=False,
    verbose=False,
):
    logging.basicConfig(level=logging.INFO)
    W_train = df_train["W"].to_numpy().astype(int)
    y_train = df_train["Y"].to_numpy().astype(float)
    y1_train = df_train["Y1"].to_numpy().astype(float)
    y0_train = df_train["Y0"].to_numpy().astype(float)
    X_train = df_train[["X" + str(i) for i in range(1, num_cov + 1)]].to_numpy().astype(float)
    ps_train = df_train["ps"].to_numpy().astype(float)

    W_test = df_test["W"].to_numpy().astype(float)
    y_test = df_test["Y"].to_numpy().astype(float)
    y1_test = df_test["Y1"].to_numpy().astype(float)
    y0_test = df_test["Y0"].to_numpy().astype(float)
    X_test = df_test[["X" + str(i) for i in range(1, num_cov + 1)]].to_numpy().astype(float)
    ps_test = df_test["ps"].to_numpy().astype(float)

    columns = [
        "approach",
        "rmse_y0",
        "rmse_y1",
        "rmse_ite",
        "coverage_y0",
        "coverage_y1",
        "coverage_ite",
        "efficiency_y0",
        "efficiency_y1",
        "efficiency_ite",
        "crps_y0",
        "crps_y1",
        "crps_ite",
        "ll_y0",
        "ll_y1",
        "ll_ite",
        "dispersion_y0",
        "dispersion_y1",
        "dispersion_ite",
    ]
    if store_p_values:
        columns += ["p_values_y0", "p_values_y1", "p_values_ite"]
    df_eval = pd.DataFrame(columns=columns)

    # Oracle CPS
    logging.info("Fit and evaluate Oracle CPS ...")
    with suppress_output(not verbose):
        oracle_cps = OracleCPS(
            learner(n_jobs=-1), learner(n_jobs=-1), learner(n_jobs=-1), normalized_conformal=True
        )
        oracle_cps.fit(X_train, y0_train, y1_train)
        evaluate = oracle_cps.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "OracleCPS"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # pseudo CMC-T-learner
    logging.info("Fit and evaluate pseudo CMC-T-learner ...")
    with suppress_output(not verbose):
        pseudo_cmc_t_learner = CMC_T_Learner(
            learner(n_jobs=-1), learner(n_jobs=-1), normalized_conformal=True, pseudo_MC=True
        )
        pseudo_cmc_t_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = pseudo_cmc_t_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "pseudo-CMC-T-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CMC-T-learner
    logging.info("Fit and evaluate CMC-T-learner ...")
    with suppress_output(not verbose):
        cmc_t_learner = CMC_T_Learner(
            learner(n_jobs=-1), learner(n_jobs=-1), normalized_conformal=True, pseudo_MC=False
        )
        cmc_t_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = cmc_t_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CMC-T-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # pseudo CMC-S-learner
    logging.info("Fit and evaluate pseudo CMC-S-learner ...")
    with suppress_output(not verbose):
        pseudo_cmc_s_learner = CMC_S_Learner(
            learner(n_jobs=-1), normalized_conformal=True, pseudo_MC=True
        )
        pseudo_cmc_s_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = pseudo_cmc_s_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "pseudo-CMC-S-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CMC-S-learner
    logging.info("Fit and evaluate CMC-S-learner ...")
    with suppress_output(not verbose):
        cmc_s_learner = CMC_S_Learner(
            learner(n_jobs=-1), normalized_conformal=True, pseudo_MC=False
        )
        cmc_s_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = cmc_s_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CMC-S-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # pseudo CMC-X-learner
    logging.info("Fit and evaluate pseudo CMC-X-learner ...")
    with suppress_output(not verbose):
        pseudo_cmc_x_learner = CMC_X_Learner(
            learner(n_jobs=-1),
            learner(n_jobs=-1),
            learner(n_jobs=-1),
            normalized_conformal=True,
            pseudo_MC=True,
        )
        pseudo_cmc_x_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = pseudo_cmc_x_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "pseudo-CMC-X-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CMC-X-learner
    logging.info("Fit and evaluate CMC-X-learner ...")
    with suppress_output(not verbose):
        cmc_x_learner = CMC_X_Learner(
            learner(n_jobs=-1),
            learner(n_jobs=-1),
            learner(n_jobs=-1),
            normalized_conformal=True,
            pseudo_MC=False,
        )
        cmc_x_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = cmc_x_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CMC-X-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CCT-learner
    logging.info("Fit and evaluate CCT-learner ...")
    with suppress_output(not verbose):
        conformal_CT_learner = CCT_Learner(
            learner(n_jobs=-1), learner(n_jobs=-1), normalized_conformal=True
        )
        conformal_CT_learner.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = conformal_CT_learner.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CTT-learner"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # # BART
    logging.info("Fit and evaluate BART ...")
    with suppress_output(not verbose):
        bart = BART()
        bart.fit(X_train, y_train, W_train)
        evaluate = bart.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "BART"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CMGP
    logging.info("Fit and evaluate CMGP ...")
    with suppress_output(not verbose):
        cmgp = CMGP(X=X_train, Treatments=W_train, Y=y_train)
        evaluate = cmgp.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CMGP"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CEVAE
    logging.info("Fit and evaluate CEVAE ...")
    dim_bin = 0
    dim_cont = X_train.shape[1]
    if dataset_name == "ihdp":
        batch_size = 100
        iters = 7000
        n_h = 64
    elif dataset_name in ["edu", "acic2016", "nlsm"]:
        batch_size = 256
        iters = 18000
        n_h = 64
    else:
        batch_size = 100
        iters = 7000
        n_h = 64
    with suppress_output(not verbose):
        cevae = CEVAE(
            dim_bin=dim_bin, dim_cont=dim_cont, batch_size=batch_size, iters=iters, n_h=n_h
        )
        cevae.fit(X=X_train, Y=y_train, W=W_train)
        evaluate = cevae.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CEVAE"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # FCCN
    logging.info("Fit and evaluate FCCN ...")
    if dataset_name == "ihdp":
        alpha_fccn = 5e-4
        beta = 1e-5
        iters = 20000
    elif dataset_name in ["edu", "acic2016", "nlsm"]:
        alpha_fccn = 5e-4
        beta = 1e-4
        iters = 50000
    else:
        alpha_fccn = 5e-4
        beta = 1e-5
        iters = 20000
    with suppress_output(not verbose):
        fccn = FCCN(input_size=X_train.shape[1], alpha=alpha_fccn, beta=beta)
        fccn.train(X_train, y_train, W_train, iters=iters)
        evaluate = fccn.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "FCCN"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CCT-learner-FCCN
    logging.info("Fit and evaluate CCT-learner-FCCN ...")
    with suppress_output(not verbose):
        conformal_CT_learner_fccn = CCT_Learner_FCCN(
            input_size=X_train.shape[1], alpha=alpha_fccn, beta=beta, iters=iters
        )
        conformal_CT_learner_fccn.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = conformal_CT_learner_fccn.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CTT-learner-FCCN"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # # GANITE
    logging.info("Fit and evaluate GANITE ...")
    if dataset_name == "ihdp":
        ganite_params = {
            "h_dim": 30,  # hidden dimensions
            "batch_size": 64,  # the number of samples in each batch
            "iterations": 10000,  # the number of iterations for training
            "alpha": 2.0,
            "beta": 5.0,  # hyper-parameter to adjust the loss importance
            "input_size": X_train.shape[1],
            # the number of features
        }
    elif dataset_name in ["edu", "acic2016", "nlsm"]:
        ganite_params = {
            "h_dim": 64,  # hidden dimensions
            "batch_size": 128,  # the number of samples in each batch
            "iterations": 15000,  # the number of iterations for training
            "alpha": 2.0,
            "beta": 1e-3,  # hyper-parameter to adjust the loss importance
            "input_size": X_train.shape[1],
        }
    else:
        ganite_params = {
            "h_dim": 8,  # hidden dimensions
            "batch_size": 64,  # the number of samples in each batch
            "iterations": 10000,  # the number of iterations for training
            "alpha": 2,
            "beta": 5,  # hyper-parameter to adjust the loss importance
            "input_size": X_train.shape[1],
        }
    with suppress_output(not verbose):
        ganite = GANITE(**ganite_params)
        ganite.fit(X_train, y_train, W_train)
        evaluate = ganite.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "GANITE"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # DKLITE
    logging.info("Fit and evaluate DKLITE ...")
    with suppress_output(not verbose):
        dklite = DKLITE(input_dim=X_train.shape[1], output_dim=1)
        dklite.fit(X_train, y_train, W_train)
        evaluate = dklite.evaluate(
            X_test, y0_test, y1_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "DKLITE"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # diffpo
    logging.info("Fit and evaluate diffpo ...")
    config = {
        "train": {"epochs": 500, "batch_size": 256, "lr": 0.001, "valid_epoch_interval": 1000},
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "f_dim": 180,
            "cond_dim": X_train.shape[1] + 1,  # conditional variable dimension
            "hidden_dim": 128,
            "side_dim": 33,
            "nheads": 2,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 100,
            "schedule": "quad",
            "mixed": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 32,
            "featureemb": 32,
            "target_strategy": "random",
            "mixed": False,
            "test_missing_ratio": 0.0,
        },
    }
    with suppress_output(not verbose):
        diffpo = DiffPOITE(config=config)
        diffpo.fit(X=X_train, Y0=y0_train, Y1=y1_train, W=W_train, ps=ps_train)
        evaluate = diffpo.evaluate(
            X_test,
            y0_test,
            y1_test,
            W=W_test,
            ps=ps_test,
            alpha=alpha,
            return_p_values=store_p_values,
        )
        evaluate["approach"] = "DiffPO"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # NOFLITE
    logging.info("Fit and evaluate NOFLITE ...")
    if dataset_name in ["edu", "acic2016", "nlsm"]:
        params = {
            "input_size": X_train.shape[1],
            "lr": 5e-4,
            "lambda_l1": 0,
            "lambda_l2": 1e-3,
            "batch_size": 256,
            "noise_reg_x": 0,
            "noise_reg_y": 1e-1,
            "hidden_neurons_encoder": 16,
            "hidden_layers_balancer": 2,
            "hidden_layers_encoder": 0,
            "hidden_layers_prior": 2,
            "hidden_neurons_trans": 4,
            "hidden_neurons_cond": 16,
            "hidden_layers_cond": 2,
            "dense": False,
            "n_flows": 4,
            "datapoint_num": 8,
            "resid_layers": 1,
            "max_steps": 5000,
            "flow_type": "SigmoidX",
            "metalearner": "T",
            "lambda_mmd": 0.01,
            "n_samples": 500,
            "trunc_prob": 0.01,
            "bin_outcome": False,
            "iterations": 1,
            "visualize": False,
        }
    else:
        params = {
            "input_size": X_train.shape[1],
            "lr": 5e-4,
            "lambda_l1": 1e-3,
            "lambda_l2": 5e-3,
            "batch_size": 128,
            "noise_reg_x": 0,
            "noise_reg_y": 5e-1,
            "hidden_neurons_encoder": 8,
            "hidden_layers_balancer": 2,
            "hidden_layers_encoder": 3,
            "hidden_layers_prior": 2,
            "hidden_neurons_trans": 4,
            "hidden_neurons_cond": 16,
            "hidden_layers_cond": 2,
            "dense": False,
            "n_flows": 1,
            "datapoint_num": 8,
            "resid_layers": 1,
            "max_steps": 5000,
            "flow_type": "SigmoidX",
            "metalearner": "T",
            "lambda_mmd": 1,
            "n_samples": 500,
            "trunc_prob": 0.01,
            "bin_outcome": False,
            "iterations": 1,
        }
    with suppress_output(not verbose):
        noflite = NOFLITE(params=params)
        noflite.fit(X_train, y_train, W_train)
        evaluate = noflite.evaluate(
            X_test, y0_test, y1_test, W=W_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "NOFLITE"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # CCT-learner-NOFLITE
    logging.info("Fit and evaluate CCT-learner-NOFLITE ...")
    with suppress_output(not verbose):
        conformal_CT_learner_noflite = CCT_Learner_NOFLITE(params=params)
        conformal_CT_learner_noflite.fit(X_train, y_train, W_train, p=ps_train)
        evaluate = conformal_CT_learner_noflite.evaluate(
            X_test, y0_test, y1_test, ps_test, alpha=alpha, return_p_values=store_p_values
        )
        evaluate["approach"] = "CTT-learner-NOFLITE"
        df_eval = pd.concat([df_eval, pd.DataFrame(evaluate, index=[0])], ignore_index=True)

    # logging.info results
    if verbose:
        logging.info(df_eval)

    if store_p_values:
        df_eval["p_values_y0"] = df_eval["p_values_y0"].apply(lambda x: x.tolist())
        df_eval["p_values_y1"] = df_eval["p_values_y1"].apply(lambda x: x.tolist())
        df_eval["p_values_ite"] = df_eval["p_values_ite"].apply(lambda x: x.tolist())
    # save results
    # check if output path dir exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if output_path.endswith(".parquet"):
        df_eval.to_parquet(output_path)
    else:
        df_eval.to_csv(output_path)
        # check if df_eval is saved
    if os.path.exists(output_path):
        print(f"Saved results to {output_path}")
    else:
        print("Failed to save results.")
