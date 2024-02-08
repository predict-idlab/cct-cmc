# CMC-learner

> The *CMC* or Conformal Monte Carlo framework leverages conformal predictive systems, Monte Carlo sampling, and meta-learners to provide a non-parametric predictive distribution estimate for Conditional Average Treatment Effect models and thereby come closer to Individual Treatment Effect estimation. The preprint version of our paper is accessible for review and feedback. To access the preprint, please visit [here](https://arxiv.org/abs/2402.04906). 

## Usage 🛠
The following code shows how the CMC Framework can be used. The example uses a T-learner trained using Random Forest Regressors from sklearn, however, a X- or S-learner is also supported using Conformal_MC_X_Learner and Conformal_MC_S_Learner respectively. 

```py
from src.mc_conformal_metalearner.mc_conformal_metalearners import Conformal_MC_T_Learner
from sklearn.ensemble import RandomForestRegressor


X, y, T = ...  # your treatment effect dataset (X = covariates, y = outcome, T = treatment variable)

#If you use the pseudo version
CMC_Learner = Conformal_MC_T_Learner(
            RandomForestRegressor(),
            RandomForestRegressor(),
            adaptive_conformal=adaptive_conformal,
            pseudo_MC=True,
            MC_samples=MC_samples,
)

#If you use the full Monte Carlo version
CMC_Learner = Conformal_MC_T_Learner(
            RandomForestRegressor(),
            RandomForestRegressorRandomForestRegressor(),
            adaptive_conformal=adaptive_conformal,
            pseudo_MC=False,
            MC_samples=MC_samples,
)


CMC_Learner.fit(X, y, T)  # Fit the CMC-Learner on the treatment effect dataset

#Get a symmetric interval (i.e. the same amount of coverage in the right and left parts of the interval) of the CATE estimate with coverage of 1-alpha for all samples in X
int_CMC_Learner = CMC_Learner.predict_int(X, confidence=1-alpha)

#Get the predictive distribution of the CATE estimate
cps_CMC_Learner = CMC_Learner.predict_cps(X)
```

Various Python Notebooks illustrating a complete pipeline using CMC are provided [here](notebooks/illustrations/). 

## Benchmarks ⏱

Check out our benchmark results [here](results/figures/).  

## How does it work ⁉️

The CMC framework is built using Monte Carlo sampling, conformal predictive systems, and meta-learners. First, it fits the meta-learner and calibrates the separate learners afterwards using a conformal predictive system. Using the calibrated learners, n ITE Monte Carlo samples are calculated for every training sample using: 
* $MC_{ITE} = \hat{Y}^1 - \hat{Y}^0 $
* $Pseudo-MC_{ITE} = W(Y-\hat{Y}^0) + (1-W)\hat{Y}^1  $

The full meta-learner is then calibrated again using conformal predictive systems on these sampled ITEs to provide a predictive distribution of the CATE, given X. 

## Referencing our code :memo:

If you use *CMC-Learner* in a scientific publication, we would highly appreciate citing us as:

```bibtex
@misc{jonkers2024conformal,
      title={Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects}, 
      author={Jef Jonkers and Jarne Verhaeghe and Glenn Van Wallendael and Luc Duchateau and Sofie Van Hoecke},
      year={2024},
      eprint={2402.04906},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
The preprint version of the paper can be found at [https://arxiv.org/abs/2402.04906](https://arxiv.org/abs/2402.04906). 

---

<p align="center">
👤 <i>Jef Jonkers, Jarne Verhaeghe</i>
</p>

## License
This package is available under the MIT license. More information can be found here: https://github.com/predict-idlab/cmc-learner/blob/main/LICENSE
