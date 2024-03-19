## 🧅🍄🥯 PiML-Introduction 🥯🍄🧅
- PiML (read -ML, /`pai·`em·`el/) is an integrated and open-access Python toolbox for interpretable machine learning model development and model diagnostics.
- It is designed with machine learning workflows in both low-code and high-code modes, including data pipeline, model training, model interpretation and explanation, and model diagnostics and comparison.
- The toolbox supports a growing list of interpretable models (e.g., GAM, GAMI-Net, XGB2) with inherent local and/or global interpretability.
- It also supports model-agnostic explainability tools (e.g., PFI, PDP, LIME, SHAP)
- A powerful suite of model-agnostic diagnostics (e.g., weakness, uncertainty, robustness, fairness).
- Integration of PiML models and tests to existing MLOps platforms for quality assurance are enabled by flexible high-code APIs.
- Furthermore, PiML toolbox comes with a comprehensive user guide and hands-on examples, including the applications for model development and validation in banking. 

## 🧅🍄🥯 Toolbox Introduction 🥯🍄🧅
- Supervised machine learning has being increasingly used in domains where decision making can have significant consequences. However, the lack of interpretability of many machine learning models makes it difficult to understand and trust the model-based decisions. This leads to growing interest in interpretable machine learning and model diagnostics. There emerge algorithms and packages for model-agnostic explainability, including the inspection module (including permutation feature importance, partial dependence) in scikit-learn [Pedregosa2011] and various others, e.g., [Kokhlikyan2020], [Klaise2021], [Baniecki2021], [Li2022].

- Post-hoc explainability tools are useful for black-box models, but they are known to have general pitfalls [Rudin2019], [Molnar2020]. Inherently interpretable models are suggested for machine learning model development [Yang2021a], [Yang2021b], [Sudjianto2020]. The InterpretML package [Nori2013] by Microsoft Research is such a package of promoting the use of inherently interpretable models, in particular their explainable boosting machine (EBM) based on the GA2M model [Lou2013]. One may also refer to [Sudjianto2021] for discussion about how to design inherently interpretable machine learning models.

- In the meantime, model diagnostic tools become increasingly important for model validation and outcome testing. New tools and platforms are developed for model weakness detection and error analysis, e.g., [Chung2019], PyCaret package, TensorFlow model analysis, FINRA’s model validation toolkit, and Microsoft’s responsible AI toolbox. They can be used for arbitrary pre-trained models, in the same way as the post-hoc explainability tools. Such type of model diagnostics or validation is sometimes referred to as black-box testing, and there is an increasing demand of diagnostic tests for quality assurance of machine learning models.

- It is our goal to design an integrated Python toolbox for interpretable machine learning, for both model development and model diagnostics. This is particularly needed for model risk management in banking, where it is a routine exercise to run model validation including evaluation of model conceptual soundness and outcome testing from various angles. An inherently interpretable machine learning model tends to be more conceptually sound, while it is subject to model diagnostics in terms of accuracy, weakness detection, fairness, uncertainty, robustness and resilience. The PiML toolbox we develop is such a unique Python tool that supports not only a growing list of interpretable models, but also an enhanced suite of multiple diagnostic tests. It has been adopted by multiple banks since its first launch on May 4, 2022.

## 🧅🍄🥯 Toolbox Design 🥯🍄🧅
PiML toolbox is designed to support machine learning workflows by both low-code interface and high-code APIs; see Figure below for the overall design.

![image](https://github.com/diantyapitaloka/PiML-Introduction/assets/147487436/db6b69e4-554a-42dd-a905-662a37590b50)

- Low-code panels: interactive widgets or dashboards are developed for Jupyter notebook or Jupyter lab users. A minimum level of Python coding is required. The data pipeline consists of convenient exp.data_load(), exp.data_summary(), exp.eda(), exp.data_quality(), exp.feature_select(), exp.data_prepare(), each calling an interactive panel with choices of parameterization and actions.

- High-code APIs: each low-code panel can be also executed through one or more Python functions with manually specified options and parameters. Such high-code APIs are flexible to be called both in Jupyter notebook cells and by Terminal command lines. High-code APIs usually provide more options than their default use in low-code panels. End-to-end pipeline automation can be enabled with appropriate high-code settings.

- Existing models: a pre-trained model can be loaded to PiML experimentation through pipeline registration. It is mandatory to include both training and testing datasets, in order for the model to take the full advantage of PiML explanation and diagnostic capabilities. It can be an arbitrary model in supervised learning settings, including regression and binary classification.

## 🧅🍄🥯 PiML Trained 🥯🍄🧅
For PiML-trained models by either low-code interface or high-code APIs, there are four follow-up actions to be executed:

- model_interpret(): this unified API works only for inherently interpretable models (a.k.a., glass models) to be discussed in section_3. It provides model-specific interpretation in both global and local ways. For example, a linear model is interpreted locally through model coefficients or marginal effects, while a decision tree is interpreted locally through the tree path.

- model_explain(): this unified API works for arbitrary models including black-box models and glass-box models. It provides post-hoc global explainability through permutation feature importance (PFI) and partial dependence plot (PDP) through sklearn.inspect module, accumulated local effects [Apley2016], and post-hoc local explainability through LIME [Ribeiro2016] and SHAP [Lundberg2017].

- model_diagnose(): this unified API works for arbitrary models and performs model diagnostics to be discussed in section_4. It is designed to cover standardized general-purpose tests based on model data and predictions, i.e., model-agnostic tests. There is no need to access the model internals.

- model_compare(): this unified API is to compare two or three models at the same time, in terms of model performance and other diagnostic aspects. By inspecting the dashboard of graphical plots, one can easily rank models under comparison.

For registered models that are not trained from PiML, they are automatically treated as black-box models, even though such a model may be inherently interpretable (e.g., linear regression model). This is due to simplification of pipeline registration, where only the model prediction method is considered. For these models, model_interpret() is not valid, while the other three unified APIs are fully functional.

Regarding PiML high-code APIs, it is worthwhile to mention that these APIs are flexible enough for integration into existing MLOps platforms. After PiML installation to MLOps backend, the high-code APIs can be called not only to train interpretable models, but also to perform arbitrary model testing for quality assurance.

## 🧅🍄🥯 Interpretable Models 🥯🍄🧅
PiML supports a growing list of inherently interpretable models. For simplicity, we only list the models and the references. The following list of interpretable models are included PiML toolbox V0.5 (latest update: May 4, 2023).

PiML supports a growing list of inherently interpretable models. For simplicity, we only list the models and the references. The following list of interpretable models are included PiML toolbox V0.5 (latest update: May 4, 2023).

- GLM: Linear/logistic regression with 
 and/or 
 regularization [Hastie2015]

- GAM: Generalized additive models using B-splines [Serven2018]

- Tree: Decision tree for classification and regression [Pedregosa2011]

- FIGS: Fast interpretable greedy-tree sums [Tan2022]

- XGB1: Extreme gradient boosted trees of depth 1, using optimal binning [Chen2015], [Guillermo2020]

- XGB2: Extreme gradient boosted trees of depth 2, with purified effects [Chen2015], [Lengerich2020]

- EBM: Explainable boosting machine [Lou2013], [Nori2013]

- GAMI-Net: Generalized additive model with structured interactions [Yang2021b]

- ReLU-DNN: Deep ReLU networks using Aletheia unwrapper and sparsification [Sudjianto2020]

## 🧅🍄🥯 Diagnostic Suite 🥯🍄🧅
PiML comes with a continuously enhanced suite of diagnostic tests for arbitrary supervised machine learning models under regression and binary classification settings. Below is a list of the supported general-purpose tests with brief descriptions.

Accuracy: popular metrics like MSE, MAE for regression tasks and ACC, AUC, Recall, Precision, F1-score for binary classification tasks.

WeakSpot: identification of weak regions with high magnitude of residuals by 1D and 2D slicing techniques.

Overfit/Underfit: identification of overfitting/underfitting regions according to train-test performance gap, also by 1D and 2D slicing techniques.

Reliability: quantification of prediction uncertainty by split conformal prediction and slicing techniques.

Robustness: evaluation of performance degradation under different sizes of covariate noise perturbation [Cui2023].

Resilience: evaluation of performance degradation under different out-of-distribution scenarios.

Fairness: disparity test, segmented analysis and model de-bias through binning and thresholding techniques.

## 🧅🍄🥯 Future Plan 🥯🍄🧅
PiML toolbox is our new initiative of integrating state-of-the-art methods in interpretable machine learning and model diagnostics. It provides convenient user interfaces and flexible APIs for easy use of model interpretation, explanation, testing and comparison. Our future plan is to continuously improve the user experience, add new interpretable models, and expand the diagnostic suite. It is also our plan to enhance PiML experimentation with tracking and reporting.


## 🧅🍄🥯 Copyright 🥯🍄🧅
By Diantya Pitaloka

## 🧅🍄🥯 References 🥯🍄🧅
[Pedregosa2011] (1,2)
Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay (2011). Scikit-learn: Machine learning in Python, Journal of machine Learning research, 12, 2825-2830.

[Kokhlikyan2020]
Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds,Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, Orion Reblitz-Richardson (2020). Captum: A unified and generic model interpretability library for pytorch, arXiv preprint arXiv:2009.07896.

[Klaise2021]
Janis Klaise, Arnaud Van Looveren, Giovanni Vacanti, Alexandru Coca (2021). Alibi explain: Algorithms for explaining machine learning models, Journal of Machine Learning Research, 22(1), 8194-8200.

[Baniecki2021]
Hubert Baniecki, Wojciech Kretowicz, Piotr Piatyszek, Jakub Wisniewski, Przemyslaw Biecek (2021). Dalex: responsible machine learning with interactive explainability and fairness in python, Journal of Machine Learning Research, 22(1), 9759-9765.

[Li2022]
Xuhong Li, Haoyi Xiong, Xingjian Li, Xuanyu Wu, Zeyu Chen, Dejing Dou (2022). InterpretDL: Explaining Deep Models in PaddlePaddle, Journal of Machine Learning Research, 23(197), 1-6.

[Rudin2019]
Cynthia Rudin (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead, Nature Machine ntelligence, 1(5), 206-215.

[Molnar2020]
Christoph Molnar, Gunnar König, Julia Herbinger, Timo Freiesleben, Susanne Dandl, Christian A. Scholbeck, Giuseppe Casalicchio, Moritz Grosse-Wentrup, Bernd Bischl (2022, April). General pitfalls of model-agnostic interpretation methods for machine learning models, In xxAI-Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers (pp. 39-68). Cham: Springer International Publishing.

[Nori2013] (1,2)
Harsha Nori, Samuel Jenkins, Paul Koch, Rich Caruana (2019). InterpretML A Unified Framework for Machine Learning Interpretability, arXiv preprint arXiv:1909.09223.

[Lou2013] (1,2)
Yin Lou, Rich Caruana, Johannes Gehrke, Giles Hooker (2013). Accurate intelligible models with pairwise interactions, In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 623-631).

[Sudjianto2021]
Agus Sudjianto, Aijun Zhang (2021). Designing Inherently Interpretable Machine Learning Models, arXiv preprint arXiv:2111.01743.

[Chung2019]
Yeounoh Chung, Tim Kraska, Neoklis Polyzotis, Ki Hyun Tae, Steven Euijong Whang (2019, April). Slice finder: Automated data slicing for model validation, In 2019 IEEE 35th International Conference on Data Engineering (ICDE) (pp. 1550-1553). IEEE.

[Apley2016]
Daniel W. Apley, Jingyu Zhu (2016). Visualizing the effects of predictor variables in black box supervised learning models, Journal of the Royal Statistical Society Series B: Statistical Methodology 82.4 (2020): 1059-1086.

[Ribeiro2016]
Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016). Why should i trust you?” Explaining the predictions of any classifier, Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining.

[Lundberg2017]
Scott Lundberg, Su-In Lee (2017). A unified approach to interpreting model predictions, Advances in neural information processing systems 30.

[Hastie2015]
Trevor Hastie, Robert Tibshirani, Martin Wainwright (2015). Statistical learning with sparsity: the lasso and generalizations, CRC press.

[Friedman2001]
Jerome H. Friedman (2001). Greedy function approximation: a gradient boosting machine, The Annals of Statistics. 29(5): 1189-1232.

[Serven2018]
Daniel Servén, Charlie Brummitt (2018). pyGAM: Generalized Additive Models in Python, Zenodo. DOI: 10.5281/zenodo.1208723

[Tan2022]
Yan Shuo Tan, Chandan Singh, Keyan Nasseri, Abhineet Agarwal, Bin Yu (2022). Fast interpretable greedy-tree sums (FIGS), arXiv preprint arXiv:2201.11931.

[Lengerich2020]
Benjamin Lengerich, Sarah Tan, Chun-Hao Chang, Giles Hooker, Rich Caruana (2020, June). Purifying interaction effects with the functional anova: An efficient algorithm for recovering identifiable additive models, In International Conference on Artificial Intelligence and Statistics (pp. 2402-2412). PMLR.

[Chen2015] (1,2)
Tianqi Chen, Tong He (2015). Xgboost: extreme gradient boosting, R package version 0.4-2, 1(4), 1-4.

[Sudjianto2020] (1,2)
Agus Sudjianto, William Knauth, Rahul Singh, Zebin Yang, Aijun Zhang. (2020) Unwrapping the black box of deep ReLU networks: interpretability, diagnostics, and simplification, arXiv preprint arXiv:2011.04041.

[Cui2023]
Shijie Cui, Agus Sudjianto, Aijun Zhang, Runze Li (2023). Enhancing Robustness of Gradient-Boosted Decision Trees through One-Hot Encoding and Regularization, arXiv preprint arXiv:2304.13761.

[Yang2021a]
Zebin Yang, Aijun Zhang, Agus Sudjianto (2021). Enhancing explainability of neural networks through architecture constraints, IEEE Transactions on Neural Networks and Learning Systems, 32(6), 2610-2621.

[Yang2021b] (1,2)
Zebin Yang, Aijun Zhang, Agus Sudjianto (2021). GAMI-Net: An explainable neural network based on generalized additive models with structured interactions, Pattern Recognition, 120, 108192.

[Guillermo2020]
Navas-Palencia, Guillermo (2020). Optimal binning: mathematical programming formulation., arXiv preprint arXiv:2001.08025.
