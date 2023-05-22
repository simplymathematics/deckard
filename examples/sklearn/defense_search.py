# - model.art.preprocessor.name: art.defences.preprocessor.FeatureSqueezing
#   model.art.preprocessor.params:
#     clip_values :
#      - [0,255]
#     bit_depth : [4, 8, 16, 32, 64]

# - model.art.preprocessor.name: art.defences.preprocessor.GaussianAugmentation
#   model.art.preprocessor.params:
#     clip_values :
#      - [0,255]
#     sigma : [.1, .3, 1]
#     ratio : [.1, .5, 1]

# - model.art.preprocessor.name: art.defences.preprocessor.SpatialSmoothing
#   model.art.preprocessor.params:
#     clip_values :
#      - [0,255]
#     window_size : [2,3,4]

# - model.art.preprocessor.name: art.defences.preprocessor.TotalVarMin
#   model.art.preprocessor.params:
#     clip_values :
#      - [0,255]
#     prob : [.001, .01, .1]
#     norm : [1, 2, 3]
#     lamb : [.05, .5, .95]
#     max_iter : [100]

# - model.art.postprocessor.name : art.defences.postprocessor.GaussianNoise
#   model.art.postprocessor.params:
#     clip_values :
#      - [0,255]
#     scale: [.1, .9, .999]

# - model.art.postprocessor.name : art.defences.postprocessor.HighConfidence
#   model.art.postprocessor.params:
#     cutoff : [.1, .5, .9, .99]


# - model.art.postprocessor.name : art.defences.postprocessor.Rounded
#   model.art.preprocessor.params:
#     clip_values :
#      - [0,255]
#     decimals : [1, 2, 4, 8]

# from omegaconf import OmegaConf, DictConfig
# from pathlib import Path
# from optuna import create_study, Trial, TrialPruned, TrialState

# def configure(cfg: DictConfig, trial:Trial) -> None:
#     print(trial.params.keys())
#     input("Press enter to continue")
