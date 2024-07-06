from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.datasets import load_rossi
from lifelines.calibration import survival_probability_calibration
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt

rossi = load_rossi()
regression_dataset = load_rossi()


models = {
    "weibull": WeibullAFTFitter(),
    "log-normal": LogNormalAFTFitter(),
    "log-logistic": LogLogisticAFTFitter(),
    "cox": CoxPHFitter(),
}
fig, ax = plt.subplots(1, len(models), figsize=(17, 5))
i = 0
for model_name, model in models.items():
    model.fit(rossi, duration_col="week", event_col="arrest")
    survival_probability_calibration(model, regression_dataset, t0=10, ax=ax[i])
    i += 1
