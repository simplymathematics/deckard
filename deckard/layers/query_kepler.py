from datetime import datetime
import argparse
import logging
import sys
from dataclasses import dataclass
import yaml

try:
    from prometheus_api_client import PrometheusConnect
except ImportError:
    ImportError("Please install prometheus_api_client")
    sys.exit(1)


v100 = 250 / 3600
p100 = 250 / 3600
l4 = 72 / 3600

@dataclass
class PromQuery:
    def __init__(self):
        self.prom_host = "34.147.65.220"
        self.prom_port = "9090"
        self.prom_address = "http://" + self.prom_host + ":" + self.prom_port + "/"
        self.warmup = 0
        self.cooldown = 0
        self.step = 1
        self.total = 0
        self.query = ""
        self.start = 0
        self.end = 0
        self.service = ""
        self.namespace = ""

    def query_prometheus(self):
        """
            This function collects data in prometheus for a given query, in a given time interval, with a given
            step.
        :return:
        """
        prom = PrometheusConnect(url=self.prom_address, disable_ssl=True)
        start = datetime.fromtimestamp((self.start + self.warmup))
        end = datetime.fromtimestamp((self.end - self.cooldown))
        result = prom.custom_query_range(
            query=self.query,
            start_time=start,
            end_time=end,
            step=self.step,
        )
        return abs(
            float(result[0]["values"][-1 * int(self.total)][1])
            - float(result[0]["values"][-1][1]),
        )

    def get_power(self):
        self.query = (
            "sum(increase((kepler_container_joules_total["
            + self.caluculate_minutes()
            + "])))"
        )

    def caluculate_minutes(self):
        self.total = self.end - self.start
        print("total_time:", self.total)
        if abs(self.total) < 60:
            return "1m"
        return str(int(self.total / 60)) + "m"


def run_query(input_file, output_file):
    new_columns = [
        "train_power",
        "predict_power",
        "predict_proba_power",
        "predict_log_proba_power",
        "adv_fit_power",
        "adv_predict_power",
    ]
    start_times = [
        "train_start_time",
        "predict_start_time",
        "predict_proba_start_time",
        "predict_log_proba_start_time",
        "adv_fit_start_time",
        "adv_predict_start_time",
    ]
    end_times = [
        "train_end_time",
        "predict_end_time",
        "predict_proba_end_time",
        "predict_log_proba_end_time",
        "adv_fit_end_time",
        "adv_predict_end_time",
    ]

    promObj = PromQuery()
    data = pd.read_csv(input_file, index_col=0)
    for new_column in new_columns:
        data[new_column] = 0
        data["peak_power"] = 0
    for index, row in data.iterrows():
        for start_time in start_times:
            promObj.start = data[start_time]
            promObj.end = data[end_times[start_times.index(start_time)]]
            promObj.get_power()
            consumed_power = promObj.query_prometheus()
            peak_power = 0
            if "v100" in row["device_id"]:
                peak_power = 250
            elif "p100" in row["device_id"]:
                peak_power = 250
            elif "l4" in row["device_id"]:
                peak_power = 72
            data.at[index, new_columns[start_times.index(start_time)]] = consumed_power
            data.at[index, "peak_power"] = peak_power
    data.to_csv(output_file)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    dvc_parser = argparse.ArgumentParser()
    dvc_parser.add_argument("--input_file", type=str, default=None)
    dvc_parser.add_argument("--output_file", type=str, default=None)
    dvc_parser.add_argument("--verbosity", type=str, default="INFO")

    args = dvc_parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Quering the Prometheus for power metrics")

    results = run_query(input_file=input_file, output_file=output_file)
