from datetime import datetime
from pathlib import Path
import argparse
import sys
from dataclasses import dataclass
from hydra.utils import instantiate
import yaml
from prometheus_api_client import PrometheusConnect

try:
    from prometheus_api_client import PrometheusConnect
except ImportError:
    ImportError("Please install prometheus_api_client")
    sys.exit(1)
from .compile import load_results, save_results


@dataclass
class PromQuery:
    prom_host = "34.147.65.220"
    prom_port = "9090"
    prom_address = None
    warmup = 0
    cooldown = 0
    step = 1
    total = 0
    query = ""
    start = 0
    end = 0
    service = ""
    namespace = ""
    input_file = ""
    output_file = ""
    device_power_dict = {}
    device_id = "device_id"
    start_time_string = "_start_time"
    end_time_string = "_end_time"
    power_string = "_power"

    def query_prometheus(self):
        """
            This function collects data in prometheus for a given query, in a given time interval, with a given
            step.
        :return:
        """
        if self.prom_address is None:
            prom_address = "http://" + self.prom_host + ":" + self.prom_port
        else:
            prom_address = self.prom_address
        is_https = prom_address.startswith("https")
        should_disable = not is_https
        prom = PrometheusConnect(
            url=prom_address,
            disable_ssl=should_disable,
        )
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
            + self.calculate_minutes()
            + "])))"
        )

    def calculate_minutes(self):
        self.total = self.end - self.start
        print("total_time:", self.total)
        if abs(self.total) < 60:
            return "1m"
        return str(int(self.total / 60)) + "m"

    def load(self):
        result_file = Path(self.input_file).name
        result_folder = Path(self.input_file).parent
        data = load_results(results_file=result_file, results_folder=result_folder)
        return data

    def run_query(self, data):
        data = self.load()
        start_times = [col for col in data.columns if self.start_time_string in col]
        end_times = [col for col in data.columns if self.end_time_string in col]
        new_columns = [
            col.replace(self.start_time_string, self.power_string)
            for col in start_times
        ]
        for new_column in new_columns:
            data[new_column] = 0
        for index, _ in data.iterrows():
            for start_time in start_times:
                self.start = data[start_time]
                self.end = data[end_times[start_times.index(start_time)]]
                self.get_power()
                consumed_power = self.query_prometheus()
                data.at[index, new_columns[start_times.index(start_time)]] = (
                    consumed_power
                )
        for device in self.device_power_dict.keys():
            data.loc[data[self.device_id] == device, "peak_power"] = (
                self.device_power_dict[device]
            )
        return data

    def save(self, data):
        output_file = Path(self.output_file).name
        output_folder = Path(self.output_file).parent
        save_results(data, results_file=output_file, results_folder=output_folder)

    def __call__(self):
        data = self.run_query()
        self.save(data)


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--verbosity", type=str, default="INFO")
parser.add_argument("--prometheus_config", type=str, default=None)


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    # Read the prometheus config from yaml
    with Path(args.prometheus_config).open("r") as stream:
        prometheus_config = yaml.safe_load(stream)
    if prometheus_config is None:
        promObj = PromQuery(input_file=input_file, output_file=output_file)
    else:
        prometheus_config["_target_"] = "deckard.layers.compile.PromQuery"
        prometheus_config["input_file"] = (
            input_file if input_file is not None else prometheus_config["input_file"]
        )
        prometheus_config["output_file"] = (
            output_file if output_file is not None else prometheus_config["output_file"]
        )
        promObj = instantiate(prometheus_config)
    promObj()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
