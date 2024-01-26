import experiments.libs.functions
from prometheus_api_client import PrometheusConnect
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PromQuery:
    prom_host = "labumu.se"
    prom_port = "30090"
    prom_address = "http://" + prom_host + ":" + prom_port + "/"
    warmup = 9000
    warmdown = 3000
    step = 5
    query = ""
    start = 0
    end = 0
    service = ""
    namespace = ""
    percentile = ""
    reporter = "source"
    response_code = ""

    def query_prometheus(self):
        """
            This function collects data in prometheus for a given query, in a given time interval, with a given
            warmup/warmdown time offset and a given step.
        :return:
        """
        prom = PrometheusConnect(url=self.prom_address, disable_ssl=True)
        start = datetime.fromtimestamp((self.start + self.warmup) / 1000)
        end = datetime.fromtimestamp((self.end - self.warmdown) / 1000)

        result = prom.custom_query_range(
            query=self.query,
            start_time=start,
            end_time=end,
            step=self.step,
        )
        return result

    def get_response_time(self, version=None):
        """
        This function will get the response time for a given service in a given time interval and based on a given
        percentile
        :return:
        """
        if version == None:
            version = "latest"
        if self.response_code == "":
            self.query = (
                "(histogram_quantile("
                + str(self.percentile)
                + ', sum(irate(istio_request_duration_milliseconds_bucket{reporter="'
                + self.reporter
                + '", destination_service=~"'
                + self.service
                + "."
                + self.namespace
                + '.svc.cluster.local", destination_canonical_revision="'
                + version
                + '"}[1m])) '
                "by (le)) / 1000)"
            )
        elif self.response_code == "200":
            self.query = (
                "(histogram_quantile("
                + str(self.percentile)
                + ', sum(irate(istio_request_duration_milliseconds_bucket{reporter="'
                + self.reporter
                + '", destination_service=~"'
                + self.service
                + "."
                + self.namespace
                + '.svc.cluster.local",'
                'response_code="'
                + self.response_code
                + '", destination_canonical_revision="'
                + version
                + '"}[1m])) by (le)) / 1000)'
            )
        else:
            self.query = (
                "(histogram_quantile("
                + str(self.percentile)
                + ', sum(irate(istio_request_duration_milliseconds_bucket{reporter="'
                + self.reporter
                + '", destination_service=~"'
                + self.service
                + "."
                + self.namespace
                + '.svc.cluster.local",'
                'response_code!="200", destination_canonical_revision="'
                + version
                + '"}[1m])) by (le)) / 1000)'
            )

        result = self.query_prometheus()
        return result

    def get_status_codes(self, version=None):
        """
        This function will get the request status codes for agiven service, in agiven time interval with a given
        warmup/warmdown time offset and a given step

        """
        if version == None:
            version = "latest"
        self.query = (
            'round(sum(irate(istio_requests_total{destination_service=~"'
            + self.service
            + ""
            "."
            + self.namespace
            + '.svc.cluster.local", reporter="source", destination_canonical_revision="'
            + version
            + '"}[1m])) by (response_code, response_flags), 0.001)'
        )
        result = self.query_prometheus()
        return result

    def get_retried_requests(self, port, version=""):
        """
        This function gets the number of retried requests for a given service, in given time interval with a given
        warmup/warmdown time offset and a given step
        """
        self.query = (
            'round(sum(irate(envoy_cluster_upstream_rq_retry{cluster_name="outbound|'
            + str(port)
            + "|"
            + version
            + "|"
            + self.service
            + '.default.svc.cluster.local"}[1m])) by (), 0.001)'
        )
        result = self.query_prometheus()
        return result

    def get_requests_in_queue(self):
        """
        This function will get the request in the queue for a given service
        """
        self.query = (
            'round(sum(irate(envoy_http_inbound_0_0_0_0_5000_downstream_rq_active{app=~"'
            + self.service
            + '"}[1m])) by (service_istio_io_canonical_name), 0.001)'
        )
        result = self.query_prometheus()
        return result

    def get_current_queue_size(self, job="istio"):
        """
        This function will get the current queue size which is pushed in pushgateway (HTTP2MaxRequests)
        """
        self.query = 'destination_rule_http2_max_requests{exported_job="' + job + '"}'
        result = self.query_prometheus()
        return result

    def get_retry_attempt(self):
        """
        This function get the retry attempt which is pushed in pushgateway (attempts)
        """
        self.query = "retry_attempts_" + self.service
        result = self.query_prometheus()
        return result

    def __call__(self, config_file, output_file, output_folder) -> None:
        """
        This function will call the prometheus query function and write the result in a given file
        """
        # Available metrics:
        # train_time, train_start_time, train_end_time,
        # predict_proba_time, predict_proba_start_time, predict_proba_end_time,
        # adv_train_time, adv_train_start_time, adv_train_end_time,
        # adv_predict_proba_time, adv_predict_proba_start_time, adv_predict_proba_end_time,
        # Find all output_file recursively inside output_folder
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        files = Path(output_folder).rglob(output_file)
        # Each file will have train_start_time train_end_time, predict_proba_start_time predict_predict_proba_end_time, adv_
        # Query Prometheus
        # Do calulations
        # Write to file
        # Use a lambda function so that this will be parallelized across all the files in the files iterator and across each entry of the config
        # Return None
        None
