import watchdog.events
import watchdog.observers
import time, json
from pathlib import Path
import pandas as pd
sub_reports = []
TOTAL = len(pd.read_csv("original.csv"))
QUEUE = "queue.csv"
import logging
logger = logging.getLogger(__name__)
class HTMLHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=[f'**/*predictions.json'], ignore_directories=True, case_sensitive=False)
        # Set the directory on watch
        
    def on_closed(self, event):
        
        sub_reports.append(event.src_path)
        print("Watchdog received closed event - % s." % event.src_path)
        self.update_index(Path(event.src_path).parent.parent)
        # check_queue()
        progress = self.calculate_progress(TOTAL, QUEUE)
        print(f'{progress["achieved"]*100:.2f} percent of experiments completed!'.format(progress["achieved"] * 100))
        # Event is created, you can process it now1

    def parse_params(self, param_paths):
        param_df = pd.DataFrame()
        for i in range(len(param_paths)):
            param = param_paths[i]
            id_ = param.parent.name
            try:
                with open(param, "r") as f:
                    params = json.load(f)
            except:
                continue
            df2 = pd.DataFrame(params, index = [id_])
            param_df = pd.concat([param_df, df2])
        return param_df

    def parse_metrics(self, metr_paths, columns = ["f1","accuracy", "precision","recall","auc","scale","epochs","learning_rate","loss","input_noise", "output_noise", "type"]):
        df = pd.DataFrame(columns = columns)
        for i in range(len(metr_paths)):
            metr = metr_paths[i]
            id_ = metr.parent.name
            try:
                with open(metr, "r") as f:
                    metr = json.load(f)
            except:
                continue
            df2 = pd.Series(metr, index = columns, name = id_)
            df = df.append(df2)
        return df

    def merge_all_results(self, directory = "reports", met_file = "**/metrics.json", html_file = "**/report.html", pred_file = "**/predictions.json", param_file = "**/params.json"):
        metr_paths = list(Path(directory).rglob(met_file))
        html_paths = list(Path(directory).rglob(html_file))
        pred_paths = list(Path(directory).rglob(pred_file))
        param_paths = list(Path(directory).rglob(param_file))
        params = self.parse_params(param_paths)
        metrics = self.parse_metrics(metr_paths)
        p_cols = set(params.columns)
        m_cols = set(metrics.columns)
        samesies = list(p_cols.intersection(m_cols))
        metrics.drop(samesies, axis = 1, inplace = True)
        big = params.merge(metrics, left_index = True, right_index = True, how = 'outer')
        big['link'] = [Path(big.index[i] , big.index[i] , "report.html") for i in range(len(big))]
        print(big['link'])
        big['link'] = big['link'].apply(lambda x: '<a href="{}">{}</a>'.format(x, "Summary"))
        big.index = big['link']
        del big['link']
        big['predictions'] = pred_paths
        big['predictions'] = big['predictions'].apply(lambda x: '<a href="{}">{}</a>'.format(x, "Predictions"))
        return big

    def calculate_progress(self, total, queue):
        progress =  (total - len(pd.read_csv(queue))) / total
        directory = Path(queue).parent
        dict_ = {"name" : "experiments" , "achieved": progress, "remaining" : 1 - progress}
        with open(Path(directory , "progress.json"), "w") as f:
            json.dump(dict_, f)
        return dict_

    def update_index(self, directory = "reports", met_file = "**/metrics.json", html_file = "**/report.html", pred_file = "**/predictions.json", param_file = "**/params.json"):
        progress = self.calculate_progress(TOTAL, QUEUE)
        df = self.merge_all_results(directory, met_file, html_file, pred_file, param_file)
        html = df.to_html(escape = False)
        df.to_csv(Path(directory , "reports.csv"))
        text_file = open(Path(directory ,"table.html"), "w")
        text_file.write(html)
        text_file.close()
        print(f'{progress["achieved"]*100:.2f} percent of experiments completed!'.format(progress["achieved"] * 100))
        return df

  
if __name__ == "__main__":
    src_path = r"reports"
    event_handler = HTMLHandler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()