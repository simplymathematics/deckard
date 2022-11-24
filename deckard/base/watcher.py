




class WatcherConfig(
    collections.namedtuple(
        typename="WatcherConfig",
        field_names="data, model, scorers, plots, files",
        defaults=({}, {}, {}, {}, {}),
        rename = True
    ),
    BaseHashable,
):


class JSONHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self, regex, filename, recursive = True, **kwargs):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(
            self,
            patterns=[REGEX],
            ignore_directories=True,
            case_sensitive=False,
        )
        self.regex = regex
        self.recurse = recursive
        self.filename = filename

        logger.info("Regex is {}".format(REGEX))

    def on_closed(self, event):
        logger.info("Watchdog received created event - % s." % event.src_path)
        events.append(event.src_path)
        self.filename = event.src_path
        try:
            df = self.transform_json()
            logger.info("Transformed JSON")
        except Exception as e:
            logger.warning("Could not transform json")
            logger.warning(e)
        if "TOTAL" and "QUEUE" in locals():
            try:
                self.calculate_progress(TOTAL, QUEUE)
                logger.info("Calculated progress")
            except Exception as e:
                logger.warning("Could not calculate progress")
                logger.warning(e)
        except KeyboardInterrupt as e:
            logger.warning("Keyboard interrupt")
            raise e

        # Event is created, you can process it now

    def calculate_progress(total, queue):
        progress = (total - queue) / total
        dict_ = {"complete": progress, "remaining": 1 - progress}
        with open(PROGRESS_FILE, "w") as f:
            json.dump(dict_, f)
        return dict_

    def parse_params(param_paths):
        param_df = pd.DataFrame()
        for i in range(len(param_paths)):
            param = param_paths[i]
            id_ = param.parent.name
            with open(param, "r") as f:
                params = json.load(f)
            _ = {}
            if "data" in params.keys():
                _.update(**params['data'].pop('classification'))
                _.update(**params['data'].pop('add_noise'))
            if 'model' in params.keys():
                _.update(**params['model'].pop('params'))
            else:
                _.update(params)
            # print(params)
            df2 = pd.DataFrame(_, index = [id_])
            param_df = pd.concat([param_df, df2])
        return param_df

    def merge_all_results(directory = "experiments", met_file = "**/scores.csv", html_file = "**/index.html", pred_file = "**/predictions.csv", ground_truth="**/ground_truth.csv", param_file = "**/params.json"):
        metr_paths = list(Path(directory).rglob(met_file))
        html_paths = list(Path(directory).rglob(html_file))
        pred_paths = list(Path(directory).rglob(pred_file))
        param_paths = list(Path(directory).rglob(param_file))
        ground_truth_paths = list(Path(directory).rglob(ground_truth))
        print(f"Found {len(metr_paths)} metrics, {len(html_paths)} htmls, {len(pred_paths)} predictions, {len(param_paths)} params, {len(ground_truth_paths)} ground truths")
        params = parse_params(param_paths)
        metrics = parse_params(metr_paths)
        p_cols = set(params.columns)
        m_cols = set(metrics.columns)
        # samesies = list(p_cols.intersection(m_cols))
        # metrics.drop(samesies, axis = 1, inplace = True)
        big = params.merge(metrics, left_index = True, right_index = True, how = 'outer')
        ids_ = big.index
        big['link'] = [Path(big.index[i], "index.html") for i in range(len(big))]
        big['predictions'] = [Path(big.index[i], "predictions.csv") for i in range(len(big))]
        big['ground_truth'] = [Path(big.index[i], "ground_truth.csv") for i in range(len(big))]
        big['params'] = [Path(big.index[i], "params.json") for i in range(len(big))]
        big['live'] = [Path(big.index[i], "report.html") for i in range(len(big))]
        big['link'] = big['link'].apply(lambda x: f'<a href="{x}">{str(x).split("/")[0]}</a>')
        big['predictions'] = big['predictions'].apply(lambda x: f'<a href="{x}">Predictions</a>')
        big['ground_truth'] = big['ground_truth'].apply(lambda x: f'<a href="{x}">Ground Truth</a>')
        big['params'] = big['params'].apply(lambda x: f'<a href="{x}">Params</a>')
        big['live'] = big['live'].apply(lambda x: f'<a href="{x}">Live Report</a>')

        big.index = big['link']
        del big['link']
        return big

    _

    def transform_json(directory = "experiments", met_file = "**/metrics.json", html_file = "**/report.html", pred_file = "**/predictions.json", param_file = "**/params.json"):
        # progress = calculate_progress(TOTAL, QUEUE)
        df = merge_all_results(directory, met_file, html_file, pred_file, param_file)

        print(f"Index updated at {Path(directory, 'index.html')}")
        # print(f'{progress["achieved"]*100:.2f} percent of experiments completed!'.format(progress["achieved"] * 100))
        return df

    def update_index(df, directory = "experiments"):
        df.to_html(Path(directory, "index.html"))
