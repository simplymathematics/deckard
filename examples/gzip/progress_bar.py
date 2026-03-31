from pathlib import Path
import time
import yaml
import argparse
import sys
import logging

logger = logging.getLogger(__name__)


def find_total_runs(dvc_pipeline_file, stage, scale=1):
    with open(dvc_pipeline_file) as f:
        dvc_pipeline = yaml.safe_load(f)["stages"][stage]
    # Get the number of runs from the dvc pipeline file
    assert "matrix" in dvc_pipeline, f"Matrix not found in stage {stage}"
    total_runs = scale
    for _, list_ in dvc_pipeline["matrix"].items():
        total_runs *= len(list_)
    return total_runs


def count_files(regex, folder):
    assert Path(folder).exists(), FileNotFoundError(f"Folder: {folder} does not exist.")
    result = list(Path(folder).glob(regex))
    if not result:
        logger.warning(f"No files found in {folder} matching {regex}")
    return len(result)


def progress_bar_main(folder, regex, sleep_time, count_files, total_runs):
    while sys.stdin.isatty():
        # Check for end
        completed_runs = count_files(regex, folder)
        progress = completed_runs / total_runs

        # Log
        log_file = "progress.log"
        if not Path(log_file).exists():
            start_progress = 0
            start_time = time.time()
        else:
            start_time, start_progress = read_progress(log_file)
        if "first_count" not in locals():
            first_count = int(completed_runs)
        # Create the progress bar

        bar_length = 40
        filled_length = int(round(bar_length * progress))
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        bar_string = (
            f"Progress: [{bar}] {completed_runs}/{total_runs} ({progress*100:.2f}%)"
        )

        time.sleep(sleep_time)
        timestamp = time.time()
        log_progress(progress, timestamp)
        if progress > start_progress and completed_runs > first_count:
            # Estimate time remaining based on the rate of progress
            end_time = time.time()
            change_in_time = end_time - start_time
            change_in_progress = progress - start_progress
            progress_per_second = change_in_progress / change_in_time
            seconds_per_unit_progress = 1 / progress_per_second
            if sleep_time > seconds_per_unit_progress:
                sleep_time = seconds_per_unit_progress / 2
            time_remaining = (1 - progress) / progress_per_second
            # Make time remaining human-readable using time module
            if time_remaining > 7 * 24 * 3600:
                completed_at_string = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(end_time + time_remaining),
                )
                sleep_time = min(3600, sleep_time)
            elif time_remaining > 24 * 3600:
                completed_at_string = time.strftime(
                    "%m-%d %H:%M:%S",
                    time.localtime(end_time + time_remaining),
                )
                sleep_time = min(3600, sleep_time)
            else:
                completed_at_string = time.strftime(
                    "%H:%M:%S",
                    time.localtime(end_time + time_remaining),
                )
                sleep_time = min(600, sleep_time)
        else:
            completed_at_string = "Unknown"
        # Erase the current line content
        sys.stdout.write("\x1b[2K")
        print(
            bar_string + f". To be completed at: {completed_at_string}",
            end="\r",
            flush=True,
        )
        if completed_runs >= total_runs:
            end_string = time.strftime("%m-%d %H:%M:%S", time.localtime())
            print(f"\nAll {total_runs} runs completed at {end_string}!")
            break


def log_progress(progress, timestamp, file="progress.log"):
    assert isinstance(progress, (int, float))
    assert isinstance(timestamp, (int, float))

    Path(file).parent.mkdir(exist_ok=True)
    if not Path(file).exists():
        Path(file).touch()
    with Path(file).open("+a") as f:
        f.write(f"{timestamp} : {progress}\n")


def read_progress(file):
    assert Path(file).exists(), f"File {file} does not exist"
    # Read the file and parse "timestamp : progress"
    with open(file, "r") as f:
        lines = f.readlines()
        first_line = lines[0]
    result = first_line.split(":")
    if len(result) == 2:
        return float(result[0]), float(result[1])
    else:
        print(f"Result: {result}")
        input("Press Enter to raise error")
        raise FileExistsError("File blank")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor progress of DVC pipeline runs",
    )
    parser.add_argument("--folder", default="output/", help="Output folder path")
    parser.add_argument(
        "--regex",
        default="*/logs/search/*/*/*/optimization_results.yaml",
        help="File pattern to match",
    )
    parser.add_argument(
        "--stage",
        default="grid_search",
        help="DVC pipeline stage name",
    )
    parser.add_argument("--dvc-file", default="dvc.yaml", help="DVC pipeline file path")
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Scale factor for total runs",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=1,
        help="Sleep time between checks in seconds",
    )
    parser.add_argument(
        "--total-runs",
        type=int,
        default=None,
        help="Total number of runs (optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # full_path = Path(args.folder) / args.regex
    # Split the full path into folder and regex
    folder = str(Path(args.folder))
    regex = str(args.regex)
    logger.info(f"Monitoring folder: {folder}, regex: {regex}")
    stage = args.stage
    dvc_pipeline_file = args.dvc_file
    scale = args.scale
    sleep_time = args.sleep_time
    total_runs = None

    if total_runs is None:
        total_runs = find_total_runs(dvc_pipeline_file, stage, scale)
    # Check for keyboard interrupt to exit the progress bar
    progress_bar_main(folder, regex, sleep_time, count_files, total_runs)
