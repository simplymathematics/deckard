import hashlib
import subprocess
from pathlib import Path
import dvc.api
from os import rename, listdir
from shutil import copy2 as copy
from shutil import rmtree
EXPERIMENT_PATH = "Home/staff/cmeyers/deckard/examples/tikho"
if __name__ == '__main__':
    params = dvc.api.params_show()
    path = Path(params["hash"]["out"])
    report_path = Path(params["hash"]["in"])
    filename = Path(params["hash"]["file"])
    root_path = Path(EXPERIMENT_PATH)
    print(params)
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    ground_truth = Path(params['data']['files']['path'] , params["data"]["files"]["ground_truth"])
    predictions = Path(params['model']['files']['path'] , params["model"]["files"]["predictions"])
    scores = Path(params['model']['files']['path'] , params["model"]["metrics"]["scores"])
    results = [ground_truth, predictions, scores]
    unique_id = file_hash.hexdigest()
    print(unique_id)
    new_path = Path(root_path, path , unique_id)
    print("*"*80)
    print(new_path)
    print("*"*80)
    if new_path.exists():
        print("Already exists. Removing old files")
        rmtree(new_path)
    new_path.mkdir(exist_ok=True, parents=True)
    real_time_report = Path(report_path, "report.html")
    try:
        copy(real_time_report, Path(new_path, "report.html"))
    except:
        try:
            print("!1"*80)
            print(listdir(real_time_report.parent))
            print("!1"*80)
            print(listdir(real_time_report.parent.parent))
        except:
            print("!2"*80)
            print(listdir(real_time_report.parent.parent))
    new_results = [new_path / result.name for result in results]
    print("Moving results to new location")
    for result, new_result in zip(results, new_results):
        assert result.exists()
        new_result.parent.mkdir(exist_ok=True, parents=True)
        print(f"Moving {result} to {new_result}")
        copy(result, new_result.resolve())
    run_time = [Path(report_path, "scalars"), ]
    new_run_time = [path / run.name for run in run_time]
    print(f"Moving params file from {filename.resolve()} to {new_path.resolve()}")
    print(f"Old Path tree: {path}")
    print(subprocess.run("tree", cwd=path.resolve()))
    print(f"New Path tree : {new_path}")
    print(subprocess.run("tree", cwd=new_path.resolve()))
    rename(filename, new_path / filename)
    print(f"Rendering plots in {path}/index.html")
    subprocess.run(["dvc", "plots", "show", "-o", path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert Path(new_path / "index.html").exists(), "Plots were not rendered"
    assert Path(new_path / filename).exists(), "Params was not saved"