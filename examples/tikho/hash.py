import hashlib
import subprocess
from pathlib import Path
import dvc.api
from os import rename, rmdir
from shutil import copy2 as copy
from shutil import rmtree
if __name__ == '__main__':
    params = dvc.api.params_show()
    path = Path(params["hash"]["out"])
    report_path = Path(params["hash"]["in"])
    filename = params["hash"]["file"]
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
    path.mkdir(exist_ok=True, parents=True)
    path = path / unique_id
    if path.exists():
        print("Already exists. Removing old files")
        rmtree(path)
    new_results = [path / result.name for result in results]
    print("Moving results to new location")
    for result, new_result in zip(results, new_results):
        copy(result, new_result)
    run_time = [Path(report_path, "scalars"), Path(report_path, "report.html")]
    new_run_time = [path / run.name for run in run_time]
    print("Moving run time results to new location")
    
    print(f"Moving params file from {filename} to {path}")
    rename(filename, path / filename)
    print(f"Rendering plots in {path}/index.html")
    subprocess.run(["dvc", "plots", "show", "-o", path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert Path(path / "index.html").exists(), "Plots were not rendered"
    assert Path(path / filename).exists(), "Params was not saved"