import hashlib
import subprocess
from pathlib import Path
import dvc.api
from shutil import copy2 as copy
EXPERIMENT_PATH = "/Home/staff/cmeyers/deckard/examples/tikho"
if __name__ == '__main__':
    params = dvc.api.params_show()
    report_path = Path(params["hash"]["in"])
    filename = Path(params["hash"]["file"])
    if EXPERIMENT_PATH != Path.cwd():
        root_path = Path(EXPERIMENT_PATH)
    else:
        root_path = Path.cwd()
    print(params)
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    ground_truth = Path(params['data']['files']['path'] , params["data"]["files"]["ground_truth"])
    predictions = Path(params['model']['files']['path'] , params["model"]["files"]["predictions"])
    scores = Path(params['model']['files']['path'] , params["model"]["metrics"]["scores"])
    real_time_report = Path(report_path, "report.html")
    results = [ground_truth, predictions, scores, real_time_report,Path("params.yaml")]
    new_path = root_path/params["hash"]["out"]/str(file_hash.hexdigest())
    new_path.mkdir(parents=True, exist_ok=True)
    for result in results:
        _ = new_path/result.name
        print(f"Moving {result} to {_}")
        assert result.exists(), f"{result} does not exist"
        copy(result, _)
        assert _.exists(), f"{_} could not be copied"
    print("Rendering report")
    subprocess.run(["dvc", "plots", "show", "-o", new_path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    plot_report = Path(report_path, "index.html")
    assert Path(new_path / "index.html").exists(), f"Plots were not rendered: {new_path / 'index.html'}"
    assert Path(new_path / filename).exists(), f"Params was not saved: {new_path / filename}"