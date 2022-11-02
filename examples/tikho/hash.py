import hashlib
import subprocess
from pathlib import Path
import dvc.api
from os import listdir
from shutil import copy2 as copy
EXPERIMENT_PATH = "Home/staff/cmeyers/deckard/examples/tikho"
if __name__ == '__main__':
    params = dvc.api.params_show()
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
    real_time_report = Path(report_path, "report.html")
    results = [ground_truth, predictions, scores, real_time_report,Path("params.yaml")]
    new_path = root_path/params["hash"]["out"]/str(file_hash.hexdigest())
    for result in results:
        _ = new_path/result.name
        print(f"Moving {result} to {_}")
        try:
            assert result.exists(), f"{result} does not exist"
            copy(result, _)
            assert _.exists(), f"{_} could not be copied"
        except:
            print("~"*80)
            print(f"Could not find {result}")
            print(listdir(result.parent))
            print("~"*80)
    print("Rendering report")
    subprocess.run(["dvc", "plots", "show", "-o", new_path.parent, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    plot_report = Path(report_path, "index.html")
    assert Path(new_path.parent / "index.html").exists(), f"Plots were not rendered: {new_path.parent / 'index.html'}"
    assert Path(new_path.parent / filename).exists(), f"Params was not saved: {new_path.parent / filename}"