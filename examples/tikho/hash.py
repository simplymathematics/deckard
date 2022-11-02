import hashlib
import subprocess
from pathlib import Path
import dvc.api
from os import listdir
from shutil import rmtree, move
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
    print("Rendering report")
    subprocess.run(["dvc", "plots", "show", "-o", report_path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    real_time_report = Path(report_path, "report.html")
    plot_report = Path(report_path, "index.html")
    results = [ground_truth, predictions, scores, real_time_report, plot_report]
    for result in results:
        try:
            assert result.exists(), f"{result} does not exist"
            new_path = root_path/params["hash"]["out"]/str(file_hash.hexdigest())/result.name
            print(f"Moving {result} to {new_path}")
            move(result, new_path)
            assert new_path.exists(), f"{new_path} does not exist"
        except:
            print("~"*80)
            print(f"Could not find {result}")
            print(listdir(result.parent))
            print("~"*80)
    assert Path(new_path.parent / "index.html").exists(), "Plots were not rendered"
    assert Path(new_path.parent / filename).exists(), "Params was not saved"