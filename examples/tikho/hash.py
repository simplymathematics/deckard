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
    print("Rendering report")
    subprocess.run(["dvc", "plots", "show", "-o", path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    real_time_report = Path(report_path, "report.html")
    plot_report = Path(path, "index.html")
    
    results = [ground_truth, predictions, scores, real_time_report, plot_report]
    for result in results:
        try:
            assert result.exists(), f"{result} does not exist"
        except:
            print("~"*80)
            print(listdir(result.parent))
            print("~"*80)
    unique_id = file_hash.hexdigest()
    new_path = Path(str(root_path), str(path) , str(unique_id)).resolve()
    print("*"*80)
    print(new_path)
    print("*"*80)
    
    if new_path.exists():
        print("Already exists. Removing old files")
        rmtree(new_path)
    new_path.mkdir(exist_ok=True, parents=True)
    
    subprocess.run(["dvc", "plots", "show", "-o", path, "--html-template", "template.html"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert Path(new_path / "index.html").exists(), "Plots were not rendered"
    assert Path(new_path / filename).exists(), "Params was not saved"