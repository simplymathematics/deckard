import hashlib
import subprocess
from pathlib import Path
import dvc.api
from shutil import copy2 as copy
import json
from hashlib import md5

EXPERIMENT_PATH = "../.tikho"

if __name__ == "__main__":
    params = dvc.api.params_show()
    files = params.pop("files")
    report_path = "reports"
    filename = md5(json.dumps(params).encode()).hexdigest()
    if EXPERIMENT_PATH != Path.cwd():
        root_path = Path(EXPERIMENT_PATH)
    else:
        root_path = Path.cwd()
    print(params)
    with open("params.yaml", "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    ground_truth = Path(
        files["path"],
        files["ground_truth_file"],
    )
    predictions = Path(
        files["path"],
        files["predictions_file"],
    )
    scores = Path(
        files["path"],
        files["score_dict_file"],
    )
    real_time_report = Path(report_path, "report.html")
    results = [
        ground_truth,
        predictions,
        scores,
        real_time_report,
    ]
    new_path = root_path / str(file_hash.hexdigest())
    new_path.mkdir(parents=True, exist_ok=True)
    for result in results:
        _ = new_path / result.name
        print(f"Moving {result} to {_}")
        assert result.exists(), f"{result} does not exist"
        copy(result, _)
        assert _.exists(), f"{_} could not be copied"
    with open(new_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    print("Rendering report")
    subprocess.run(
        ["dvc", "plots", "show", "-o", new_path, "--html-template", "template.html"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    plot_report = Path(report_path, "index.html")
    assert Path(
        new_path / "index.html",
    ).exists(), f"Plots were not rendered: {new_path / 'index.html'}"
    