import hashlib
import subprocess
from pathlib import Path
import dvc.api
from os import rename
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

    unique_id = file_hash.hexdigest()
    print(unique_id)
    path.mkdir(exist_ok=True, parents=True)
    path = path / unique_id
    print(f"Moving folder from {report_path} to {path}")
    rename(report_path, path)
    print(f"Moving file from {filename} to {path}")
    rename(filename, path / filename)
    print(f"Rendering plots in {path}/index.html")
    subprocess.run(["dvc", "plots", "show", "-o", path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)