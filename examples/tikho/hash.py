import hashlib
import subprocess
from pathlib import Path
import dvc.api

if __name__ == '__main__':
    params = dvc.api.params_show()
    path = Path(params["hash"]["out"])
    report_path = Path(params["hash"]["in"])
    filename = params["hash"]["file"]
    print(params)
    input("Press Enter to continue...")
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    unique_id = file_hash.hexdigest()
    print(unique_id)
    path = path / unique_id
    path.mkdir(exist_ok=True, parents=True)
    print(f"Moving folder from {report_path} to {path}")
    subprocess.run(["mv", f"{report_path}/*", f"{path}/*"])
    print(f"Moving file from {filename} to {path}")
    subprocess.run(["cp", "-v", filename, path ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Rendering plots in {path}/index.html")
    subprocess.run(["dvc", "plots", "show", "-o", path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    with open("dvc.yaml", "r") as f:
        dvc_yaml = f.read()
    
    dvc_yaml = dvc_yaml.replace(r"$hash$", unique_id)
    
    with open("dvc.yaml", "w") as f:
        f.write(dvc_yaml)
    
    