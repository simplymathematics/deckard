from typing import Union
from pathlib import Path
from ..base.parse import generate_object_from_tuple, generate_tuple_from_yml, generate_experiment_list

def make_output_folder(output_folder:Union[str, Path]) -> Path:
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    global ART_DATA_PATH
    ART_DATA_PATH = output_folder
    assert Path(output_folder).exists(), "Problem creating output folder: {}".format(output_folder)
    return Path(output_folder).resolve()

def reproduce_directory_tree(input_folder:Union[str, Path], output_folder:Union[str, Path], input_file:Union[str,Path]) -> None:
    old_files = [path for path in Path(input_folder).rglob('*' + input_file)]
    old_folders = [path.parent for path in old_files]
    new_folders = [Path(output_folder, path).resolve() for path in old_folders]
    for folder in tqdm(new_folders, desc = "Creating Directories"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        assert(os.path.isdir(folder.resolve())), "Problem creating folder: {}".format(folder)
    return old_files, new_folders

def parse_config(config:Union[dict, str, Path],**kwargs) -> object:
    tuple_ = generate_tuple_from_yml(config)
    assert isinstance(tuple_, tuple), "Problem generating tuple from config file: {}".format(config)
    obj_ = generate_object_from_tuple(tuple_)
    assert isinstance(obj_, object), "Problem generating object from tuple: {}".format(tuple_)
    return obj_

def create_config_dict(config:Union[str, Path]) -> list:
    big = {}
    assert Path(config).exists(), "Config file does not exist: {}".format(config)
    if Path(config).is_file():
        big[config] = parse_config(config)
    elif Path(config).is_dir():
        for file in tqdm(Path(config).rglob('*.yml'), desc = "Parsing Config Files"):
            big[file] = parse_config(file)
    else:
        raise ValueError("Config must be a file or directory. It is neither.")
    return big