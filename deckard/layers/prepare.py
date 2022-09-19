from deckard.base.data import Data
import logging, argparse
from pathlib import Path
import dvc.api
from deckard.base.parse  import make_output_folder

logger = logging.getLogger(__name__)


def prepare(args) -> None:
    if hasattr(args, "config") and not hasattr(args, "inputs"):
        data = Data(dataset=args.config["name"], **args.config["params"])
    elif hasattr(args, "inputs") and not hasattr(args, "config"):
        data = Data(dataset=args.inputs["file"], **args.inputs["params"])
    elif not hasattr(args, "inputs") and not hasattr(args, "config"):
        raise ValueError("No data file specified.")
    else:
        raise ValueError("Both data file and config file specified.")
    filename = Path(args.outputs["folder"], args.outputs["file"]).resolve()
    data.save(filename)
    return filename


if __name__ == "__main__":
    # arguments
    params = dvc.api.params_show()
    args = argparse.Namespace(**params["prepare"])
    output_folder = make_output_folder(args.outputs["folder"])
    filename = prepare(args)
    assert Path(filename).exists(), "Problem saving data file: {}".format(filename)
