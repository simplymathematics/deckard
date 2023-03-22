import logging
import subprocess

# from .compile import compile


logger = logging.getLogger(__name__)
LAYERS = ["runner", "optimize", "compile", "parse"]

if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage to load params from. If None, loads the last stage in dvc.yaml",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="runner",
        help="Layer to run. Choose from " + str(LAYERS),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Verbosity of logging. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'.",
    )
    parser.add_argument(
        "params",
        type=list,
        default=None,
        help="Params to merge with params from dvc.yaml. Params can be a json dict or a path to a yaml file.",
        nargs="*",
    )
    args = parser.parse_args()
    assert args.layer in LAYERS, "Layer must be one of " + str(LAYERS) + "."
    logging.basicConfig(level=args.verbosity)
    logger.info(f"Running layer {args.layer}.")
    logger.info(f"Running stage {args.stage}.")
    if args.layer == "runner":
        cmd = "python -m deckard.layers.runner --stage " + args.stage
    elif args.layer == "parse":
        if args.stage is not None:
            cmd = "python -m deckard.layers.parse +stage=" + args.stage
        else:
            cmd = "python -m deckard.layers.parse "
    elif args.layer == "compile":
        if args.stage is not None:
            cmd = "python -m deckard.layers.compile --stage " + args.stage
        else:
            cmd = "python -m deckard.layers.compile"
        if args.params != []:
            cmd += " " + args.params
    elif args.layer == "optimize":
        cmd = "python -m deckard.layers.optimize +stage=" + args.stage
        if isinstance(args.params, list):
            for param in args.params:
                cmd += " " + "".join(param)
    else:
        raise NotImplementedError(f"Layer {args.layer} not implemented.")
    # Run command and output to stdout in real time
    logger.info("Running command: " + cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
    # while output.poll() is None:
    #     out = output.stdout.read(1)
    #     sys.stdout.write(out)
    #     sys.stdout.flush()
