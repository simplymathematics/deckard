import argparse
import json
import logging
import time
from pathlib import Path

import watchdog.events
import watchdog.observers
from gevent import joinall
from pssh.clients import ParallelSSHClient

PROGRESS_FILE = "progress.json"
events = []
logger = logging.getLogger(__name__)


def createSSHClient(hosts, port, user, password):
    client = ParallelSSHClient(hosts, port=port, user=user, password=password)
    output = client.run_command("ls -l")
    assert output[0].exit_code == 0, "SSH connection failed"
    logger.info("Parallel SSH connections established")
    return client


class JSONHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self, servers, port, user, password, filename, destination, **kwargs):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(
            self, patterns=[REGEX], ignore_directories=True, case_sensitive=False
        )
        self.ssh = createSSHClient(servers, port, user, password)
        logger.info("Initiated SSH client")
        self.filename = filename
        self.destination = destination
        self.recurse = kwargs["recursive"] if "recurse" in kwargs else False
        logger.info(
            "Source file is {} and destination is {}".format(
                self.filename, self.destination
            )
        )
        logger.info("Regex is {}".format(REGEX))

    def on_created(self, event):
        logger.info("Watchdog received created event - % s." % event.src_path)
        events.append(event.src_path)
        self.filename = event.src_path
        try:
            self.transform_json()
            logger.info("Transformed JSON")
        except Exception as e:
            logger.warning("Could not transform json")
            logger.warning(e)
        if "TOTAL" and "QUEUE" in locals():
            try:
                self.calculate_progress(TOTAL, QUEUE)
                logger.info("Calculated progress")
            except Exception as e:
                logger.warning("Could not calculate progress")
                logger.warning(e)
        try:
            self.send_json_with_scp()
            logger.info("Sent JSON")
        except KeyboardInterrupt as e:
            logger.warning("Keyboard interrupt")
            raise e
        except Exception as e:
            logger.warning("Could not send json")
            logger.warning(e)

        # Event is created, you can process it now

    def calculate_progress(total, queue):
        progress = (total - queue) / total
        dict_ = {"complete": progress, "remaining": 1 - progress}
        with open(PROGRESS_FILE, "w") as f:
            json.dump(dict_, f)
        return dict_

    def transform_json(self):
        pass

    def send_json_with_scp(self):
        remotename = Path(self.destination, self.filename).as_posix()
        cmds = self.ssh.scp_send(self.filename, remotename, recurse=self.recurse)
        joinall(cmds, raise_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some json files and send them to a server. Or send and then process. Your choice."
    )
    parser.add_argument(
        "--source", "-i", type=str, required=True, help="The source to watch for files."
    )
    parser.add_argument(
        "--destination",
        "-o",
        type=str,
        required=True,
        help="The destination to send the files to.",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        required=True,
        help="The server to send the files to.",
    )
    parser.add_argument("--port", "-p", type=int, help="The port to send the files to.")
    parser.add_argument(
        "--user", "-u", type=str, required=True, help="The user to send the files to."
    )
    parser.add_argument(
        "--password",
        "-k",
        type=str,
        required=True,
        help="The password to send the files to.",
    )
    parser.add_argument("--original", type=str, help="The original queue file.")
    parser.add_argument("--queue", type=str, help="The current queue file.")
    parser.add_argument(
        "--regex", "-e", type=str, required=True, help="The regex to watch for."
    )
    parser.add_argument(
        "--recursive", "-r", type=bool, default=True, help="Whether to recurse or not."
    )
    parser.add_argument(
        "--n_jobs",
        "-j",
        type=int,
        default=8,
        help="The number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--log", "-l", type=int, default=logging.INFO, help="The log level."
    )
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(
        level=args.log, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if args.regex is not None:
        REGEX = args.regex
    else:
        raise ValueError("You must specify a regex to watch for.")
    # Assuming this is watching some long-running process (like a model training), 
    # you may find it beneficial to watch the progress.
    # First, generate an "original" file that contains one line
    # for every experiment configuration you would like to test.
    # The contents don't matter. It only counts lines.
    # Then, when each experiment is complete, pop a line from that file. 
    # This is called the "queue" file.
    # If these files exist, you will get a log to stdout and a 
    # progress.json file containing the "completed" and "remaining" amounts.
    if args.original is not None:
        with open(args.original, "r") as f:
            TOTAL = len(f.readlines())
    if args.queue is not None:
        with open(args.queue, "r") as f:
            QUEUE = len(f.readlines())
    # SUPPORTS PARALELL HOSTS. Specify n jobs or write a list of hosts here.
    hosts = [args.server] * args.n_jobs
    src_path = Path(args.source).parent
    event_handler = JSONHandler(
        hosts,
        args.port,
        args.user,
        args.password,
        args.source,
        args.destination,
        recursive=args.recursive,
    )
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
