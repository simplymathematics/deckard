from base.utils import return_result





if __name__ == '__main__':
    import sys
    import os
    import logging
    import argparse

    # command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--folder', type=str, default='.', help='Folder containing the checkpoint.')
    parser.add_argument('--scorer', type=str, default='f1', help='Scorer string.')