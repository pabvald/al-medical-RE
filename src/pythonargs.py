#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Simple Python template with an argument parser. Put all the "main" logic into the method called "main".
             Only use the true "__main__" section to add script arguments (and eventually a logger).

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: YOU!
"""

# Base Dependencies
# -----------------
import os
import sys
import time
import logging
import argparse
import traceback


def main(opts):
    """Main loop"""
    print("Main run")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        dest="infile",
        required=False,
        default=None,
        help="Input file",
    )

    opts = parser.parse_args(sys.argv[1:])

    try:
        main(opts)
    except Exception:
        print("Unhandled error!")
        traceback.print_exc()
        sys.exit(-1)

    print("All Done.")
