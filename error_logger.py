
import sys
import os
import traceback

def log_exception(exc_type, exc_value, exc_traceback):
    with open(os.path.expanduser("~/ssvep_error.log"), "a") as f:
        f.write("Exception occurred:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    return sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_exception
