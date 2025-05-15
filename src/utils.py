import sys
import os
from contextlib import contextmanager
import logging

@contextmanager
def suppress_output(on=True):
    if on:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Suppress logging
        original_log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            original_log_level_pymc = logging.getLogger("pymc").getEffectiveLevel()
            logging.getLogger("pymc").setLevel(logging.CRITICAL)
        except:
            pass
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging.getLogger().setLevel(original_log_level)
            try:
                logging.getLogger("pymc").setLevel(original_log_level_pymc)
            except:
                pass
    else:
        yield