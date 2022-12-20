from tqdm import tqdm
import time
import logging

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


log = logging.getLogger (__name__)
log.setLevel (logging.INFO)
log.addHandler (TqdmLoggingHandler ())
for i in tqdm(range (1000)):
    if (i+1) % 10 == 0:
        log.info("Half-way there!")
    time.sleep (0.1)
