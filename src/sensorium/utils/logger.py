import os
import sys
import typing as t


class RedirectOutput:
    """Re-direct stdout or stderr to log file"""

    def __init__(self, file: t.TextIO, stream: t.Literal["stdout", "stderr"]):
        assert stream in ("stdout", "stderr")
        self.file = file
        self.console = sys.stdout if stream == "stdout" else sys.stderr
        self._tqdm_newline = False

    def write(self, message: str):
        self.console.write(message)
        self.console.flush()
        if message:
            if self._tqdm_newline and message == "\n":
                # skip the newline message by tqdm when a loop has ended
                self._tqdm_newline = False
            elif message.startswith("\r"):
                # skip carriage return from tqdm
                self._tqdm_newline = True
            else:
                self.file.write(message)
                self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


class Logger:
    """Write both stdout and stderr to file"""

    def __init__(self, args):
        filename = os.path.join(args.output_dir, "output.log")
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.file = open(filename, mode="a")
        sys.stdout = self.stdout = RedirectOutput(self.file, stream="stdout")
        sys.stderr = self.stderr = RedirectOutput(self.file, stream="stderr")

    def close(self):
        self.stdout.flush()
        self.stderr.flush()
        self.file.close()
