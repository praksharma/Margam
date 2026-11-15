"""Optional desktop-owner pipe for a headless CLI process; no external PID control."""
import os
import signal
from threading import Event,Thread,Timer


def watch_owner_stdin():
    """EOF interrupts the CLI; an isolated session has a bounded hard-stop fallback."""
    if os.name!='posix':raise ValueError('Desktop owner watching requires POSIX signals.')
    finished=Event();own_session=os.getpid()==os.getpgrp()==os.getsid(0)
    def expired():
        if not finished.is_set() and own_session:os.killpg(os.getpgrp(),signal.SIGKILL)
    def read():
        try:
            while os.read(0,1):pass
        except OSError:pass
        if finished.is_set():return
        if own_session:
            timer=Timer(10,expired);timer.daemon=True;timer.start()
        os.kill(os.getpid(),signal.SIGINT)
    Thread(target=read,daemon=True).start()
    return finished
