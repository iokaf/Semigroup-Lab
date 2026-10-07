"""Process-wide lock for mpmath's working precision.

mpmath keeps its working precision in one global context, and ``mp.workdps(n)`` changes it for
everyone. When several users analyse at the same time (a hosted Streamlit app runs each session
in its own thread), one session can change the precision under another's computation. All engine
code that uses the mpmath context runs while holding this lock; it is re-entrant, so nested
engine calls are fine.
"""
import threading

MP_LOCK = threading.RLock()
