"""synapforge.native -- torch-free runtime primitives.

This package hosts production code that does NOT import torch. Sub-packages:

* ``auxsched`` -- async coordinator + per-component policy for curiosity
  / TTT / NeuroMCP / ActionHead. Runs each on its own stream / thread
  so they overlap with the main forward+backward pass instead of
  blocking it. (Named ``auxsched`` rather than ``aux`` because Windows
  reserves ``AUX`` as a legacy DOS device name and refuses native-API
  ``open()`` on files inside any folder named ``aux``; the rename keeps
  the package portable.)

There is also a sibling ``dispatch`` package (built on a separate feature
branch) that provides ``HeteroPipeline`` and ``StreamPair`` infrastructure.
``aux`` opportunistically re-uses those when available; if not, ``aux``
ships its own minimal stream/thread primitives so the policy layer stands
alone.
"""

from __future__ import annotations
