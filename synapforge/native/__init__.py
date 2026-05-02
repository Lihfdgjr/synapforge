"""synapforge.native — LNN+SNN-architecture-specific native primitives.

This package collects optimisations that exploit properties unique to
the SynapForge LNN+SNN architecture (binary spikes, sparse activation,
event-driven temporal dynamics) and that have no analogue in dense
transformer training.

Subpackages
-----------
synapforge.native.spike
    Bit-packed spike storage + sparse-matmul Triton kernels.  16 binary
    spikes per uint16 word, 16x memory and HBM-bandwidth savings at the
    PLIF -> synapse boundary.

This namespace is parallel to ``synapforge.kernels`` (which holds the
older int32-storage spike-pack and the EmbeddingBag-based sparse-spike
matmul).  The ``native`` versions target the next-generation Triton
kernels that operate directly on packed bits without unpacking to HBM.
"""
