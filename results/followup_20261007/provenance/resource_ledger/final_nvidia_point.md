# Final idle NVIDIA point observation

Read-only node query, captured at2026-10-07 13:55:35UTC after root confirmed
all experiment GPU jobs stopped:

```text
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
NVIDIA GeForce RTX 3090, 24576, 32, 0
```

Total24,576MiB, observed used32MiB and instantaneous utilization0%. This is
an idle point observation, not a training/inference peak, PyTorch allocator
reservation or utilization averaged over the experiment.

Saved7.739920896 decimalGB allocated peak is a lineage maximum carried through
continuation metadata. Root's earlier22,716MiB transcript-stage NVIDIA usage
sample is another point observation and must not be called reserved memory.
No inference-memory peak or utilization-hours are inferred.
