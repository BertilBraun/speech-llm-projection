# Speech projector overnight research

Frozen Whisper Small encoder → trainable temporal compressor/projector → frozen
Qwen3.5-2B text decoder. Previous dialogue turns use text, the current utterance
uses only cached speech features, and teacher-forced CE trains only the projector.

The experiment covers feasibility, nested data scaling, temporal compression and
linear/MLP/learned temporal architectures. DeepDialogue XTTS is synthetic spoken
dialogue, so conclusions concern this dataset rather than natural conversations.

Code uses typed Pydantic configuration/manifests and local JSON/JSONL artifacts.
The rented node is `/workspace/speech-projector`; checkpoints and results must
also be copied locally because the node has no persistent volume.

See [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md) for live progress. Exact
commands and results will be added as the pipeline is validated.
