## About this prototype

**Purpose.** Educational research prototype for a Generative AI course assignment. It screens a patient
record against each inclusion and exclusion criterion of a trial and cites the patient sentences it relied on.
**It is not a clinical enrollment decision tool.**

**Model.** Qwen2.5-1.5B-Instruct (Apache-2.0) fine-tuned with LoRA on the training split of the
TrialGPT Criterion Annotations dataset (37 patients). Draft model for speculative decoding:
Qwen2.5-0.5B-Instruct (same tokenizer).

**Labels.** Inclusion: included, not included, not enough information, not applicable.
Exclusion: excluded, not excluded, not enough information, not applicable. Missing information is never
treated as proof of eligibility or ineligibility.

**Explanations.** Built only from the predicted label and the exact cited sentences, so every
explanation points to text that exists in the record. A decision with no cited sentence is flagged.

**Preliminary outcome.** A rule-based aggregation, not a ground-truth label: any exclusion predicted
"excluded" or inclusion predicted "not included" gives "likely ineligible"; otherwise any
"not enough information" gives "undetermined"; otherwise "potentially eligible".

**Decoding modes.** Explicit KV cache (our own Qwen2 forward pass and cache), Hugging Face forward with a
DynamicCache driven by our loop, no-cache reference (full recomputation each step), and greedy speculative
decoding (our implementation). All are greedy, so they should produce identical tokens.
