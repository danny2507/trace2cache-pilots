# Runtime evidence to latent repair messages: focused literature review

Review date: 2026-09-17. Focus: why our role-aware runtime encoder does not yet transfer
behavioral identity into useful patches, what prior work already covers, and what to test next.
This is a targeted review of primary papers and official project/conference pages, not an
exhaustive systematic review or a proof that a proposed combination is novel.

## 1. What the current experiments actually establish

The expanded oracle codebook supports receiver capacity on 24 behaviors / 12 pairs. Within a
pair, buggy code and public test are identical. Trained codes solve 23/24 with training wording
and 21/24 with held-out wording; paired opposite codes solve 0/24 and 1/24. The receiver is
Qwen2.5-1.5B-Instruct, frozen, and the interface uses eight input-embedding states. This is
soft-state injection, not a direct per-layer KV writer. The expanded codebook has 294,912
trainable parameters, as recorded in its artifact.

This establishes neither unseen-program generalization nor runtime compression: the oracle
lookup sees all behavior identities and programs during training. Held-out prompt wording
tests instruction robustness, not new source code.

The expanded trace encoder run (seed 317) achieves 40.1% nearest-code accuracy and 0.583
target-code cosine on combined longer-bundle/value-OOD evidence. True, paired-swap, and
expected-removed repairs each solve 6/24. Equal aggregate counts do not mean unchanged
outputs: an audit finds identical extracted patch strings in only 8/24 true-versus-swap cases
(identical complete responses in 6/24). Expected removal preserves the patch string in 7/24.
The supported claim is no demonstrated useful repair gap, not complete evidence insensitivity.

Current artifacts:

- `artifacts/repair_latent/expanded_codebook_seed313.json`
- `artifacts/repair_latent/expanded_role_aware_ood_seed317.json`

The current encoder averages *input token embeddings* within each event text before its
event Transformer. It therefore loses within-event order: sequences with the same token
multiset have the same content vector. Separate role/test/event positions do not recover
the lost token order. Number identity can remain without arithmetic or relational structure.
The current training objective is code-vector MSE plus cosine, with no repair-decoder loss.
These are plausible failure mechanisms, not established causes of the observed result.

## 2. Prior work on execution representations and repair

| Work | What it does | Relevance and limit |
|---|---|---|
| [Dynamic Neural Program Embeddings for Program Repair, ICLR 2018](https://arxiv.org/abs/1711.07163) | Compares per-variable traces, full-state traces, and a hybrid that exchanges recurrent hidden states along dependencies. Predicts error categories and guides repair search. | Execution embedding and explicit dependency preservation already exist. It does not supply a frozen pretrained CodeLM with latent runtime messages. |
| [Synthesize, Execute and Debug, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/cd0f74b5955dc87fd0605745c4b49ee8-Paper.pdf) | Embeds Karel execution states together with input/desired output; associates states with program tokens using an execution-to-token bipartite graph; learns edit generation. Synthetic mutations precede training on synthesizer errors. | A trace should retain its source association and specification, rather than be an ungrounded vector sequence. Architecture and decoder are trained for this domain. |
| [TraceFixer, 2023](https://arxiv.org/abs/2304.12743) | Learns Python repairs from source, partial buggy traces, and a user-specified desired state at a divergence point. Uses large synthetic edit training data. | Closest Python runtime-to-repair precedent. Desired intermediate state is stronger supervision than our observed buggy state alone; it should not be silently assumed available at inference. |
| [Towards Effectively Leveraging Execution Traces, KnowledgeNLP 2025](https://aclanthology.org/2025.knowledgenlp-1.17/) | Studies trace-augmented GPT repair prompts. Naive trace prompts improve only 2/6 dataset/model configurations; utility falls with complexity. Optimized prompts help more consistently. | Supports an interface/formatting problem. Does not establish that text is universally the wrong medium. Optimized textual evidence is a required baseline. |
| [Do Code Semantics Help?, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.548/) | Broadly evaluates execution-trace information for SFT and inference and reports limited usefulness in studied settings. | Trace usefulness is conditional on method and task; it is not evidence of latent superiority. |
| [CausalRepair, preprint August 2026](https://arxiv.org/abs/2608.10613) | Combines static slicing of test context and execution-based dynamic slicing of source into compact causal context for iterative LLM repair. | Filtering causal context already overlaps our slicing idea. Compare equal evidence as text and latent to isolate interface effects. |
| [NExT, 2024](https://arxiv.org/abs/2404.14662) | Self-trains execution-aware rationales filtered through correct solutions; evaluates repair based on MBPP and HumanEval. | Evidence-to-explanation training and execution-validated synthetic data are precedented. This trains the LLM and communicates through textual rationales. |

The key lesson is that neither trace embedding, dependency-aware encoding, synthetic bug
generation, nor learning runtime semantics for repair is new by itself.

## 3. Learning an interface that a language model can consume

| Work | Training/interface | What to borrow |
|---|---|---|
| [BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q.html) | Query Transformer connects frozen vision and language models; representation learning precedes generation learning through the frozen LLM. | Separate semantic sufficiency from receiver usability. Contrastive features need generative alignment. Runtime would be our modality; vision success is an analogy, not direct evidence for traces. |
| [ICAE, 2023/2024](https://arxiv.org/html/2307.06945v4) | LoRA-adapted LLM encoder produces memory states for a fixed decoder; autoencoding and continuation pretraining precede instruction supervision. | Reuse contextual model features; train decoding from compressed states. Reconstruction is a training objective and is not required before downstream inference. |
| [xRAG, NeurIPS 2024](https://arxiv.org/html/2405.13792v2) | Frozen document embedder and frozen LLM, with a learned projector; paraphrase pretraining then task NLL plus KL from the full-text teacher. | Very close to the proposed frozen receiver bridge. Distill task behavior, not just Euclidean proximity to a chosen latent vector. A semantic retriever embedding need not preserve exact runtime values. |
| [Cache-to-Cache, 2025](https://arxiv.org/html/2510.03215v2) | Projects/fuses contextual per-layer caches; both LLMs frozen; trains the fuser using receiver next-token loss. | Input embedding averages are not the same representation as contextual KV semantics. Receiver loss trains an interface to be usable. It is inter-LLM communication, not raw debugger-state encoding. |
| [LatentPress, preprint September 2026](https://arxiv.org/html/2609.01507v2) | Uses the reader's bottom two frozen Transformer layers, pooling and an identity-initialized trainable linear writer; frozen reader consumes soft states. Training combines target CE and forward KL from full-context predictions. | Directly challenges novelty of a small native writer for frozen Qwen. Its role allocation is user/assistant compression rate, not runtime def/use/expected/actual binding. Compression budget varies with context length. |
| [PISCO, 2025](https://arxiv.org/abs/2501.16075) | Uses sequence-level distillation of full-document answers to learn compressed RAG context. | A lighter alternative to vocabulary-wide KL: train on verified teacher answer/patch sequences. Do not assume all of its training keeps the decoder frozen. |
| [Memorization–Utilization Gap, ACL 2026](https://aclanthology.org/2026.acl-long.682/) | Separates compression fidelity from downstream use and trains with pretraining, SFT and outcome RL. | A reconstruction or semantic probe pass need not imply task use. Its recipe updates model components beyond our tiny frozen-receiver setup. RL is an option later, not a necessary first response to our failure. |

[Gist, NeurIPS 2023](https://arxiv.org/abs/2304.08467) and
[AutoCompressors, EMNLP 2023](https://arxiv.org/abs/2305.14788) additionally establish cached
activation/summary-state compression. They adapt the consuming language model, so their
results should not be represented as proof that an untouched decoder reads arbitrary vectors.

### Novelty alert: latent communication for repair

The official [ASE 2026 program entry for LaMAR](https://conf.researchr.org/details/ase-2026/ase-2026-research-track/127/LaMAR-Latent-Multi-agent-Collaboration-via-KV-Cache-Communication-for-Automated-Prog)
lists *Latent Multi-agent Collaboration via KV-Cache Communication for Automated Program Repair*.
As of this review, the retrieved entry and title searches did not provide an accessible full
method description. The conference is scheduled for October 2026. This verifies a serious
overlap in title/application, not that its encoder uses debugger traces or that it resolves our
specific role-binding problem. A first-KV-communication-for-repair claim is unsafe; detailed
comparison must await the full paper/artifact.

## 4. Numeric, relational, and execution generalization

[Neural Execution Engines, NeurIPS 2020](https://arxiv.org/abs/2006.08084) examines two distinct
problems: attention fidelity when sequences grow, and representation of unseen numbers.
It proposes conditional masking for the first and learned binary number representations for
the second. This supports testing value representations and relational masks explicitly;
it does not guarantee arbitrary integer extrapolation or Python repair transfer.

[CLRS, 2022](https://arxiv.org/html/2205.15659v2) assigns probes explicit stages, locations and
types: scalars, categories, masks and pointers, with intermediate algorithmic hints. Ordered
items and references are represented structurally, rather than by a sentence embedding of
an entire state. This is a strong template for typed runtime records and auxiliary tasks.

[IPA-GNN, NeurIPS 2020](https://arxiv.org/abs/2010.12621) couples recurrent state propagation
to control-flow structure and instruction-pointer attention for systematic execution
generalization. The lesson is to align information flow with execution dependencies. Since
we already execute the program, our encoder need not relearn a complete interpreter.

[NALU, 2018](https://arxiv.org/abs/1808.00508) and
[Neural Arithmetic Units, ICLR 2020](https://arxiv.org/abs/2001.05016) provide arithmetic
inductive biases; the latter also analyzes optimization difficulties of the former. They
are candidates if arithmetic remains the measured bottleneck, not a universal substitute
for identifiers, order, pointers, strings or control dependencies.

[TRACED, 2023](https://arxiv.org/abs/2306.07487) and
[CodeExecutor, 2023](https://arxiv.org/abs/2305.05383) already pretrain execution-aware code
representations; CodeExecutor uses mutation augmentation and a curriculum. Such semantic
pretraining is precedented. We still need to show what it preserves and whether the repair
receiver uses it.

## 5. Concrete changes supported by this review

These are proposals inferred from the papers and our artifacts, not reported prior results
on our benchmark.

1. **Audit sufficiency before another architecture run.** Evaluate four separate splits:
   short/in-range, long/in-range, short/new values, long/new values. Include a train-set check
   and more than one repair evidence bundle per behavior. The expanded run jointly changes
   bundle size, item length and value distributions; its cosine cannot isolate a cause.

2. **Keep binding information before compression.** Replace whole-event mean embeddings with
   typed fields: event/role, source node, variable/object/version, test ID, time/item index,
   exact value representation, executed branch outcome and predecessor references. First
   compare an order-preserving contextual Qwen encoder against the current input-embedding
   mean. Independently test a general numeric/sequence representation. Do not silently
   encode case-specific facts such as 'sum' or 'last' into features.

3. **Optimize patch distinctions through the frozen receiver.** Retain code distillation as
   initialization or auxiliary loss, then use patch NLL and paired opposite-patch ranking.
   For same visible context C, evidence A/B and corresponding patches P_A/P_B, score each
   candidate under each evidence. Require A to favor P_A and B to favor P_B. A candidate loss
   is `softplus(m - s(P_A|C,Z_A) + s(P_B|C,Z_A))` plus the reversed B term. Length-normalize
   scores and audit the differing patch tokens: shared function boilerplate can dominate
   whole-sequence loss. This proposed objective directly measures the desired identity use.

4. **Use counterfactuals that differ only in the intended binding.** Generate both sides of
   a pair from the *same inputs and buggy executions*, changing expected behavioral evidence.
   Include role/value/source-association swaps; preserve true pass/fail labels. The current
   script independently samples paired bundles and unconditionally writes 'test failed',
   even though some sampled executions may match their expected output. These are audit
   issues to resolve before interpreting a role-specific result.

5. **Separate runtime utility from I/O-specification utility.** On the same examples compare
   input/expected-output-only messages, actual-output deltas, and full/sliced traces. Within
   a pair with fixed buggy code and matched inputs, the buggy execution itself is identical;
   expected outputs supply the intended behavior distinction. A win here can establish
   latent behavioral-specification communication while still failing to establish added
   value from internal runtime traces. Use localization probes and realistic bugs to test
   intermediate-state utility separately.

6. **Test continuous-code degradation.** Compare predicted continuous latents with a diagnostic
   nearest-oracle-code projection. MSE can favor an average of behavior codes under ambiguity;
   a similar vector need not trigger the same autoregressive program. Projection can isolate
   code identification from receiver robustness. It remains an oracle-vocabulary diagnostic,
   not the final method or unseen-behavior solution.

## 6. Minimal next experiment and paper claim

Run a controlled 2x2 on matched ambiguous evidence:

| Representation | Vector MSE/cosine | Frozen-decoder patch loss + paired ranking |
|---|---|---|
| Current whole-event embedding mean | Reproduce current baseline | Is the objective the limiting factor? |
| Order-preserving contextual native encoding | Does richer content improve inference? | Do representation and receiver supervision jointly recover useful control? |

Keep data, slot budget, prompts and evaluation bundles fixed; assess all four transfer splits.
Then add typed numeric/pointer fields to the winning configuration. For each condition measure
true repair, paired-swap repair, both-members-of-pair success, intended opposite-behavior
selection, and the probability margin between opposite patches. Different code text alone
does not establish a semantic change. Repeat promising results across seeds.

Only after this gate should evaluation move to task-disjoint mutated MBPP/HumanEval code,
with hidden tests isolated from input construction and source/function/template overlap
audited. Match evidence for text and latent comparisons, include optimized text summaries
and slicing, and report encoder/write cost, decoder/read cost, precision, and total memory.
Eight vectors are not eight text tokens in bytes; layerwise KV has a different footprint.

A potential contribution is an interface that preserves runtime role/value/source bindings
and transfers those bindings into useful repairs in frozen small CodeLMs, demonstrated by
matched counterfactual patch selection on unseen programs. No paper in this focused review
was verified to cover that entire combination. This is a research target, not a novelty or
performance claim already supported by our pilots. The broad ideas 'embed traces', 'small
adapter into frozen LLM', and 'latent/KV communication for repair' are already occupied or
have substantial overlap.
