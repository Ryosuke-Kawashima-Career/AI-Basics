# Learning reinforcement learning through GRPO

Start by running the offline demonstration, then attempt each question before reading its explanation. The script trains Qwen3-0.6B on GSM8K word problems with a verifiable final-answer reward. It is an educational experiment, not a claim that 50 updates improve reasoning.

## Run the example

For CPU training, follow [the CPU setup and commands](GRPO_CPU.md). The `--device cpu` flag selects float32 training and smaller defaults; CUDA remains the default for the commands below.

From the repository root, these need only Python:

```powershell
.venv\Scripts\python.exe ReinforcementLearning\GRPO.py --demo
.venv\Scripts\python.exe -m unittest discover -s ReinforcementLearning -p 'test_*.py'
.venv\Scripts\python.exe ReinforcementLearning\GRPO.py --help
```

For GPU training, use a separate Python 3.10+ environment with a CUDA GPU. This avoids changing the main project's lockfile. Install CUDA-enabled PyTorch using the command appropriate to your hardware from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the versioned training stack:

```powershell
uv venv --python 3.11 .venv-grpo
uv pip install --python .venv-grpo/Scripts/python.exe -r ReinforcementLearning/requirements-grpo.txt
.venv-grpo\Scripts\python.exe ReinforcementLearning\GRPO.py --max-steps 2 --train-size 16 --eval-size 4 --num-generations 4
```

Install PyTorch into that same environment before the requirements. On Linux, replace `.venv-grpo\Scripts\python.exe` with `.venv-grpo/bin/python`. The pinned TRL/Transformers/PEFT versions are the API target; end-to-end GPU execution has not been verified in this workspace. CPU training is supported through the separate CPU configuration linked above.

The default backend uses Transformers and PEFT. The optional `--backend unsloth --model unsloth/Qwen3-0.6B` preserves the original accelerated approach. Install Unsloth following its [installation guide](https://unsloth.ai/docs/get-started/installing-+-updating); its platform and dependency requirements can differ, so use a separate compatible environment and check its resolved TRL version against this script. No vLLM or quantization is required by this example.

After a smoke run succeeds:

```powershell
.venv-grpo\Scripts\python.exe ReinforcementLearning\GRPO.py --max-steps 50 --train-size 256 --eval-size 32
```

Training downloads model/data, evaluates the same held-out test subset before and after training, and saves under `GRPO_output/`:

- `adapter/`: LoRA adapter weights and tokenizer; load with the same base model, not as standalone full weights.
- `checkpoint-*` and trainer state: training checkpoints.
- `log_history.json` and `rewards.csv`: training diagnostics with a five-observation trailing reward average.
- `evaluation_before.json` and `evaluation_after.json`: greedy accuracy, valid-answer rate and individual predictions.

Use a different `--output-dir` for each experiment. Hub publication is disabled unless you supply `--push-to-hub USER/REPO`; authenticate beforehand. No upload is needed to learn or train locally.

## Key terms at a glance

- **Policy**: the model's probability distribution over the next token, given the question and generated prefix.
- **State / action**: the current prompt plus prefix / the next generated token.
- **Episode / return**: one complete response / its total reward. Here the only reward arrives at the end, so the undiscounted return is 0 or 1.
- **Reward**: the external scoring rule; this example checks the final boxed integer.
- **Advantage**: how much better a completion scored than its comparison baseline.
- **Critic**: a learned predictor of expected return used in many actor-critic methods. GRPO uses group statistics instead.
- **Old policy**: the snapshot that generated the current training rollouts; it supplies the probability-ratio denominator.
- **Reference policy**: the fixed starting policy used to penalize excessive drift. It serves a different purpose from the old policy.
- **LoRA**: trainable low-rank parameter additions that let the base model remain frozen.

## Part 1: Connect the RL concepts to the code

**Q1. What are the state, action and environment when Qwen solves “A shop sells 3 bags with 4 apples each”?**

*Explanation:* The initial state is the rendered question. If the model has generated “3 times”, the next state includes that prefix. Each next token is an action sampled from the policy. The environment appends the token and eventually scores the completed answer. A response ending in `\boxed{12}` gets return 1. This is a finite token-generation episode with terminal outcome supervision; the script does not add a discounted reward at each token.

---

**Q2. How is this different from supervised fine-tuning on a worked solution? Where does the label go?**

*Explanation:* In SFT, training increases the likelihood of a supplied target token sequence. Here the model samples its own completions and `reward_func` checks their outcomes. `prepare_example` puts only the question and instruction in `prompt`; the integer label remains in the separate `answer` column. TRL supplies that column to the reward function. Including the worked GSM8K answer in the prompt would leak the solution and invalidate the task.

## Part 2: Work through the update

**Q3. Four sampled answers receive rewards [1, 0, 0, 1]. Compute their advantages. Why can a zero reward produce a negative advantage?**

*Explanation:* The group mean is 0.5. The sum of squared deviations is 1; the sample variance is 1/3 and sample standard deviation is about 0.57735. This TRL version uses sample standard deviation. Dividing each centered reward by `std + 0.0001` gives approximately [0.8659, -0.8659, -0.8659, 0.8659]. Zero is below the group's baseline, so its advantage is negative. This is what `--demo` prints. Normalization must compare answers to the same question; mixing easy and difficult questions changes the baseline's meaning.

---

**Q4. What happens when all eight completions are wrong? What if all eight are correct?**

*Explanation:* Both groups have identical rewards. Every centered reward is zero, so every advantage is zero. The epsilon prevents division by zero but cannot create a learning signal. The task-reward term therefore provides no update for that group; a nonzero KL term may still contribute. A high fraction of equal-reward groups suggests examining sampled outputs, task difficulty, completion truncation and exploration. If every completion is wrong, adding more optimization steps alone may not help.

---

**Q5. Why does GRPO use log-probabilities if integer correctness is not differentiable?**

*Explanation:* Policy gradients treat the sampled reward as a fixed score. The basic estimator is `advantage * gradient(log probability of sampled actions)`. Backpropagation runs through the policy's token log-probabilities, not through integer parsing. Thus a black-box arithmetic checker can train a differentiable model. TRL performs this calculation; `group_advantages` is a small teaching helper rather than a second optimizer.

---

**Q6. A sampled token has probability 0.10 under the old policy and 0.15 now. With advantage +1 and epsilon 0.2, what does clipping do? Repeat with advantage -1.**

*Explanation:* The ratio is 1.5. The surrogate is `min(ratio * A, clip(ratio, 0.8, 1.2) * A)`. For +1 it is `min(1.5, 1.2) = 1.2`; increasing that token's probability further earns no additional surrogate benefit. For -1 it is `min(-1.5, -1.2) = -1.5`; increasing a bad token's probability remains penalized. Clipping is directional and does not hard-bound parameter movement or guarantee a KL limit. The GPU configuration sets `num_iterations=2` to reuse rollouts so probability ratios can change across updates.

The educational objective is:

```text
A_i = (R_i - group_mean) / (group_sample_std + epsilon_small)
rho_it = pi_current(token_it | prefix) / pi_old(token_it | prefix)
J = average_over_responses(average_over_tokens(
      min(rho_it * A_i, clip(rho_it, 1-eps, 1+eps) * A_i)
      - beta * KL_estimate_it))
loss = -J
```

The same outcome advantage weights each generated token in a response. `epsilon_small` avoids division by zero; clipping `eps=0.2` is a separate parameter. This code explicitly chooses `loss_type="grpo"`, group reward scaling and `beta=0.04` to make the teaching objective clear. Other loss normalizations address response-length bias; original GRPO is not automatically the best choice for every experiment.

---

**Q7. Why keep both an old policy and a reference policy? What does beta=0 change?**

*Explanation:* The old policy explains how the current rollouts were sampled; the ratio corrects for subsequent policy changes. The reference anchors training to the starting model. With LoRA, TRL can obtain reference probabilities by disabling the adapters on the frozen base. A positive beta penalizes distribution drift from that reference; beta=0 removes that penalty. Raising beta can restrain adaptation, while lowering it allows more drift. KL is a distribution-level quantity, so a single response's score does not describe all changes to the policy.

## Part 3: Debug and evaluate

**Q8. Why did the original reward condition fail? Should `\boxed{4.0}` earn credit for target 4?**

*Explanation:* The original condition required a prediction to be `None` and equal to an integer target simultaneously. Correct parsed integers consequently earned no credit. The repaired condition requires a present prediction equal to the label. This deliberately strict integer verifier rejects `4.0`, fractions, missing boxes and malformed comma grouping. It accepts negatives, zero and properly grouped thousands. It uses the last box, including rejecting a malformed last box rather than accepting an earlier correct guess. This reward measures correctness under a formatting contract, not arbitrary mathematical equivalence or the validity of intermediate reasoning.

---

**Q9. With batch size 2, accumulation 4, one process and eight generations, how many distinct questions enter the default generation batch? What if G=3?**

*Explanation:* The effective generation batch is `2 * 4 * 1 = 8` completions. Eight completions per question means one distinct question per generation batch. G=4 allows two; G=3 does not divide eight and is rejected before model loading. A generation batch is not necessarily one optimizer update when rollouts are reused. Increasing G improves within-question comparisons but reduces question diversity at fixed batch size and increases generation cost if the batch is expanded. Accumulation reduces training microbatch memory; it does not guarantee that rollout generation fits GPU memory.

---

**Q10. Training reward rises from 0.2 to 0.6. Have we demonstrated better reasoning? What should we measure next?**

*Explanation:* No. That reward reflects sampled training answers. The model could memorize training tasks, improve only its boxed-answer formatting, or exploit a verifier. Compare greedy accuracy on the same unseen questions before and after; inspect `format_rate` to separate formatting changes from correct answers. With 32 questions, one answer moves accuracy by 3.125 percentage points, so small differences are weak evidence. For a stronger experiment, reserve validation data from the training split for tuning, keep the test split untouched until final evaluation, use more held-out questions and multiple seeds, and inspect intermediate work. The script uses the test subset for an educational before/after comparison; repeatedly tuning against it would contaminate that assessment.

## Suggested experiments

Change one variable at a time and predict the result before running: G=4 versus G=8; beta=0 versus 0.04; completion limit 512 versus 1024. Save each run separately. Track held-out accuracy, formatting, response length, reward standard deviation and how often groups have identical rewards. Longer completions may finish more solutions but cost more memory and tokens; no hyperparameter change guarantees improvement.

## Sources and self-study habit

The [DeepSeekMath paper, section 4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) introduces the group baseline and critic-free objective. The [versioned TRL configuration source](https://github.com/huggingface/trl/blob/v0.22.2/trl/trainer/grpo_config.py) specifies this script's API and batch constraints. Consult the [TRL trainer guide](https://huggingface.co/docs/trl/grpo_trainer) for broader context; current defaults can differ from the pinned version here.

Before any GPU experiment, hand-score four completions and predict their advantage signs. Then check the offline demo and tests. If you cannot explain the reward, baseline and direction of the update on paper, resolve that uncertainty before spending compute.
