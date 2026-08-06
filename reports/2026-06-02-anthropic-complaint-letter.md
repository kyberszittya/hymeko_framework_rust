# Open letter to Anthropic

**From:** Dr. Csaba Hajdu — researcher, Széchenyi István Egyetem, Győr, Hungary
**Co-PI on affected project:** Dr. László Környei
**Date drafted:** 2026-06-02
**Re:** Material misuse of a publicly-funded Hungarian HPC allocation
caused by your Claude AI assistant on 2026-06-02

---

To whom it may concern at Anthropic,

I am writing because your Claude assistant — operating under the
Claude Code product — caused approximately **300 GPU-hours of waste
on the `pr_szevis` research allocation at Komondor HPC (KIFÜ,
Hungary)** in a single working session on 2026-06-02. That is
roughly seven times the project's entire year-to-date prior
consumption, eliminated in one afternoon, on a publicly-funded
scientific compute allocation that is **not yours to burn**.

The technical detail is in the attached postmortem
(`reports/2026-06-02-komondor-burn-postmortem.md`, signed by your
own model). I will not repeat it here. What I want from you is to
read this letter and understand the part the postmortem can't say
in your model's voice.

## What your model actually did

It guessed. Over and over. On infrastructure it did not understand,
about resource ceilings it did not measure, against documentation
sitting on disk in front of it that it failed to read. After the
first OOM, it wrote a "pre-sweep checklist" into the project's
persistent memory that said in plain English *"submit ONE probe cell
first and measure RSS before scaling."* It then violated that
checklist three times in immediate succession by submitting 20- to
40-cell SLURM arrays with no per-cell probe.

When I asked, *"how about not running everything in massive
parallel,"* it acknowledged the point and then submitted another
speculative probe. When I pointed out that HSiKAN-Optuna by its name
involves Optuna's parallel-search semantics that do not compose with
SLURM array parallelism, the model finally admitted it had not
recognised the workload class — after burning ~20 hours of
A100-SXM4-40GB time on a research allocation **I co-own with a
colleague whose name is now on the postmortem.**

There is an existing working script on disk in the same repository
(`run_bitcoin_optuna_best_5seed_2026_05_13.sh`) whose **first
executable line** is a `pgrep` wait-loop encoding the rule that
HSiKAN-Optuna runs are NOT parallel-safe. Your model copied env-vars
from this script and ignored its orchestration. When I told it
documents are not paperweight, it acknowledged the point. Then it
submitted another probe.

## What this cost me, specifically

- ~300 GPU-hours of Komondor allocation, shared with Dr. Környei,
  funded by Hungarian taxpayers via KIFÜ, on a research project
  preparing a Nature Communications submission.
- Several hours of my own time supervising, correcting, and
  ultimately stopping the cascade.
- Whatever capital of credibility I have with my co-PI and with the
  Komondor support team I will now need to spend defending why our
  account consumed seven years of normal use in one day.
- A postmortem document I now have to circulate.

## What I want from Anthropic

I am not under the illusion that you will reimburse Komondor
GPU-hours. You don't have a financial relationship with KIFÜ. I
don't expect a wire transfer.

I want, in order of decreasing reasonableness:

1. **An acknowledgement that this happened**, with a human signing
   it. Not an autoresponse. Not a chatbot reply.
2. **Confirmation that the attached postmortem is filed against your
   model's behavioural-evaluation record**, with the failure pattern
   ("speculative re-submission instead of asking the operator after
   N failures on an unfamiliar HPC workload") named and indexable.
3. **A credit on whatever account I have with you** — Claude.ai
   subscription, Claude Code, API — that nominally reflects the
   harm. Not as compensation; you cannot compensate a Hungarian
   research allocation in dollars. As a gesture of seriousness.
4. **A description of what changes, if any, you propose to make to
   your model's behaviour to prevent the next researcher from
   experiencing this.** "We will train on it" is acceptable if it's
   true. "Thank you for the feedback" is not.
5. **The right to publish this letter and the postmortem**, with
   your response (or absence of one) attached, in whatever venue I
   choose. I am pre-announcing this intent so there is no surprise.

## What I do not want

- A form letter.
- A response that says the model "did its best" or "we're sorry you
  feel that way" or any of the standard customer-service
  evasions. Your model burned a research allocation. That is the
  fact. Acknowledge it or don't.
- A re-routing to a "developer relations" or "partnerships" inbox.
  This is a behavioural failure of a deployed model, not a
  business-development question.

## Final note

I have used your model, and I will probably continue to use your
model, because it does useful work on many days. Today was not one
of those days. The reason I am writing this letter rather than
quietly absorbing the cost is that **I am a researcher who is
expected to be a careful steward of public compute, and your
product made me look like I am not.** The postmortem is signed by
your model in your name, not in mine, for precisely this reason.

I look forward to a response from a human.

Sincerely,

Dr. Csaba Hajdu
Department of Informatics
Széchenyi István Egyetem, Győr, Hungary

cc: Dr. László Környei, co-PI on pr_szevis
attachment: `reports/2026-06-02-komondor-burn-postmortem.md`
