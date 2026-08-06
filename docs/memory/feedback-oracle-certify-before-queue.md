---
name: feedback-oracle-certify-before-queue
description: "RULE (Hajdu, 2026-07-05 03:30, after a wasted overnight launch): NEVER queue an RL training run without oracle-certifying its TRAINING reward first (reward_oracle.certify, ms-fast). Also: match the reward to the TEACHER's strategy, and do not chat per monitor-event."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d2ccb45c-9c6f-4422-a725-08dd14fe9109
---

On-record failure (2026-07-05 ~02:50): launched a 3-seed × 200k TD3+BC overnight with `--variant baseline`,
whose training reward `galambos_task.hymeko` was ALREADY documented (in `galambos_task_deliver.hymeko`'s
header and certified by `reward_oracle.certify` → `delivers=False`) as a farming reward whose optimum does
not deliver. 135k wasted steps before the user called it. §6.5 #15 exactly: the artifact existed and was not
consulted at launch time.

**Why:** the oracle costs milliseconds; a wrong overnight costs hours plus the user's trust and tokens.

**How to apply:**
1. Before ANY `Campaign`/`train_offpolicy`/PPO queue: `certify(RewardSpec.from_hymeko(<training reward>))`
   must return `delivers=True`, quoted in the launch message. No exceptions for "it's just the baseline arm."
2. Reward–teacher alignment check: if the BC teacher's strategy changed (e.g. pinch → push-controller), re-ask
   whether contact/grasp-shaped terms now FIGHT the anchor. Dense terms aligned with the OLD strategy destroy
   the clone (measured: peak 0.12 vs BC floor 0.34, both_contact → 0).
3. The Campaign must always evaluate the step-0 (post-BC) policy so the warm-start floor is in the
   best-checkpoint race (fixed 2026-07-05 in `campaign.py`).
4. Token discipline: do NOT reply per monitor event; arm monitors on final-verdict/error patterns only and
   report at completion.

Related: [[project-galambos-reward-fixed-rl-below-demo]], [[feedback-baseline-once-parallel-ab]].
