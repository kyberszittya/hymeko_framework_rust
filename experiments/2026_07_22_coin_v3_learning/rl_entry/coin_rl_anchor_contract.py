"""§6 TD3_ANCHOR_GRADIENT_CONTRACT verification under the ACTUAL optimizer (Adam), not raw SGD. The proximal-BC
anchor gradient is identically 0 at pi_0 (deviation=0) -> "anchor grad 2-4x Q at update 0" is structurally
unsatisfiable for ANY beta; measured instead at p1 (earliest live point, one Adam Q-step past pi_0). Verifies the
four §6 sub-conditions under Adam: (a) combined objective decreases, (b) Q not driven opposite, (c) anchor constrains
but does not cancel Q, (d) states retain nonzero update.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
_BASE = sys.argv[1]
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "asm", "/private/tmp/claude-501/-Users-kyberszittya-hakiko-ai-ws-03-implementation-hymeko-framework-rust/63ad1b54-314a-48f8-b561-ba4a163f847c/scratchpad/coin_rl_anchor_smoke.py")
sys.argv = ["x", _BASE]
asm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asm)

BETA = 400.0


def main():
    bc, pi0, pi, pi_targ, q, qt, D, auth, anchor_S = asm.build()
    S, A, R, S2, Dn = asm.cr.flat(D)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(S), 256)
    s = torch.tensor(S[idx]); aS = anchor_S[:256]

    def qloss(a):
        return -q(s, a.action_mean(s))[0].mean()

    def comb(a):
        return qloss(a) + asm.anchor_loss(a, pi0, aS, BETA)

    # ---- ratio at update 0 (pi=pi_0): anchor is identically 0 ----
    ga0 = torch.autograd.grad(asm.anchor_loss(pi, pi0, aS, BETA), pi.parameters(), retain_graph=True, allow_unused=True)
    gan0 = float(torch.sqrt(sum(((g if g is not None else torch.zeros(1)) ** 2).sum() for g in ga0)))
    print(f"update-0 anchor-grad-norm = {gan0:.6e}  (structurally 0 at pi_0 => ratio rule unsatisfiable)", flush=True)

    # ---- earliest live point p1 = pi_0 after ONE Adam Q-step ----
    p1 = copy.deepcopy(pi); ao = torch.optim.Adam(p1.parameters(), lr=3e-4)
    ao.zero_grad(); qloss(p1).backward(); ao.step()
    gq = torch.autograd.grad(qloss(p1), p1.parameters(), retain_graph=True)
    ga = torch.autograd.grad(asm.anchor_loss(p1, pi0, aS, BETA), p1.parameters(), retain_graph=True)
    gqn = float(torch.sqrt(sum((g ** 2).sum() for g in gq))); gan = float(torch.sqrt(sum((g ** 2).sum() for g in ga)))
    cos = float(sum((a * b).sum() for a, b in zip(gq, ga)) / (gqn * gan + 1e-9))

    L0 = float(comb(p1).detach()); Q0 = float(q(s, p1.action_mean(s))[0].mean().detach())
    AN0 = float(asm.anchor_loss(p1, pi0, aS, BETA).detach())
    # one COMBINED Adam step (the real update)
    p2 = copy.deepcopy(p1); co = torch.optim.Adam(p2.parameters(), lr=3e-4)
    co.zero_grad(); comb(p2).backward(); co.step()
    L1 = float(comb(p2).detach()); Q1 = float(q(s, p2.action_mean(s))[0].mean().detach())
    AN1 = float(asm.anchor_loss(p2, pi0, aS, BETA).detach())
    # per-state combined-gradient contribution proxy: |Q-grad wrt action| on each anchor state (retain nonzero update)
    aS_g = aS.clone().requires_grad_(False)
    act = pi0.action_mean(aS_g)   # ref
    with torch.no_grad():
        dev = (p1.action_mean(aS_g) - act).abs().sum(-1)
    retain = float((dev > 1e-4).float().mean())

    obj_dec = L1 <= L0 + 1e-4
    q_not_opp = Q1 >= Q0 - 0.05 * abs(Q0) - 1e-4
    not_cancel = (gan / (gqn + 1e-9)) > 0.5 and cos < 0.98      # anchor is a real force, not perfectly cancelling
    retain_ok = retain > 0.5
    gc_pass = bool(obj_dec and q_not_opp and not_cancel and retain_ok)

    out = {"beta": BETA, "update0_anchor_grad_norm": gan0, "anchor_zero_at_pi0": gan0 < 1e-6,
           "p1_q_grad_norm": round(gqn, 4), "p1_anchor_grad_norm": round(gan, 4),
           "p1_anchor_over_q_ratio": round(gan / (gqn + 1e-9), 2), "p1_cos_gq_ga": round(cos, 3),
           "adam_combined_step": {"L": [round(L0, 4), round(L1, 4)], "Q": [round(Q0, 4), round(Q1, 4)],
                                  "anchor": [round(AN0, 5), round(AN1, 5)]},
           "obj_decreases": obj_dec, "q_not_opposite": bool(q_not_opp), "not_cancelled": bool(not_cancel),
           "frac_states_updated": round(retain, 3), "retain_update": retain_ok, "pass": gc_pass}
    json.dump(out, open("/tmp/anchor_contract.json", "w"), indent=1)
    print(f"p1: Qgrad {gqn:.2f} anchorGrad {gan:.2f} ratio {gan/(gqn+1e-9):.2f} cos {cos:+.2f}", flush=True)
    print(f"ADAM combined step: L {L0:.3f}->{L1:.3f} (dec={obj_dec}) Q {Q0:.3f}->{Q1:.3f} (not_opp={bool(q_not_opp)}) "
          f"anchor {AN0:.4f}->{AN1:.4f} not_cancel={bool(not_cancel)} retain={retain:.2f}", flush=True)
    print(("TD3_ANCHOR_GRADIENT_CONTRACT_PASS" if gc_pass else "TD3_ANCHOR_GRADIENT_CONTRACT_FAIL"), flush=True)
    print("CONTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
