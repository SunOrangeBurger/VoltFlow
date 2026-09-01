"""Manual multi-step trace of cell temperature / thermal penalty behavior.

Added during local setup after the first `VoltFlowEnv.step()` smoke test
showed t_cell_k jumping from ~298K (ambient) to 322.32K -- already past
t_crit (318.15K) -- in a single 15-min step at action=0.5. Before spending
compute on a full PPO run, this script checks whether that's a one-step
transient (temp settles toward some physically sane equilibrium under the
analytic thermal solution) or an unbounded/runaway climb (a real bug,
likely in thermal.rs's h*A / C_thermal parameters or the reward's kappa
weighting -- see STATUS.md "Known Specification Deviations" for the ODE fix
this builds on).

Also prints obs[2] -- the *normalized* T_cell observation fed to the PPO
policy -- alongside the raw t_cell_K/C from info. These are two different
things: t_cell_k/thermal_penalty/reward come straight from the `info`
dict (raw physics, unaffected by the observation-normalization fix below),
while obs[2] comes from `compute_observation()` (what the agent actually
sees, and IS affected by that fix). After the fix documented in
STATUS.md (see "Known Specification Deviations"), obs[2] should track smoothly up toward 1.0
rather than saturating as soon as t_cell crosses ~60C (333.15K), which is
what the old spec-literal /60.0 divisor did. The direct check: compare
obs[2] between the action=0.5 trace (peaks ~67C) and the action=0.8 trace
(clamped at 100C) below -- if the fix is in place these should read
noticeably different, not both ~1.0000.

Not a pytest / automated check -- this is a human-in-the-loop diagnostic.
Read the printed trace and eyeball whether t_cell_K plateaus, and whether
obs[2] tracks it without prematurely saturating.

Usage:
    python python/voltflow/scripts/trace_thermal.py \
        --csv data/raw/energy_weather_spain.csv --steps 30
"""

from __future__ import annotations

import argparse

from voltflow.envs.gym_wrapper import VoltFlowEnv


def run_trace(csv_path: str, action: float, steps: int) -> None:
    env = VoltFlowEnv(csv_path, max_steps=steps)
    obs, _ = env.reset()
    print(f"\n--- constant action = {action:+.2f} ---")
    print(
        f"{'step':>4}  {'t_cell_K':>9}  {'t_cell_C':>9}  {'obs[2]':>7}  "
        f"{'thermal_pen':>12}  {'reward':>8}  {'soc':>6}"
    )
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step([action])
        print(
            f"{i:>4}  {info['t_cell_k']:>9.2f}  {info['t_cell_k'] - 273.15:>9.2f}  "
            f"{obs[2]:>7.4f}  {info['thermal_penalty']:>12.2f}  {reward:>8.3f}  "
            f"{info['soc']:>6.3f}"
        )
        if terminated or truncated:
            print(f"  (episode ended at step {i}: terminated={terminated} truncated={truncated})")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/energy_weather_spain.csv")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--actions",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 0.8],
        help="One trace per action value, each run from a fresh reset.",
    )
    args = parser.parse_args()

    for action in args.actions:
        run_trace(args.csv, action, args.steps)

    print(
        "\nWhat to look for (physics): does t_cell_K plateau (approach some "
        "fixed equilibrium) as steps increase, or does it climb without "
        "bound? A plateau under the analytic thermal solution is "
        "expected/correct. Unbounded growth, or every nonzero action "
        "instantly blowing past t_crit (318.15K) and staying there, "
        "suggests the thermal params (h*A / C_thermal in ThermalParams) "
        "or the reward's kappa weighting need re-checking before trusting "
        "a PPO run built on top of this."
        "\n\nWhat to look for (observation fix): obs[2] should read "
        "noticeably different between the action=0.5 trace (~67C peak) "
        "and the action=0.8 trace (clamped at 100C) -- NOT both reading "
        "~1.0000. If they're indistinguishable, the widened normalization "
        "either wasn't rebuilt (`maturin develop --release` after editing "
        "simulation.rs) or didn't take effect."
    )


if __name__ == "__main__":
    main()