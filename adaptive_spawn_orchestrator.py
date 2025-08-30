
"""
Adaptive Spawning Orchestrator (Concept)
- Dynamically spawns specialized agents when quality is low or gaps are detected.
- Builds on the minimal Agent + generation utilities from multi_agent_bfs.py

Usage:
  python adaptive_spawn_orchestrator.py --idea "Ship a simple web demo that fetches live weather" --model-name gpt2 --max-rounds 4 --max-agents 8 --spawn-threshold 7

Note: This is still concept code (demo-level). Not production-ready.
"""
from __future__ import annotations
import re
import argparse
from typing import List, Dict, Optional

from multi_agent_bfs import (
    Agent, GenConfig, load_model_and_tokenizer, setup_device_and_seed,
)

# ----------------------------
# CLI
# ----------------------------

def build_args():
    p = argparse.ArgumentParser(description="Adaptive spawning multi-agent concept")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--idea", type=str, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--spawn-threshold", type=int, default=7, help="If score < threshold, consider spawning")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ----------------------------
# Spawn policy & helpers
# ----------------------------

SKILL_MAP: Dict[str, List[str]] = {
    # keyword -> suggested specialized roles
    "data": ["researcher", "data_cleaner"],
    "dataset": ["researcher", "data_cleaner"],
    "api": ["api_integrator"],
    "integration": ["api_integrator"],
    "ui": ["ux_writer"],
    "ux": ["ux_writer"],
    "test": ["tester"],
    "deploy": ["devops"],
}

ALLOWED_SPECIALISTS = {
    "researcher", "critic", "evaluator", "api_integrator", "ux_writer",
    "data_cleaner", "devops"
}

def parse_score(text: str) -> Optional[int]:
    """Find an integer score like 'Score: 6/10' or '7/10' in text."""
    m = re.search(r"(Score\\s*:\\s*)?(\\d{1,2})\\s*/\\s*10", text)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None

def infer_missing_roles(critic_text: str) -> List[str]:
    """Very simple heuristic: map detected keywords to roles."""
    critic_l = critic_text.lower()
    suggested: List[str] = []
    for k, roles in SKILL_MAP.items():
        if k in critic_l:
            for r in roles:
                if r not in suggested:
                    suggested.append(r)
    # Also allow critic to explicitly say: "spawn: researcher, api_integrator"
    m = re.search(r"spawn\\s*:\\s*([a-z_,\\s-]+)", critic_l)
    if m:
        for tok in m.group(1).split(','):
            r = tok.strip().replace('-', '_')
            if r and r.isalpha() and r not in suggested:
                suggested.append(r)
    return [r for r in suggested if r in ALLOWED_SPECIALISTS]

# ----------------------------
# Orchestrator
# ----------------------------

class Orchestrator:
    def __init__(self, model, tokenizer, device, gen_cfg: GenConfig, max_agents=8, debug=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gen_cfg = gen_cfg
        self.max_agents = max_agents
        self.debug = debug
        self.agents: List[Agent] = []

    def add_agent(self, role: str):
        if len(self.agents) >= self.max_agents:
            return False
        name = f"Agent{len(self.agents)+1}"
        self.agents.append(Agent(name, role, debug=self.debug))
        return True

    def get(self, role: str) -> Optional[Agent]:
        return next((a for a in self.agents if a.role == role), None)

    def step_round(self, context: str) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        for a in list(self.agents):
            style = "one_sentence"
            instr = {
                "planner": "Return one concrete requirement (<=25 words).",
                "architect": "Describe the approach in one sentence (<=25 words), no code.",
                "implementer": "Return one actionable next step (<=18 words), imperative.",
                "tester": "Return one acceptance check (<=18 words), imperative.",
                "researcher": "State one missing fact or data source needed (<=18 words).",
                "api_integrator": "Name one external API/service to integrate and why (<=18 words).",
                "ux_writer": "Write one short UX microcopy requirement (<=18 words).",
                "data_cleaner": "State one data quality action required (<=18 words).",
                "devops": "State one deployment/reliability step (<=18 words).",
                "critic": "Name one critical gap or risk (<=18 words).",
                "evaluator": "Give a concise 0–10 score like 'Score: X/10' (one reason).",
            }.get(a.role, "Return one short, concrete improvement suggestion (<=18 words).")

            out = a.think(
                self.model, self.tokenizer, self.device,
                context=context, instruction=instr,
                gen_cfg=self.gen_cfg, style=style
            )
            outputs[a.role] = out
        return outputs

    def maybe_spawn(self, critic_text: str, score_text: str):
        # Parse evaluator score
        score = parse_score(score_text or "")
        # Map critic text to suggested specialist roles
        want = infer_missing_roles(critic_text or "")
        spawned = []
        for r in want:
            if len(self.agents) >= self.max_agents:
                break
            if self.get(r) is None:
                if self.add_agent(r):
                    spawned.append(r)
        return score, spawned

def run(args):
    device = setup_device_and_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Bootstrap team
    orch = Orchestrator(model, tokenizer, device, gen_cfg, max_agents=args.max_agents, debug=args.debug)
    for r in ["planner", "architect", "implementer", "tester", "critic", "evaluator"]:
        orch.add_agent(r)

    context = args.idea.strip()
    print(f"[INFO] Goal: {context}")

    for round_id in range(1, args.max_rounds + 1):
        print(f"\n=== ROUND {round_id} ===")
        outs = orch.step_round(context)
        for role, text in outs.items():
            print(f"{role:12s}: {text}")

        critic_text = outs.get("critic", "")
        evaluator_text = outs.get("evaluator", "")
        score, spawned = orch.maybe_spawn(critic_text, evaluator_text)
        if spawned:
            print(f"[SPAWN] Added specialists: {', '.join(spawned)}")
        if score is not None:
            print(f"[EVAL] Score parsed: {score}/10")

        # Simple stop condition: high score
        if score is not None and score >= args.spawn_threshold + 2:
            print("[STOP] Quality high enough; stopping early.")
            break

    # Final summary
    summ = orch.get("evaluator") or orch.get("critic")
    if summ:
        out = summ.think(
            model, tokenizer, device,
            context=context,
            instruction="Summarize status in five bullets (<=18 words each).",
            gen_cfg=gen_cfg, style="bullets", bullets=5
        )
        print("\n[SUMMARY]\n" + out)

if __name__ == "__main__":
    args = build_args()
    run(args)
