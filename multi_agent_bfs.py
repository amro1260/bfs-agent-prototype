
# ============================
# Multi-Agent BFS Pipeline (Concept, improved)
# - Small, fast, no chat template required
# - Clear toggles for bullet vs sentence outputs
# - CLI args, better parsing, non-interactive mode
# ============================

from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# CLI
# ----------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny multi-agent BFS concept pipeline")
    p.add_argument("--model-name", default="gpt2", help="HF model id (causal LM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=2, help="BFS preview depth (>=0)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--idea", type=str, default=None, help="Initial idea (skip interactive input)")
    p.add_argument("--noninteractive", action="store_true", help="Force auto idea when no --idea")
    p.add_argument("--debug", action="store_true", help="Print agent debug lines")
    return p.parse_args()

# ----------------------------
# Device & reproducibility
# ----------------------------

def setup_device_and_seed(seed: int) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ----------------------------
# Model loading
# ----------------------------

def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    # Note: Avoid fp16 on MPS/CPU

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer

# ----------------------------
# Generation helpers
# ----------------------------

@dataclass
class GenConfig:
    max_new_tokens: int = 80
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True


def complete(model, tokenizer, device, prompt: str, gen_cfg: GenConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
            pad_token_id=gen_cfg.pad_token_id,
            eos_token_id=gen_cfg.eos_token_id,
            use_cache=gen_cfg.use_cache,
        )
    gen = out[0, input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def ask_llm(model, tokenizer, device, system: str, user: str, gen_cfg: GenConfig) -> str:
    # Simple guided prompting for non-instruction models
    prompt = f"System: {system}\nUser: {user}\nAssistant:"
    return complete(model, tokenizer, device, prompt, gen_cfg)

# ----------------------------
# Agents
# ----------------------------

@dataclass
class Agent:
    name: str
    role: str
    depth: int = 0
    memory_limit: int = 24
    debug: bool = False
    _mem: List[str] = field(default_factory=list)

    def learn(self, text: str):
        if not text:
            return
        for ln in [l.strip() for l in text.splitlines() if l.strip()]:
            # keep short, no URLs/backticks
            if 3 <= len(ln.split()) <= 25 and "http" not in ln and "`" not in ln:
                self._mem.append(ln)
        self._mem = self._mem[-self.memory_limit:]

    def memory_text(self) -> str:
        return " ".join(self._mem[-self.memory_limit:])

    def think(
        self,
        model,
        tokenizer,
        device,
        context: str,
        instruction: str,
        gen_cfg: GenConfig,
        style: str = "one_sentence",  # "one_sentence" or "bullets"
        bullets: int = 5,
    ) -> str:
        tail = ""
        if style == "one_sentence":
            tail = "Return a single sentence."
        elif style == "bullets":
            tail = f"Return exactly {bullets} short bullets (<=18 words each), imperative. No code."

        system = f"You are a concise {self.role}. Be concrete. No code, no URLs."
        user = (
            f"Agent: {self.name}\n"
            f"Role: {self.role}\n"
            f"Known notes: {self.memory_text()}\n\n"
            f"Context: {context}\n"
            f"{instruction}\n"
            f"{tail}"
        )
        out = ask_llm(model, tokenizer, device, system, user, gen_cfg)
        out = out.splitlines()[0].strip()
        if self.debug:
            print(f"[DEBUG] {self.name} ({self.role}) out:\n{out}\n" + "-" * 60)
        self.learn(out)
        return out

# ----------------------------
# BFS role expansion (preview)
# ----------------------------

def bfs_spawn(roles: List[str], max_depth: int) -> Dict[int, List[str]]:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if not roles:
        raise ValueError("roles must be non-empty")

    levels: Dict[int, List[str]] = {}
    q: deque[Tuple[int, str]] = deque()
    q.append((0, roles[0]))
    seen = 0

    while q:
        depth, role = q.popleft()
        levels.setdefault(depth, []).append(role)
        if depth >= max_depth:
            continue
        next_depth = depth + 1
        for i in range(len(roles)):
            child_role = roles[(seen + i) % len(roles)]
            q.append((next_depth, child_role))
        seen += 1
    return levels

# ----------------------------
# Idea source
# ----------------------------

def is_interactive_stdin() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def ask_or_generate_initial_idea(model, tokenizer, device, gen_cfg: GenConfig, provided: Optional[str], noninteractive: bool) -> str:
    if provided:
        return provided.strip()

    if not noninteractive and is_interactive_stdin():
        print("Enter your initial idea (press Enter to auto-generate):")
        try:
            user_idea = input().strip()
        except Exception:
            user_idea = ""
        if user_idea:
            return user_idea

    system = "You produce short, concrete requirement sentences. No lists. No code."
    user = "Propose one business-relevant requirement (<=25 words), concrete and testable."
    idea = ask_llm(model, tokenizer, device, system, user, gen_cfg)
    idea = idea.splitlines()[0].strip()
    print(f"[INFO] Auto-generated initial idea: {idea}")
    return idea

# ----------------------------
# Pipeline
# ----------------------------

def run_pipeline(args: argparse.Namespace):
    device = setup_device_and_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 1) initial idea
    initial_idea = ask_or_generate_initial_idea(model, tokenizer, device, gen_cfg, args.idea, args.noninteractive)
    print(f"\n[INFO] Initial idea: {initial_idea}\n")

    # 2) roles (base + optional proposals)
    base_roles = ["planner", "architect", "implementer", "tester"]
    proposer = Agent("RoleProposer", "researcher", debug=args.debug)
    proposed = proposer.think(
        model, tokenizer, device,
        context=initial_idea,
        instruction=(
            "Propose up to 3 additional roles as comma-separated single words (lowercase). "
            "Examples: researcher, critic, evaluator."
        ),
        gen_cfg=gen_cfg,
        style="one_sentence",
    )
    extra_roles: List[str] = []
    # parse simple comma/and separated words
    for tok in proposed.replace(" and ", ",").replace(";", ",").split(","):
        r = tok.strip().lower().strip(". ")
        if r and r.isalpha() and r not in base_roles and r not in extra_roles:
            extra_roles.append(r)
    active_roles = base_roles + extra_roles[:3]
    print("[INFO] Active roles:", active_roles)

    # 3) primary agents
    agents = [Agent(f"Agent{i+1}", role, debug=args.debug) for i, role in enumerate(active_roles)]

    # helpers to fetch by role
    def get(role: str) -> Optional[Agent]:
        return next((a for a in agents if a.role == role), None)

    # 4) planner -> requirement
    planner = get("planner") or agents[0]
    req = planner.think(
        model, tokenizer, device,
        context=initial_idea,
        instruction="Return one concrete requirement (<=25 words).",
        gen_cfg=gen_cfg,
        style="one_sentence",
    )

    # 5) architect -> approach
    architect = get("architect")
    if architect:
        arch = architect.think(
            model, tokenizer, device,
            context=req,
            instruction="Describe the approach in one sentence (<=25 words), no code.",
            gen_cfg=gen_cfg,
            style="one_sentence",
        )

    # 6) implementer -> next step
    implementer = get("implementer")
    if implementer:
        impl = implementer.think(
            model, tokenizer, device,
            context=req,
            instruction="Return one actionable next step (<=18 words), imperative.",
            gen_cfg=gen_cfg,
            style="one_sentence",
        )

    # 7) tester -> acceptance check
    tester = get("tester")
    if tester:
        tst = tester.think(
            model, tokenizer, device,
            context=req,
            instruction="Return one acceptance check (<=18 words), imperative.",
            gen_cfg=gen_cfg,
            style="one_sentence",
        )

    # 8) BFS expansion preview
    levels = bfs_spawn(active_roles, max_depth=args.depth)
    print("\n[BFS Expansion]")
    for d in sorted(levels):
        print(f"Depth {d}: {', '.join(levels[d])}")

    # 9) summarizer -> bullets (disable one_sentence)
    summarizer = Agent("Summarizer", "summarizer", debug=args.debug)
    summary = summarizer.think(
        model, tokenizer, device,
        context=f"Idea: {initial_idea}\nRequirement: {req}",
        instruction="Summarize the plan.",
        gen_cfg=gen_cfg,
        style="bullets",
        bullets=5,
    )
    print("\n[SUMMARY]\n" + summary)

    # 10) demo: fixed roles list
    print("\n[DEMO OUTPUT]")
    demo_roles = ["planner","architect","implementer","tester","researcher","critic","evaluator"]
    demo_levels = bfs_spawn(demo_roles, max_depth=2)
    for d in sorted(demo_levels):
        print(f"Depth {d}: {', '.join(demo_levels[d])}")


if __name__ == "__main__":
    args = build_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print("[ERROR]", type(e).__name__, str(e))
        print("Tip: Small chat-tuned models follow instructions better than GPT-2. Keep prompts short and concrete.")
