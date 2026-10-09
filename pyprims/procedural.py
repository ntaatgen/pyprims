"""Productions, utility learning and production compilation.

Port of Procedural.swift, Production.swift and Instantiation.swift.

In PRIMs a production is *task-general*: it is keyed on the remaining PRIM
strings of the operator in the operator buffer, not on the operator itself.
Initially every step executes one PRIM (a "t" production with fixed utility
``primU``).  Compilation merges two consecutive productions into one that
does both steps; it starts at utility ``nu`` and must outgrow ``primU`` before
it is chosen.  Because operators in different tasks share PRIM sequences,
compiled productions transfer between tasks (Taatgen, 2013).
"""
from __future__ import annotations

from dataclasses import dataclass

from .prim import Prim
from .values import actr_noise, chop_prims

DEFAULTS = dict(
    utility_noise=0.05,                   # egs:
    default_u=0.0,                        # nu:
    prim_u=2.0,                           # primU:
    alpha=0.1,                            # alpha:
    production_action_latency=0.05,       # dat:
    production_and_prim_latency=0.3,      # production-prim-latency:
    procedural_reward=4.0,                # procedural-reward:
)


class Production:
    __slots__ = ("name", "full_name", "model", "condition", "action",
                 "new_condition", "new_action", "conditions", "actions", "u",
                 "goal_checks", "parent1", "parent2", "task_id")

    def __init__(self, name, model, condition, action, parent1, parent2, task_id, u):
        self.name = name
        self.full_name = None
        self.model = model
        self.condition = condition          # PRIM string this production starts from
        self.action = action
        self.new_condition = condition      # what is left after it fires
        self.new_action = action
        self.conditions: list[Prim] = []
        self.actions: list[Prim] = []
        self.u = u
        self.goal_checks = []
        self.parent1 = None if parent1 is None or parent1.name.startswith("t") else parent1
        self.parent2 = None if parent2 is None or parent2.name.startswith("t") else parent2
        self.task_id = task_id

    def __repr__(self):
        return f"<Production {self.name} u={self.u:.3f}>"

    def add_condition(self, prim: Prim):
        self.conditions.append(prim)
        _, new = chop_prims(self.condition, len(self.conditions))
        self.new_condition = new if new != "" else None

    def add_action(self, prim: Prim):
        self.actions.append(prim)
        _, new = chop_prims(self.action, len(self.actions))
        self.new_action = new if new != "" else None

    def instantiate(self):
        op = self.model.buffers.get("operator")
        if op is None:
            return None
        cond = op.slotvals.get("condition")
        act = op.slotvals.get("action")
        cond = cond if isinstance(cond, str) else None
        act = act if isinstance(act, str) else None
        if self.condition == cond and self.action == act:
            u = self.u + actr_noise(self.model.rng, self.model.procedural.utility_noise)
            return Instantiation(self, self.model.time, u)
        return None

    def _advance_operator(self, also_action: bool):
        op = self.model.buffers["operator"]
        if self.new_condition is not None:
            op.set_slot_value("condition", self.new_condition)
        else:
            op.slotvals.pop("condition", None)
        if also_action:
            if self.new_action is not None:
                op.set_slot_value("action", self.new_action)
            else:
                op.slotvals.pop("action", None)

    def test_fire(self):
        """Check conditions only (used when trying out a candidate operator)."""
        for c in self.conditions:
            if not c.fire(True)[0]:
                return False, c, 0.0
        self._advance_operator(also_action=False)
        return True, None, 0.0

    def fire(self):
        latency = 0.0
        for c in self.conditions:
            if not c.fire(True)[0]:
                return False, c, 0.0
        for a in self.actions:
            ok, lat = a.fire(False)
            if not ok:
                return False, a, 0.0
            latency += lat
        self._advance_operator(also_action=True)
        return True, None, latency


@dataclass
class Instantiation:
    p: Production
    time: float
    u: float


class Procedural:
    def __init__(self, model):
        self.model = model
        self.productions: dict[str, Production] = {}
        self.productions_for_reward: list[Instantiation] = []
        self.last_production = None
        self.last_operator = None
        self.set_parameters_to_default()

    def set_parameters_to_default(self):
        for k, v in DEFAULTS.items():
            setattr(self, k, v)

    def reset(self):
        self.last_production = None
        self.clear_reward_trace()

    def clear_reward_trace(self):
        self.productions_for_reward = []

    def issue_reward(self, reward: float):
        """Utility learning: U += alpha (reward - elapsed - U)."""
        t = self.model.time
        for inst in self.productions_for_reward:
            payoff = reward - (t - inst.time)
            inst.p.u = inst.p.u + self.alpha * (payoff - inst.p.u)
        self.clear_reward_trace()

    def fire_production(self, inst: Instantiation, compile: bool):
        if compile:
            if not inst.p.name.startswith("t"):
                self.productions_for_reward.append(inst)
            if self.last_production is not None:
                self.compile_productions(self.last_production, inst)
            self.last_production = inst.p
            return inst.p.fire()
        return inst.p.test_fire()

    def find_matching_production(self) -> Instantiation:
        m = self.model
        op = m.buffers.get("operator")
        condition = op.slotvals.get("condition") if op is not None else None
        action = op.slotvals.get("action") if op is not None else None
        best = None
        for p in list(self.productions.values()):
            ins = p.instantiate()
            if ins is not None and (best is None or best.u < ins.u):
                best = ins
        if best is None or best.u < self.prim_u:
            if condition is not None:
                prim_name, _ = chop_prims(condition, 1)
                p = Production("t" + prim_name, m, condition, action, None, None, 0, self.prim_u)
                p.add_condition(Prim(prim_name, m))
                return Instantiation(p, m.time, self.prim_u)
            prim_name, _ = chop_prims(action, 1)
            p = Production("t" + prim_name, m, None, action, None, None, 0, self.prim_u)
            p.add_action(Prim(prim_name, m))
            return Instantiation(p, m.time, self.prim_u)
        return best

    def compile_productions(self, p1: Production, inst2: Instantiation):
        p2 = inst2.p
        n1 = p1.name[1:] if p1.name.startswith("t") else p1.name
        n2 = p2.name[1:] if p2.name.startswith("t") else p2.name
        new_name = n1 + ";" + n2
        full = new_name
        if p2.new_condition is not None:
            full += "|" + p2.new_condition
        if p2.new_action is not None:
            full += "|" + p2.new_action
        existing = self.productions.get(full)
        if existing is not None:
            existing.u += self.alpha * (p1.u - existing.u)
            self.model.add_to_trace(f"Reinforcing {existing.name} new u = {existing.u:.3f}", 4)
            return
        task_id = self.model.current_task_index if self.model.current_task_index is not None else 0
        np_ = Production(new_name, self.model, p1.condition, p1.action, p1, p2, task_id, self.default_u)
        np_.conditions = p1.conditions + p2.conditions
        np_.actions = p1.actions + p2.actions
        np_.new_condition = p2.new_condition
        np_.new_action = p2.new_action
        np_.goal_checks = p1.goal_checks
        np_.full_name = full
        self.productions[full] = np_
        self.model.add_to_trace(f"Compiling {np_.name}", 4)
