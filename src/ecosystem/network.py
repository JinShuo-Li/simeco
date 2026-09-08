"""Small dependency-free adaptive policies owned by individual organisms."""
from __future__ import annotations
import hashlib
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any
from .actions import HEAD_SIZES, TOTAL_ACTION_OUTPUTS, EmbodiedAction

OUTCOME_SIZE = len(HEAD_SIZES)
LEGACY_HIDDEN = 10

def _matrix(rows, columns, rng, scale):
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]

def _zeros(rows, columns):
    return [[0.0] * columns for _ in range(rows)]

def _derived_rng(values):
    digest = hashlib.sha256(repr(values).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))

def _action_context(actions):
    context = [0.0] * TOTAL_ACTION_OUTPUTS
    if actions is not None:
        offset = 0
        for size, action in zip(HEAD_SIZES, actions):
            if 0 <= action < size:
                context[offset + action] = 1.0
            offset += size
    return context

def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))

def _outer_add(target, left, right):
    for row, value in zip(target, left):
        for index, feature in enumerate(right):
            row[index] += value * feature

@dataclass(slots=True)
class AdaptivePolicy:
    """Feed-forward residual policy with an optional GRU and short TBPTT."""
    inputs: int
    hidden: int
    outputs: int
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    memory_size: int
    wz: list[list[float]]
    uz: list[list[float]]
    bz: list[float]
    wr: list[list[float]]
    ur: list[list[float]]
    br: list[float]
    wh: list[list[float]]
    uh: list[list[float]]
    bh: list[float]
    wm_out: list[list[float]]
    residual_scale: float = 1.0
    gamma: float = 0.95
    unroll: int = 8
    baseline: float = 0.0
    head_baselines: list[float] = field(default_factory=lambda: [0.0] * len(HEAD_SIZES))
    updates: int = 0
    tbptt_updates: int = 0
    reward_total: float = 0.0
    memory: list[float] = field(default_factory=list)
    previous_actions: list[int] | None = None
    previous_outcomes: list[float] = field(default_factory=lambda: [0.0] * OUTCOME_SIZE)
    trajectory: list[dict[str, Any]] = field(default_factory=list, repr=False)
    last_observation: list[float] | None = field(default=None, repr=False)
    last_actions: list[int] | None = field(default=None, repr=False)
    last_probabilities: list[float] | None = field(default=None, repr=False)
    last_gradient_scale: float = field(default=1.0, repr=False)
    _pending: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def recurrent_inputs(self):
        return self.hidden + TOTAL_ACTION_OUTPUTS + OUTCOME_SIZE

    @classmethod
    def random(cls, inputs, hidden, outputs, rng, memory_size=12):
        # Keep ecological RNG consumption equal to V3/V4: ten encoder rows.
        legacy = min(LEGACY_HIDDEN, hidden)
        w1 = _matrix(legacy, inputs, rng, math.sqrt(2.0 / inputs))
        private = _derived_rng(w1)
        w1 += _matrix(hidden - legacy, inputs, private, math.sqrt(2.0 / inputs))
        context = hidden + TOTAL_ACTION_OUTPUTS + OUTCOME_SIZE
        gate_scale = math.sqrt(1.0 / context)
        state_scale = math.sqrt(1.0 / max(1, memory_size))
        return cls(
            inputs, hidden, outputs, w1, [0.0] * hidden,
            _zeros(outputs, hidden), [0.0] * outputs, memory_size,
            _matrix(memory_size, context, private, gate_scale),
            _matrix(memory_size, memory_size, private, state_scale),
            [-0.35] * memory_size,
            _matrix(memory_size, context, private, gate_scale),
            _matrix(memory_size, memory_size, private, state_scale),
            [0.0] * memory_size,
            _matrix(memory_size, context, private, gate_scale),
            _matrix(memory_size, memory_size, private, state_scale),
            [0.0] * memory_size,
            _zeros(outputs, memory_size),
            memory=[0.0] * memory_size,
        )

    def _transition(self, observation, use_memory):
        encoded = [math.tanh(sum(w*x for w, x in zip(row, observation)) + bias)
                   for row, bias in zip(self.w1, self.b1)]
        old = self.memory[:] if use_memory else [0.0] * self.memory_size
        context = encoded + ((_action_context(self.previous_actions)
                   + [math.tanh(v) for v in self.previous_outcomes])
                   if use_memory else [0.0] * (TOTAL_ACTION_OUTPUTS + OUTCOME_SIZE))
        if use_memory:
            update = [_sigmoid(sum(w*x for w,x in zip(row,context)) + sum(w*h for w,h in zip(urow,old)) + b)
                      for row,urow,b in zip(self.wz,self.uz,self.bz)]
            reset = [_sigmoid(sum(w*x for w,x in zip(row,context)) + sum(w*h for w,h in zip(urow,old)) + b)
                     for row,urow,b in zip(self.wr,self.ur,self.br)]
            reset_old = [r*h for r,h in zip(reset,old)]
            candidate = [math.tanh(sum(w*x for w,x in zip(row,context)) + sum(w*h for w,h in zip(urow,reset_old)) + b)
                         for row,urow,b in zip(self.wh,self.uh,self.bh)]
            new = [(1-z)*h + z*c for z,h,c in zip(update,old,candidate)]
        else:
            update = reset = candidate = [0.0] * self.memory_size
            new = [0.0] * self.memory_size
        raw = [sum(w*x for w,x in zip(row,encoded)) + sum(w*h for w,h in zip(mrow,new)) + b
               for row,mrow,b in zip(self.w2,self.wm_out,self.b2)]
        return {"observation":list(observation),"encoded":encoded,"memory_before":old,
                "context":context,"update_gate":update,"reset_gate":reset,
                "candidate":candidate,"memory_after":new,"raw":raw,"use_memory":use_memory}

    def advance(self, observation, use_memory=True):
        transition = self._transition(observation, use_memory)
        self.memory = transition["memory_after"][:]
        self.last_observation = list(observation)
        self._pending = transition
        return [math.tanh(v) * self.residual_scale for v in transition["raw"]]

    def preferences(self, observation, use_memory=True):
        return [math.tanh(v)*self.residual_scale for v in self._transition(observation,use_memory)["raw"]]

    def record_decision(self, observation, action, combined_probabilities, gradient_scale=1.0):
        if self.last_observation != observation or self._pending is None:
            self.advance(observation)
        self.last_actions = action.indices()
        self.last_probabilities = combined_probabilities[:]
        self.last_gradient_scale = gradient_scale
        self._pending["actions"] = self.last_actions[:]
        self._pending["probabilities"] = combined_probabilities[:]
        self._pending["gradient_scale"] = gradient_scale

    def probabilities(self, observation, use_memory=True):
        logits = self.preferences(observation,use_memory)
        result=[]; offset=0
        for size in HEAD_SIZES:
            head=logits[offset:offset+size]; peak=max(head)
            values=[math.exp(max(-30.0,v-peak)) for v in head]; total=sum(values)
            result += [v/total for v in values]; offset += size
        return result

    def learn(self, reward, learning_rate, head_rewards=None, terminal=False):
        if self.last_actions is None or self._pending is None:
            return
        outcomes=list(head_rewards if head_rewards is not None else [reward]*len(HEAD_SIZES))
        self.reward_total += reward
        if not self._pending["use_memory"]:
            self._learn_feedforward(outcomes,learning_rate)
        else:
            transition=self._pending
            transition["outcomes"]=outcomes; transition["reward"]=reward
            self.trajectory.append(transition); self.updates += 1
            if len(self.trajectory)>=self.unroll or terminal:
                self._learn_trajectory(learning_rate)
        self.baseline=.96*self.baseline+.04*reward
        self.head_baselines=[.96*b+.04*r for b,r in zip(self.head_baselines,outcomes)]
        self.previous_actions=self.last_actions[:]; self.previous_outcomes=outcomes
        self._pending=None

    def _policy_delta(self, transition, returns):
        delta=[0.0]*self.outputs; offset=0
        for head,(size,action) in enumerate(zip(HEAD_SIZES,transition["actions"])):
            advantage=max(-4.0,min(4.0,returns[head]-self.head_baselines[head]))
            scale=1.8 if advantage<0 else 1.0
            for i in range(size):
                delta[offset+i]=-transition["probabilities"][offset+i]*advantage*scale
            delta[offset+action]+=advantage*scale; offset+=size
        return [v*transition["gradient_scale"]*self.residual_scale*(1-math.tanh(raw)**2)
                for v,raw in zip(delta,transition["raw"])]

    def _learn_feedforward(self,outcomes,rate):
        t=self._pending; delta=self._policy_delta(t,outcomes); old_w2=[row[:] for row in self.w2]
        for out,value in enumerate(delta):
            value=max(-2.0,min(2.0,value))
            for hidden in range(self.hidden): self.w2[out][hidden]+=rate*value*t["encoded"][hidden]
            self.b2[out]+=rate*value
        hidden_grad=[sum(old_w2[o][i]*delta[o] for o in range(self.outputs)) for i in range(self.hidden)]
        for hidden,gradient in enumerate(hidden_grad):
            value=max(-2.0,min(2.0,gradient*(1-t["encoded"][hidden]**2)))
            for inp in range(self.inputs): self.w1[hidden][inp]+=rate*value*t["observation"][inp]
            self.b1[hidden]+=rate*value
        self.updates+=1

    def _learn_trajectory(self,rate):
        if not self.trajectory: return
        count=len(self.trajectory); returns=[[0.0]*OUTCOME_SIZE for _ in range(count)]
        future=[0.0]*OUTCOME_SIZE
        for i in range(count-1,-1,-1):
            future=[now+self.gamma*later for now,later in zip(self.trajectory[i]["outcomes"],future)]
            returns[i]=future[:]
        g={"w1":_zeros(self.hidden,self.inputs),"b1":[0.0]*self.hidden,
           "w2":_zeros(self.outputs,self.hidden),"b2":[0.0]*self.outputs,
           "wm_out":_zeros(self.outputs,self.memory_size),
           "wz":_zeros(self.memory_size,self.recurrent_inputs),"uz":_zeros(self.memory_size,self.memory_size),"bz":[0.0]*self.memory_size,
           "wr":_zeros(self.memory_size,self.recurrent_inputs),"ur":_zeros(self.memory_size,self.memory_size),"br":[0.0]*self.memory_size,
           "wh":_zeros(self.memory_size,self.recurrent_inputs),"uh":_zeros(self.memory_size,self.memory_size),"bh":[0.0]*self.memory_size}
        dh_future=[0.0]*self.memory_size
        for t,discounted in zip(reversed(self.trajectory),reversed(returns)):
            od=self._policy_delta(t,discounted)
            _outer_add(g["w2"],od,t["encoded"]); _outer_add(g["wm_out"],od,t["memory_after"])
            for i,v in enumerate(od): g["b2"][i]+=v
            enc_grad=[sum(self.w2[o][i]*od[o] for o in range(self.outputs)) for i in range(self.hidden)]
            dh=[dh_future[i]+sum(self.wm_out[o][i]*od[o] for o in range(self.outputs)) for i in range(self.memory_size)]
            old=t["memory_before"]; z=t["update_gate"]; reset=t["reset_gate"]; candidate=t["candidate"]
            dc=[dh[i]*z[i]*(1-candidate[i]**2) for i in range(self.memory_size)]
            dz=[dh[i]*(candidate[i]-old[i])*z[i]*(1-z[i]) for i in range(self.memory_size)]
            dq=[sum(self.uh[j][i]*dc[j] for j in range(self.memory_size)) for i in range(self.memory_size)]
            dr=[dq[i]*old[i]*reset[i]*(1-reset[i]) for i in range(self.memory_size)]
            _outer_add(g["wh"],dc,t["context"]); _outer_add(g["uh"],dc,[reset[i]*old[i] for i in range(self.memory_size)])
            _outer_add(g["wz"],dz,t["context"]); _outer_add(g["uz"],dz,old)
            _outer_add(g["wr"],dr,t["context"]); _outer_add(g["ur"],dr,old)
            for name,values in (("bh",dc),("bz",dz),("br",dr)):
                for i,v in enumerate(values): g[name][i]+=v
            dh_future=[dh[i]*(1-z[i])+dq[i]*reset[i]+sum(self.uz[j][i]*dz[j]+self.ur[j][i]*dr[j] for j in range(self.memory_size)) for i in range(self.memory_size)]
            for i in range(self.hidden):
                enc_grad[i]+=sum(self.wh[j][i]*dc[j]+self.wz[j][i]*dz[j]+self.wr[j][i]*dr[j] for j in range(self.memory_size))
            de=[enc_grad[i]*(1-t["encoded"][i]**2) for i in range(self.hidden)]
            _outer_add(g["w1"],de,t["observation"])
            for i,v in enumerate(de): g["b1"][i]+=v
        scale=rate/count
        for name in ("w1","w2","wm_out","wz","uz","wr","ur","wh","uh"):
            for row,grow in zip(getattr(self,name),g[name]):
                for i,v in enumerate(grow): row[i]+=scale*max(-3.0,min(3.0,v))
        for name in ("b1","b2","bz","br","bh"):
            for i,v in enumerate(g[name]): getattr(self,name)[i]+=scale*max(-3.0,min(3.0,v))
        self.tbptt_updates+=1; self.trajectory.clear()

    def reset_runtime_memory(self):
        self.memory=[0.0]*self.memory_size; self.previous_actions=None
        self.previous_outcomes=[0.0]*OUTCOME_SIZE; self.trajectory.clear()
        self.last_observation=None; self.last_actions=None; self.last_probabilities=None
        self.last_gradient_scale=1.0; self._pending=None

    def offspring(self,rng,mutation_rate,mutation_scale):
        child=AdaptivePolicy.from_dict(self.to_dict())
        child.updates=child.tbptt_updates=0; child.reward_total=0.0
        child.baseline*=.5; child.head_baselines=[v*.5 for v in child.head_baselines]
        legacy=min(LEGACY_HIDDEN,child.hidden)
        for row in child.w1[:legacy]:
            for i in range(len(row)):
                if rng.random()<mutation_rate: row[i]+=rng.gauss(0,mutation_scale)
        for row in child.w2:
            for i in range(legacy):
                if rng.random()<mutation_rate: row[i]+=rng.gauss(0,mutation_scale)
        for i in range(legacy):
            if rng.random()<mutation_rate: child.b1[i]+=rng.gauss(0,mutation_scale)
        for i in range(child.outputs):
            if rng.random()<mutation_rate: child.b2[i]+=rng.gauss(0,mutation_scale)
        private=_derived_rng(rng.getstate())
        matrices=[child.w1[legacy:],[row[legacy:] for row in child.w2],child.wz,child.uz,child.wr,child.ur,child.wh,child.uh,child.wm_out]
        for matrix in matrices:
            for row in matrix:
                for i in range(len(row)):
                    if private.random()<mutation_rate: row[i]+=private.gauss(0,mutation_scale)
        for vector in (child.b1[legacy:],child.bz,child.br,child.bh):
            for i in range(len(vector)):
                if private.random()<mutation_rate: vector[i]+=private.gauss(0,mutation_scale)
        child.reset_runtime_memory(); return child

    def to_dict(self):
        return {name:getattr(self,name) for name in (
            "inputs","hidden","outputs","w1","b1","w2","b2","memory_size","wz","uz","bz",
            "wr","ur","br","wh","uh","bh","wm_out","residual_scale","gamma","unroll","baseline",
            "head_baselines","updates","tbptt_updates","reward_total","memory","previous_actions",
            "previous_outcomes","trajectory","last_observation","last_actions","last_probabilities",
            "last_gradient_scale")}

    @classmethod
    def from_dict(cls,data):
        values=copy.deepcopy(data); values["trajectory"]=[dict(item) for item in values.get("trajectory",[])]
        return cls(**values)

TinyMLP=AdaptivePolicy
