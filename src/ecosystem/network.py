"""Small dependency-free adaptive policies owned by individual organisms."""
from __future__ import annotations
import hashlib
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any
from .actions import (
    ADAPTIVE_HEAD_SIZES,
    HEAD_SIZES,
    TOTAL_ACTION_OUTPUTS,
    TOTAL_ADAPTIVE_OUTPUTS,
)
from .social import (
    MAX_SOCIAL_ENTRIES,
    ENTITY_HIDDEN_SIZE,
    SOCIAL_AGGREGATE_SIZE,
    SOCIAL_EMBEDDING_SIZE,
    SOCIAL_PHYSICAL_SIZE,
    SOCIAL_SENSOR_SIZE,
    SOCIAL_SLOT_SIZE,
    SOCIAL_SLOTS,
    ENTITY_SLOTS,
    SOCIAL_STALE_TICKS,
)

OUTCOME_SIZE = len(HEAD_SIZES)
LEGACY_HIDDEN = 10

def _matrix(rows, columns, rng, scale):
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]

def _zeros(rows, columns):
    return [[0.0] * columns for _ in range(rows)]

def _derived_rng(values):
    digest = hashlib.sha256(repr(values).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))

def _action_context(actions, head_sizes):
    context = [0.0] * sum(head_sizes)
    if actions is not None:
        offset = 0
        for size, action in zip(head_sizes, actions):
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
    entity_w: list[list[float]]
    entity_b: list[float]
    base_inputs: int = 33
    social_memory: dict[int, dict[str, Any]] = field(default_factory=dict)
    max_social_entries: int = MAX_SOCIAL_ENTRIES
    social_stale_ticks: int = SOCIAL_STALE_TICKS
    social_entries_created: int = 0
    social_total_encounters: int = 0
    social_known_encounters: int = 0
    social_evictions_capacity: int = 0
    social_evictions_stale: int = 0
    social_evicted_lifetime_total: int = 0
    social_evicted_entries: int = 0
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
        return self.hidden + sum(self.head_sizes) + len(self.head_sizes)

    @property
    def head_sizes(self):
        return ADAPTIVE_HEAD_SIZES if self.outputs == TOTAL_ADAPTIVE_OUTPUTS else HEAD_SIZES

    @classmethod
    def random(cls, inputs, hidden, outputs, rng, memory_size=12):
        # Keep ecological RNG consumption equal to V3/V4: ten encoder rows.
        legacy = min(LEGACY_HIDDEN, hidden)
        base_inputs = inputs
        total_inputs = base_inputs + SOCIAL_AGGREGATE_SIZE
        legacy_weights = _matrix(legacy, base_inputs, rng, math.sqrt(2.0 / base_inputs))
        private = _derived_rng(legacy_weights)
        w1 = [
            row + [private.gauss(0.0, math.sqrt(1.0 / SOCIAL_AGGREGATE_SIZE)) for _ in range(SOCIAL_AGGREGATE_SIZE)]
            for row in legacy_weights
        ]
        w1 += _matrix(hidden - legacy, total_inputs, private, math.sqrt(2.0 / total_inputs))
        head_sizes = ADAPTIVE_HEAD_SIZES if outputs == TOTAL_ADAPTIVE_OUTPUTS else HEAD_SIZES
        context = hidden + sum(head_sizes) + len(head_sizes)
        gate_scale = math.sqrt(1.0 / context)
        state_scale = math.sqrt(1.0 / max(1, memory_size))
        return cls(
            total_inputs, hidden, outputs, w1, [0.0] * hidden,
            (_zeros(min(outputs, TOTAL_ACTION_OUTPUTS), hidden)
             + _matrix(max(0, outputs - TOTAL_ACTION_OUTPUTS), hidden, private, 0.30)),
            [0.0] * outputs, memory_size,
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
            _matrix(ENTITY_HIDDEN_SIZE, SOCIAL_SLOT_SIZE, private, math.sqrt(1.0 / SOCIAL_SLOT_SIZE)),
            [0.0] * ENTITY_HIDDEN_SIZE,
            base_inputs=base_inputs,
            memory=[0.0] * memory_size,
            previous_outcomes=[0.0] * len(head_sizes),
            head_baselines=[0.0] * len(head_sizes),
        )

    def _evict_social_entry(self, identity, step, reason):
        entry=self.social_memory.pop(identity)
        self.social_evicted_lifetime_total+=max(0,step-entry.get("created_at",step))
        self.social_evicted_entries+=1
        if reason=="stale":
            self.social_evictions_stale+=1
        else:
            self.social_evictions_capacity+=1

    def _social_observation(
        self, observation, social_slots, social_enabled, step, update_entries,
        embeddings_enabled=True,
    ):
        if not social_enabled:
            return list(observation)+[0.0]*SOCIAL_AGGREGATE_SIZE, [], {
                "entities":[], "max_indices":[]
            }
        if update_entries:
            stale=[
                identity for identity,entry in self.social_memory.items()
                if step-entry["last_seen"]>self.social_stale_ticks
            ]
            for identity in stale:
                self._evict_social_entry(identity,step,"stale")
        entities=[]; keys=[]
        for slot in (social_slots or [])[:ENTITY_SLOTS]:
            identity=int(slot["id"]); entry=self.social_memory.get(identity)
            known=entry is not None
            if entry is None:
                entry={
                    "embedding":[0.0]*SOCIAL_EMBEDDING_SIZE,
                    "encounters":0,"created_at":step,"last_seen":step,
                    "consecutive_encounters":0,"max_streak":0,
                    "distance_sum":0.0,"outcome_trace":0.0,
                }
                if update_entries:
                    self.social_memory[identity]=entry
                    self.social_entries_created+=1
            if update_entries:
                self.social_total_encounters+=1
                self.social_known_encounters+=int(known)
                entry["consecutive_encounters"]=(
                    entry.get("consecutive_encounters",0)+1
                    if known and entry["last_seen"]==step-1 else 1
                )
                entry["max_streak"]=max(entry.get("max_streak",0),entry["consecutive_encounters"])
                entry["encounters"]+=1; entry["last_seen"]=step
                entry["distance_sum"]=entry.get("distance_sum",0.0)+slot["features"][2]
            sensors = list(slot["features"])
            sensors += [0.0] * (SOCIAL_SENSOR_SIZE - len(sensors))
            entity_input=sensors[:SOCIAL_SENSOR_SIZE]+(
                list(entry["embedding"]) if embeddings_enabled
                else [0.0]*SOCIAL_EMBEDDING_SIZE
            )
            representation=[
                math.tanh(sum(weight*value for weight,value in zip(row,entity_input))+bias)
                for row,bias in zip(self.entity_w,self.entity_b)
            ]
            entities.append({"id":identity,"input":entity_input,"representation":representation})
            keys.append(identity)
        if update_entries and len(self.social_memory)>self.max_social_entries:
            victims=sorted(
                self.social_memory.items(),
                key=lambda item:(item[1]["last_seen"],item[1]["encounters"])
            )[:len(self.social_memory)-self.max_social_entries]
            for identity,_ in victims:
                self._evict_social_entry(identity,step,"capacity")
        if not entities:
            aggregate=[0.0]*SOCIAL_AGGREGATE_SIZE; max_indices=[]
        else:
            count=len(entities)
            mean=[
                sum(entity["representation"][index] for entity in entities)/count
                for index in range(ENTITY_HIDDEN_SIZE)
            ]
            max_indices=[
                max(range(count),key=lambda entity:indexed[entity])
                for indexed in ([item["representation"][index] for item in entities]
                                for index in range(ENTITY_HIDDEN_SIZE))
            ]
            maximum=[
                entities[max_indices[index]]["representation"][index]
                for index in range(ENTITY_HIDDEN_SIZE)
            ]
            aggregate=mean+maximum
        return list(observation)+aggregate,keys,{"entities":entities,"max_indices":max_indices}

    def _transition(self, observation, use_memory):
        encoded = [math.tanh(sum(w*x for w, x in zip(row, observation)) + bias)
                   for row, bias in zip(self.w1, self.b1)]
        old = self.memory[:] if use_memory else [0.0] * self.memory_size
        context = encoded + ((_action_context(self.previous_actions, self.head_sizes)
                   + [math.tanh(v) for v in self.previous_outcomes])
                   if use_memory else [0.0] * (sum(self.head_sizes) + len(self.head_sizes)))
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

    def advance(
        self, observation, use_memory=True, social_slots=None, social_enabled=False,
        step=0, social_embeddings_enabled=True,
    ):
        adaptive_observation, social_keys, social_cache = self._social_observation(
            observation, social_slots, social_enabled, step, True,
            social_embeddings_enabled,
        )
        transition = self._transition(adaptive_observation, use_memory)
        transition["social_keys"]=social_keys
        transition["social_cache"]=social_cache
        self.memory = transition["memory_after"][:]
        self.last_observation = list(observation)
        self._pending = transition
        return [math.tanh(v) * self.residual_scale for v in transition["raw"]]

    def preferences(
        self, observation, use_memory=True, social_slots=None, social_enabled=False,
        step=0, social_embeddings_enabled=True,
    ):
        adaptive_observation,_,_ = self._social_observation(
            observation, social_slots, social_enabled, step, False,
            social_embeddings_enabled,
        )
        return [math.tanh(v)*self.residual_scale for v in self._transition(adaptive_observation,use_memory)["raw"]]

    def record_decision(self, observation, action, combined_probabilities, gradient_scale=1.0):
        if self.last_observation != observation or self._pending is None:
            self.advance(observation)
        self.last_actions = action.indices() if hasattr(action, "indices") else list(action)
        self.last_probabilities = combined_probabilities[:]
        self.last_gradient_scale = copy.deepcopy(gradient_scale)
        self._pending["actions"] = self.last_actions[:]
        self._pending["probabilities"] = combined_probabilities[:]
        self._pending["gradient_scale"] = gradient_scale

    def probabilities(self, observation, use_memory=True):
        logits = self.preferences(observation,use_memory)
        result=[]; offset=0
        for size in self.head_sizes:
            head=logits[offset:offset+size]; peak=max(head)
            values=[math.exp(max(-30.0,v-peak)) for v in head]; total=sum(values)
            result += [v/total for v in values]; offset += size
        return result

    def learn(self, reward, learning_rate, head_rewards=None, terminal=False):
        if self.last_actions is None or self._pending is None:
            return
        outcomes=list(head_rewards if head_rewards is not None else [reward]*len(self.head_sizes))
        self.reward_total += reward
        if not self._pending["use_memory"]:
            self._learn_feedforward(outcomes,learning_rate)
        else:
            transition=self._pending
            transition["outcomes"]=outcomes; transition["reward"]=reward
            for identity in set(transition.get("social_keys", [])):
                if identity is None or identity not in self.social_memory:
                    continue
                entry=self.social_memory[identity]
                signal=max(-1.0,min(1.0,reward))
                entry["outcome_trace"]=.95*entry.get("outcome_trace",0.0)+.05*signal
            self.trajectory.append(transition); self.updates += 1
            if len(self.trajectory)>=self.unroll or terminal:
                self._learn_trajectory(learning_rate)
        self.baseline=.96*self.baseline+.04*reward
        self.head_baselines=[.96*b+.04*r for b,r in zip(self.head_baselines,outcomes)]
        self.previous_actions=self.last_actions[:]; self.previous_outcomes=outcomes
        self._pending=None

    def _policy_delta(self, transition, returns):
        delta=[0.0]*self.outputs; offset=0
        for head,(size,action) in enumerate(zip(self.head_sizes,transition["actions"])):
            advantage=max(-4.0,min(4.0,returns[head]-self.head_baselines[head]))
            scale=1.8 if advantage<0 else 1.0
            for i in range(size):
                delta[offset+i]=-transition["probabilities"][offset+i]*advantage*scale
            delta[offset+action]+=advantage*scale; offset+=size
        scales = transition["gradient_scale"]
        if not isinstance(scales, list):
            scales = [scales] * len(self.head_sizes)
        expanded = [scale for size, scale in zip(self.head_sizes, scales) for _ in range(size)]
        return [v*scale*self.residual_scale*(1-math.tanh(raw)**2)
                for v,scale,raw in zip(delta,expanded,transition["raw"])]

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
        count=len(self.trajectory); returns=[[0.0]*len(self.head_sizes) for _ in range(count)]
        future=[0.0]*len(self.head_sizes)
        for i in range(count-1,-1,-1):
            future=[now+self.gamma*later for now,later in zip(self.trajectory[i]["outcomes"],future)]
            returns[i]=future[:]
        g={"w1":_zeros(self.hidden,self.inputs),"b1":[0.0]*self.hidden,
           "w2":_zeros(self.outputs,self.hidden),"b2":[0.0]*self.outputs,
           "wm_out":_zeros(self.outputs,self.memory_size),
           "wz":_zeros(self.memory_size,self.recurrent_inputs),"uz":_zeros(self.memory_size,self.memory_size),"bz":[0.0]*self.memory_size,
           "wr":_zeros(self.memory_size,self.recurrent_inputs),"ur":_zeros(self.memory_size,self.memory_size),"br":[0.0]*self.memory_size,
           "wh":_zeros(self.memory_size,self.recurrent_inputs),"uh":_zeros(self.memory_size,self.memory_size),"bh":[0.0]*self.memory_size,
           "entity_w":_zeros(ENTITY_HIDDEN_SIZE,SOCIAL_SLOT_SIZE),
           "entity_b":[0.0]*ENTITY_HIDDEN_SIZE}
        social_gradients={}
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
            input_gradient=[
                sum(self.w1[hidden][column]*de[hidden] for hidden in range(self.hidden))
                for column in range(self.inputs)
            ]
            social_cache=t.get("social_cache",{"entities":[],"max_indices":[]})
            entities=social_cache["entities"]
            if entities:
                entity_gradients=[[0.0]*ENTITY_HIDDEN_SIZE for _ in entities]
                for feature in range(ENTITY_HIDDEN_SIZE):
                    mean_gradient=input_gradient[self.base_inputs+feature]/len(entities)
                    for gradient in entity_gradients:
                        gradient[feature]+=mean_gradient
                    winner=social_cache["max_indices"][feature]
                    entity_gradients[winner][feature]+=input_gradient[
                        self.base_inputs+ENTITY_HIDDEN_SIZE+feature
                    ]
                for entity,representation_gradient in zip(entities,entity_gradients):
                    entity_delta=[
                        gradient*(1.0-value**2)
                        for gradient,value in zip(
                            representation_gradient,entity["representation"]
                        )
                    ]
                    _outer_add(g["entity_w"],entity_delta,entity["input"])
                    for i,value in enumerate(entity_delta):
                        g["entity_b"][i]+=value
                    input_delta=[
                        sum(self.entity_w[row][column]*entity_delta[row]
                            for row in range(ENTITY_HIDDEN_SIZE))
                        for column in range(SOCIAL_SLOT_SIZE)
                    ]
                    gradient=social_gradients.setdefault(
                        entity["id"],[0.0]*SOCIAL_EMBEDDING_SIZE
                    )
                    for i in range(SOCIAL_EMBEDDING_SIZE):
                        gradient[i]+=input_delta[SOCIAL_SENSOR_SIZE+i]
            _outer_add(g["w1"],de,t["observation"])
            for i,v in enumerate(de): g["b1"][i]+=v
        scale=rate/count
        for name in ("w1","w2","wm_out","wz","uz","wr","ur","wh","uh","entity_w"):
            for row,grow in zip(getattr(self,name),g[name]):
                for i,v in enumerate(grow): row[i]+=scale*max(-3.0,min(3.0,v))
        for name in ("b1","b2","bz","br","bh","entity_b"):
            for i,v in enumerate(g[name]): getattr(self,name)[i]+=scale*max(-3.0,min(3.0,v))
        for identity,gradient in social_gradients.items():
            entry=self.social_memory.get(identity)
            if entry is None:
                continue
            entry["embedding"]=[
                max(-1.0,min(1.0,value+scale*max(-3.0,min(3.0,change))))
                for value,change in zip(entry["embedding"],gradient)
            ]
        self.tbptt_updates+=1; self.trajectory.clear()

    def reset_runtime_memory(self):
        self.memory=[0.0]*self.memory_size; self.previous_actions=None
        self.previous_outcomes=[0.0]*len(self.head_sizes); self.trajectory.clear()
        self.last_observation=None; self.last_actions=None; self.last_probabilities=None
        self.last_gradient_scale=1.0; self._pending=None

    def reset_social_memory(self):
        self.social_memory.clear()

    def social_statistics(self, step):
        entries=list(self.social_memory.values())
        encounters=sum(entry["encounters"] for entry in entries)
        top_three=sum(sorted((entry["encounters"] for entry in entries),reverse=True)[:3])
        evictions=self.social_evictions_capacity+self.social_evictions_stale
        return {
            "entries":len(entries),
            "occupancy":len(entries)/max(1,self.max_social_entries),
            "known_fraction":self.social_known_encounters/max(1,self.social_total_encounters),
            "eviction_rate":evictions/max(1,self.social_entries_created),
            "capacity_evictions":self.social_evictions_capacity,
            "stale_evictions":self.social_evictions_stale,
            "evicted_lifetime":self.social_evicted_lifetime_total/max(1,self.social_evicted_entries),
            "current_lifetime":sum(
                max(0,step-entry.get("created_at",step)) for entry in entries
            )/max(1,len(entries)),
            "mean_encounters":encounters/max(1,len(entries)),
            "top3_concentration":top_three/max(1,encounters),
            "mean_max_streak":sum(entry.get("max_streak",0) for entry in entries)/max(1,len(entries)),
            "mean_distance":sum(entry.get("distance_sum",0.0) for entry in entries)/max(1,encounters),
        }

    def offspring(self,rng,mutation_rate,mutation_scale):
        child=AdaptivePolicy.from_dict(self.to_dict())
        child.updates=child.tbptt_updates=0; child.reward_total=0.0
        child.baseline*=.5; child.head_baselines=[v*.5 for v in child.head_baselines]
        legacy=min(LEGACY_HIDDEN,child.hidden)
        for row in child.w1[:legacy]:
            for i in range(child.base_inputs):
                if rng.random()<mutation_rate: row[i]+=rng.gauss(0,mutation_scale)
        for row in child.w2:
            for i in range(legacy):
                if rng.random()<mutation_rate: row[i]+=rng.gauss(0,mutation_scale)
        for i in range(legacy):
            if rng.random()<mutation_rate: child.b1[i]+=rng.gauss(0,mutation_scale)
        for i in range(child.outputs):
            if rng.random()<mutation_rate: child.b2[i]+=rng.gauss(0,mutation_scale)
        private=_derived_rng(rng.getstate())
        matrices=[
            child.w1[legacy:],
            [row[child.base_inputs:] for row in child.w1[:legacy]],
            [row[legacy:] for row in child.w2],
            child.wz,child.uz,child.wr,child.ur,child.wh,child.uh,child.wm_out,
            child.entity_w,
        ]
        for matrix in matrices:
            for row in matrix:
                for i in range(len(row)):
                    if private.random()<mutation_rate: row[i]+=private.gauss(0,mutation_scale)
        for vector in (child.b1[legacy:],child.bz,child.br,child.bh,child.entity_b):
            for i in range(len(vector)):
                if private.random()<mutation_rate: vector[i]+=private.gauss(0,mutation_scale)
        child.reset_runtime_memory(); child.reset_social_memory()
        child.social_entries_created=0; child.social_total_encounters=0
        child.social_known_encounters=0; child.social_evictions_capacity=0
        child.social_evictions_stale=0; child.social_evicted_lifetime_total=0
        child.social_evicted_entries=0
        return child

    def to_dict(self):
        return {name:getattr(self,name) for name in (
            "inputs","hidden","outputs","w1","b1","w2","b2","memory_size","wz","uz","bz",
            "wr","ur","br","wh","uh","bh","wm_out","entity_w","entity_b",
            "base_inputs","social_memory","max_social_entries","social_stale_ticks",
            "social_entries_created","social_total_encounters","social_known_encounters",
            "social_evictions_capacity","social_evictions_stale",
            "social_evicted_lifetime_total","social_evicted_entries",
            "residual_scale","gamma","unroll","baseline",
            "head_baselines","updates","tbptt_updates","reward_total","memory","previous_actions",
            "previous_outcomes","trajectory","last_observation","last_actions","last_probabilities",
            "last_gradient_scale")}

    @classmethod
    def from_dict(cls,data):
        values=copy.deepcopy(data); values["trajectory"]=[dict(item) for item in values.get("trajectory",[])]
        values["social_memory"]={int(key):entry for key,entry in values.get("social_memory",{}).items()}
        return cls(**values)

TinyMLP=AdaptivePolicy
