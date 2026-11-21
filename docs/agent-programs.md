# Agent programs

Design note, 2026-11-21. Not an ADR. An idea recorded so it is not lost; nothing is built.

## The idea

- **Each agent carries its own program**, not just its own numbers. Programs are composed freely
  from operators and say how the agent behaves: seek resources, avoid danger, mate.
- **Programs evolve.** Reproduction copies one with mutation; mating crosses two.
- **A fixed kernel interprets them**, as organic's `shaders/interp.vs` interprets horn structure:
  a `switch` over opcodes, the program as data in a buffer, a per-instance entry point. Changing
  a program is a buffer write, never a recompile.
- **It probably lives in inorganic.** It is the behaviour side of an art system or game, not
  physics. This repo's part is to be extensible enough that it can be built on top.

## Behaviour sits above the physics

| layer | what it is | reciprocal? |
| --- | --- | --- |
| physics | springs, contact, bonds: the ops in `spring-model-as-ops.md` | yes: equal and opposite, both ends of a bond agree |
| behaviour | what an agent wants | no, and need not be |

A program never computes a contact force or edits a bond. It outputs **intents**:

- a push, which enters the force sum like any other force op's output;
- actions (divide, eat, mate), which the rule ops take as inputs where they take a random draw
  today.

So an agent can flee without anything pushing back, and the tissue still conserves momentum
where it should.

## Shape of a program

| part | what | cost |
| --- | --- | --- |
| sensors | computed once for every agent: resource gradient, direction to the nearest neighbour of a kind, local density, own energy and age | the expensive part; uniform passes, like `pairDensity` |
| program | reads sensors and the agent's memory, writes actuators and memory | once per agent, not per pair |
| genome, memory | state on feedback edges, on the device | none beyond the buffers |

Candidate form: a small register machine with fixed-length programs (linear genetic
programming). Every agent runs the same number of steps and only the opcodes differ, so the
kernel is bounded. A neural network per agent is the uniform alternative, but its genome is
numbers, and its mutations do not read as changes of behaviour.

Evolution can stay on the device: mutation with the counter-based random numbers in
`src/gpu/sim/springs/rng.ts`, partner choice with the propose and accept handshake `formBonds`
uses, births into free slots as division does.

## What this repo has to allow

The reason for writing this now. None of it is a commitment to build agent programs here.

| requirement | status |
| --- | --- |
| A force from outside the model enters the sum like any other | yes: any op that outputs a `force` field can be added with `addForces` |
| Ops can be registered from another package | to check: `registerOp` is exported; nothing outside this repo has used it for a kernel op |
| Per-agent values beyond `species` and `radius` travel with the particles | not yet: they are separate point fields, which helps, but division must copy every per-agent field to the daughter, including ones it does not know about |
| Rule ops take their triggers as inputs | not yet: division and leader selection are still host functions with their own random draws |
| Sensing: fields that agents read and write | partly: `splatDensity` and a gradient op exist; no op samples a field at particle positions |
| State stays on the device between ticks, including buffers this repo did not define | yes for point fields on feedback edges; a resident bundle on a feedback edge is untested |
| A CPU reference for every kernel | the convention here; an interpreter would need one that matches exactly |

Two of these should shape the remaining spring-model slices:

1. **Division as an op should copy a set of per-agent fields it is handed**, not a fixed list.
2. **Rule ops should take "wants to" as an input field**, with the random draw as one op that
   can produce it.

## Open

- Where the interpreter lives. A general "agent program" op here, or wholly in inorganic.
- Whether the same interpreter could serve the fusion spike: one fixed kernel per loop, with the
  model's own terms as opcodes, is a third strategy beside per-op kernels and compiled fusion.
- Cost and reliability of a kernel that is all loops and switches on Dawn/Metal. Not measured.
  One silent wrong answer from control flow has already been seen here.
- How to see what an evolved program does.

Explainer candidate: one fixed kernel, many programs. The same picture explains organic's
interpreter.
