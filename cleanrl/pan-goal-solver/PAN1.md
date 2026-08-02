# Pan-1

Source: [A General Goal-Conditioned Minecraft Model](https://pantograph.com/journal/pan-1),
Pantograph, July 2026. Re-read July 17, 2026.

This document is a paraphrased research note, not a copy of the article.

## Central idea

Pan treats videos as observation-only trajectories and learns goal-directed
behavior without annotated rewards. A later frame from a trajectory becomes the
goal for an earlier state. This hindsight construction turns arbitrary video
into successful examples for the state that actually occurred later.

The article identifies two state-only quantities that can be learned without
actions:

1. A goal-conditioned value related to the probability that a specified goal
   appears in the discounted future.
2. The next-frame distribution of a goal-conditioned policy.

Actions are introduced only in a smaller post-training phase. The intended
benefit is that strong goal-directed state representations make action learning
closer to a single-step prediction problem than a new long-horizon RL problem.

## Reward-free goal value

The appendix describes the goal-conditioned successor measure

```text
sigma_t(s, g) = sum over future offsets d of P(s_(t+d) = g),
with d sampled according to exponential discount gamma.
```

This has the same value as ordinary RL under a hypothetical indicator reward
that is one at goal `g` and zero elsewhere. The indicator is a mathematical
equivalence, not a reward label supplied by the dataset. The article notes that
the successor measure can be represented with likelihood, energy-based, or
contrastive methods.

Pan therefore does use the term *value function*, but it means goal occupancy,
not a critic trained on Minecraft score or environment reward.

## Training setup disclosed by the article

- Models are trained from scratch.
- Pretraining data: about 500,000 hours of diverse Minecraft gameplay video.
- Pretraining contains observations but no actions or rewards.
- Hindsight goal frames are sampled from within the same 300-frame context.
- Post-training data: about 2,000 hours of contractor trajectories containing
  both video and action sequences.
- Video input: 128 by 128 pixels at 10 FPS.
- Context: 300 frames, approximately 30 seconds.
- Action frequency: 20 Hz.
- Main action space: nine discrete keys plus two continuous mouse axes. Some
  variants also receive number keys and scroll-wheel actions for hotbar use.
- Largest reported model: 4 billion parameters.

At evaluation time, a goal world is rendered once to obtain a goal image. The
agent is prompted with that image in a different initial world and acts toward
it without goal-specific retraining.

## What the article reports

The evaluation suite contains 104 Minecraft environments spanning basic
movement, building, mechanisms, combat, exploration, problem solving, and
out-of-distribution environments. Pan is compared with STEVE-1 and a VLA
post-trained on the same contractor action dataset.

Reported qualitative behaviors include:

- approaching a target view and oscillating near it to align with the goal;
- stopping near a commanded tower height instead of continuing indefinitely;
- building partial structures from a goal image;
- using mechanisms such as farming, fishing, portals, buckets, and tools;
- exploring for visually specified objects using contextual clues;
- fighting with a learned hit-retreat-return pattern; and
- generalizing to handmade environments absent from training.

The article also shows goal-image exploitation. For example, the model may
reproduce the desired viewpoint by taking an unintended route. Pantograph calls
this reward hacking by analogy, although the model was trained offline without
an online numerical reward.

Scaling improves the harder semantic and dexterous categories. The article
also reports limitations: inconsistent mechanism use, weak recovery from
building mistakes, failures on some combat and tool-switching tasks, and a
short context that restricts training goals to about 30 seconds away.

## Why goal conditioning rather than numerical reward

The same pretrained model can receive new image goals at inference without
retraining a task-specific reward policy. The authors contrast this with agents
trained on a fixed set of numerical objectives. Goal conditioning is presented
as a general self-supervised objective for observation trajectories, analogous
in scalability to next-token prediction but aimed at goal achievement.

The article distinguishes this from plain world-model learning. Predicting the
effect of a move can be easy while knowing which moves reach an arbitrary target
can be hard. Pantograph suggests that goal conditioning and world models could
be combined, but does not report doing planning inside a world model for Pan-1.

## What is not disclosed

The article does **not** specify:

- the exact neural architecture;
- the exact objective or weighting of its components;
- how the goal-conditioned value and next-frame distribution are parameterized;
- the distribution of goal offsets beyond being inside the 300-frame context;
- the precise action post-training objective;
- whether the final action head explicitly consumes value, predicted next
  frames, or another internal representation; or
- enough inference detail to reconstruct the implementation.

Do not present any proposed implementation choice for those items as a fact
about Pan-1.

## Boundary for this continuous-control family

Pan-1 itself relies on massive offline video pretraining and later action-labeled
post-training. This repository family instead requires learning from online
continuous-control experience with no pretraining dataset. That is an adaptation,
not a reproduction.

The faithful transferable principles are:

- reward-free hindsight goals;
- goal occupancy/successor learning rather than task-return prediction;
- observation-only goal representations;
- separate learning of goal-directed state structure and actions; and
- direct goal-conditioned action inference without per-action search.

Important limits of the analogy:

- In Pan pretraining, a goal is an achieved future frame. At inference it is an
  externally provided image. The article does not learn a universal “best
  imaginable state.”
- An impossible aspirational goal has no demonstrated successor occupancy. Its
  construction is additional research. This family explicitly uses realized
  environment reward to train the goal proposer while keeping the Pan-like
  follower reward-free.
- Hindsight learning can teach how to reach achieved states, but by itself does
  not define that fast forward locomotion is preferable to standing, moving
  backward, or any other reachable behavior. This family adds an explicit,
  isolated reward-trained goal selector to supply that ordering.
- Benchmark reward may influence only the detached reward-prediction head and,
  through that frozen head, the goal-prediction head. It cannot train the world
  model, successor model, or direct action policy.

## References named by the article

- Andrychowicz et al., *Hindsight Experience Replay* (NeurIPS 2017).
- Eysenbach et al., *Contrastive Learning as Goal-Conditioned Reinforcement
  Learning* (NeurIPS 2022).
- Janner, Mordatch, and Levine, *Generative Temporal Difference Learning for
  Infinite-Horizon Prediction* (NeurIPS 2020).
- Eysenbach and Zhang, *Generative AI Meets Reinforcement Learning* (ICML 2025
  tutorial notes).
- Tirinzoni et al., *Meta Motivo* (ICLR 2025).
