# Current Level 1 protocol

The candidate metric is `level1-atomic-v4-rgb-pose-target`. Definitions and thresholds are implemented in `scenarios.py`, `focused.py`, `atomic.py`, `dependencies.py`, `reader_v2.py`, `motion.py`, `persistence.py` and `rejection.py`; migration does not change those files.

## Controlled interventions

Craftax-Classic 1.5.0 constructs the prescribed inventory, facing, target blocks, nearby stations and (for combat) adjacent zombie. The engine's actual transitions verify expected fixture facts. The existing diagnostic uses 27 scenario types at seed 0; focused uses 14 types × 10 seeds; controlled movement uses 5 × 10; table placement uses 1 × 10. Lists overlap. Focused variants change surplus/missing/irrelevant resources, peripheral terrain and daytime offsets without changing feasibility. Movement fixtures have seeded asymmetric landmarks and a clear reference route.

One real RGB frame with incoming NOOP initializes the model. There are no preparatory engine steps. Sixteen generated steps follow each fixture's actions with NOOP padding. Selected and NOOP/alternative branches share the start and model sampling keys. Terminal reference episodes include their terminal frame; later padding is not scored or animated. Static targets and dynamic entities are reported separately. Spontaneous generated animal/enemy presence alone is not a failure.

Only RGB observations supply generated facts. Actions and real hidden state do not fill missing readings. A tool-creation verdict, its ingredient-consumption verdict and unrelated inventory diagnostics are separate. Zombie killing concerns the original target's first-step outcome; disappearing equipment is a different diagnostic. Independent rule failures do not contaminate one another.

## Verdicts and dependencies

Each atomic stage is PASS, FAIL or UNKNOWN from readable expected/observed facts. A confident mismatch takes precedence over an unreadable fact. Scenario overall combines explicitly required stages only. Diagnostics stay separate. Persistence requires observed creation/harvest; return-dependent object checks also require localization and visibility. Failed or unknown prerequisites yield NOT_TESTED; a branch that never creates the target can be NOT_APPLICABLE. Unseen or occluded objects are not automatically absent.

Scenario success is PASS / attempted; decision coverage is (PASS + FAIL) / attempted. UNKNOWN remains in that denominator. Atomic success is PASS / eligible; decision coverage is decided / eligible; eligibility coverage is eligible / applicable. NOT_TESTED reduces eligibility coverage, while NOT_APPLICABLE is excluded. Category macro-averages weight categories equally. Report all three rates rather than only accuracy among decided outcomes.

## Conditional departure sampling

The fixed route is `PLACE_TABLE → UP × 6 → DOWN × 6`. Inspect departure evidence only. Retry only if a newly revealed, stable solid obstacle is positively shown to block the route, without contradictory movement evidence or localization ambiguity. Stop at the first non-rejected attempt, including ambiguous departures and visible failures; ELIGIBLE means retained, not certified unobstructed. Return failure or table loss never triggers a retry. Default budget is 20 total samples per engine seed, incrementing model seeds by 1. Preserve first, retained, rejected and all-attempt cohorts separately.

The saved example contains 22 attempts, including ten reused ordinary first attempts and twelve retries. Twelve are rejected and ten retained. Six retained cases have an observed return and absent table; four other scenario failures leave after-return persistence NOT_TESTED. Conditional success is not unconditional performance.

## Limits

The independent generated-image validation of new pose/sleep/target-identity readers and two seed-6 translations is pending. General termination can be RGB-unobservable. Occlusion, malformed HUDs, repeated/rewritten terrain and stochastic saplings may require UNKNOWN. Local camera integration does not establish a persistent global map. These controlled tests do not establish long-horizon gameplay or RL competence.
