// The rules of a spring model that change who is bonded and which particles exist, gathered so
// that a scenario can say which it uses and a graph can be built from them.
import type { DivisionRule, LeaderRule } from "./growth";
import type { BreakRule, FormRule } from "./turnover";

/** Each rule that is given runs every step, after the forces, in this order. */
export interface SpringRules {
  /** Break bonds by strain or lifetime. Bonds to dead particles always go. */
  readonly breakRule?: BreakRule;
  readonly formRule?: FormRule;
  readonly division?: DivisionRule;
  readonly leaders?: LeaderRule;
  /** Leader selection runs once every this many steps, with its `tau` scaled to match. */
  readonly leaderEvery?: number;
}
