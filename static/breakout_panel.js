// The Breakout ontology panel — a domain adapter over the shared editor in
// ontology_core.js. The same file, structurally, as ontology_panel.js is for fleet; what
// differs is the vocabulary, the ABox row, and — importantly — the promise the toggle makes.
//
// READ THE `note` STRING BEFORE CHANGING THE TOGGLE. fleet's says enforcement "should not
// change any decision", which holds there by construction. It CANNOT hold here, and
// claiming it would be false: once the ball is provably gone, left/stay/right has no
// correct answer, so the option set IS the question. Gating it necessarily changes what is
// asked. What survives is weaker and worth stating precisely — see the note.

import { TBOX, VOCABULARY, IMPLEMENTED_FORMS, aboxFor, canCatch } from "./breakout_ontology.js";
import { mountTBoxEditor } from "./ontology_core.js";
import { BAND_ENTER_Y, LAST_BRICK_BOTTOM, BALL_R, measure, rolloutReachable } from "./breakout_sim.js";

// Snapshot at construction time. Read it before the panel is ever mounted, so a stray edit
// during a session is not what `reset` restores you to.
const PRISTINE = TBOX.map(a => ({ ...a, value: Array.isArray(a.value) ? [...a.value] : a.value }));

export function mountBreakoutPanel({ root, getSim, onChange }) {
  return mountTBoxEditor({
    root,
    adapter: {
      tbox: TBOX,
      pristine: PRISTINE,
      implementedForms: IMPLEMENTED_FORMS,
      vocabulary: VOCABULARY,
      getSim,
      aboxFor,
      // Breakout's counterpart to fleet's seat-limit agreement line. There is nothing to
      // disagree with here — the reachability verdict is computed, not configured — so
      // this reports the one thing a reader can act on: how much of the arena the ontology
      // can have an opinion about at all.
      notes: sim => {
        if (!sim.ball) return [{ ok: true, text: "no ball in play yet" }];
        const reachable = rolloutReachable(sim.ball, sim.paddleX);
        if (reachable !== null) {
          return [{
            ok: true,
            text: "the ontology has a verdict for this ball (below the bricks, descending, " +
                  "above paddle level)",
          }];
        }
        const aboveBricks = sim.ball.y - BALL_R <= LAST_BRICK_BOTTOM;
        return [{
          ok: true,
          text: aboveBricks
            ? `no verdict above the bricks (y > ${LAST_BRICK_BOTTOM + BALL_R}) — a brick may ` +
              "still deflect the ball, so reachability is genuinely unknown, not false"
            : "no verdict for this ball — outside the decidable band, so nothing is enforced",
        }];
      },
      enforceLabel: " suppress questions the ontology says are hopeless",
      enforceChecked: sim => !!sim.config.enforceReachability,
      // The toggle does not change any arithmetic. It changes whether the QUESTION is asked,
      // which is the difference fleet does not have — see the header.
      setEnforce: (sim, on) => { sim.config.enforceReachability = on; },
      note: "Off (the default) this demo asks CJet about every ball, exactly as it always " +
            "has — the ontology still shows its verdict, it just does not gate anything. " +
            "Unlike the fleet demo, turning this on IS a behaviour change: when the ball is " +
            "provably unreachable there is no correct answer among left/stay/right, so " +
            "suppressing the question changes what is asked. What it must never do is " +
            "suppress a question whose answer could have mattered. That is checked " +
            "deterministically by `node --test test_breakout_ontology.mjs`, which " +
            "forward-simulates the real physics and asserts zero false 'unreachable' " +
            "verdicts. Note also that above the bricks the ontology has no opinion at all, " +
            "and says so rather than guessing.",
      onChange,
    },
  });
}

/** Whether the question should be asked, as the decision loop sees it. */
export function shouldAsk(sim) {
  if (!sim.config.enforceReachability) return true;
  // `ok: true` covers both "catchable" and "no verdict exists". Only a definite
  // unreachable suppresses, which is the one-directional guarantee.
  return canCatch(sim).ok;
}

export { BAND_ENTER_Y, measure };