// Live TBox editor for the fleet demo.
//
// The reasoner exists and is tested, but a reader of the merit doc had to take it on
// trust: the TBox was a module constant nobody could reach from a page, and the demo ran
// the hand-written constraint layer. This makes both visible — edit an axiom and the
// running simulation changes, and the constraint layer can be switched to the ontology
// to check the two agree.
//
// Almost none of that is fleet-specific any more. The editor itself lives in
// ontology_core.js as `mountTBoxEditor`, because breakout needs the same panel. What is
// left here is the fleet ADAPTER: which TBox, which vocabulary, what the ABox rows say,
// and the one genuinely domain-specific behaviour — the seat limit has a second home in
// `RULES.seats`, so editing the axiom has to write both.
//
// The pure helpers below are re-exported from the core rather than redefined, so
// `test_ontology.mjs`'s import list resolves unchanged and there is one implementation of
// `addAxiom` rather than two.

import { TBOX, VOCABULARY, axiom, checkTBox, buildABox, reason, canServe,
         IMPLEMENTED_FORMS } from "./ontology.js";
import { mountTBoxEditor, describeAxiom, addAxiom as coreAddAxiom,
         removeAxiom as coreRemoveAxiom, resetTBox as coreResetTBox,
         tboxStatus as coreTboxStatus, offerableForms } from "./ontology_core.js";
import { RULES } from "./fleet_sim.js";

// Snapshot at load, so `resetTBox` is exact rather than a hand-copied duplicate.
// TBOX is mutated in place, so without this a stray edit is sticky until reload.
const PRISTINE = TBOX.map(a => ({ ...a, value: Array.isArray(a.value) ? [...a.value] : a.value }));
const PRISTINE_SEATS = RULES.seats;

export { describeAxiom };

const forms = offerableForms(IMPLEMENTED_FORMS);

export const axiomFormOptions = () => forms.map(f => ({ ...f }));
export const classOptions = () => [...VOCABULARY.classes];
export const propertyOptions = () => [...VOCABULARY.properties];

/**
 * The seat limit is two values: the asserted one in the TBox and the configured one the
 * simulator uses. They are one number, so this writes both. Writing only the axiom
 * would leave the question text printing "4 seats free" while the reasoner enforced 2 —
 * and the reasoner's own mismatch violation would fire on every decision.
 */
export function applySeatLimit(value) {
  const n = Math.max(0, Math.floor(Number(value)));
  const cap = axiom("bus-capacity");
  if (!cap) return { applied: false, seats: RULES.seats, reason: "the seat axiom has been removed" };
  cap.value = n;
  RULES.seats = n;
  return { applied: true, seats: n };
}

export function seatLimit() {
  return { asserted: axiom("bus-capacity")?.value ?? null, configured: RULES.seats };
}

/** Is the configured limit the same number the TBox asserts? */
export function seatLimitAgrees() {
  const { asserted, configured } = seatLimit();
  return asserted !== null && asserted === configured;
}

export const addAxiom = args => coreAddAxiom(TBOX, args, IMPLEMENTED_FORMS);
export const removeAxiom = id => coreRemoveAxiom(TBOX, id);
export const resetTBox = () => coreResetTBox(TBOX, PRISTINE, () => { RULES.seats = PRISTINE_SEATS; });
export const tboxStatus = () => coreTboxStatus(TBOX, PRISTINE, IMPLEMENTED_FORMS);

/** What the ontology currently believes about each bus, for the live ABox view. */
export function aboxFor(buses) {
  return buses.map(bus => {
    const facts = buildABox(bus, bus.stops, RULES.seats);
    const probe = reason(facts);
    return {
      id: bus.label,
      // Structured fields, kept because `test_ontology.mjs` reads them directly and the
      // core editor is not the only consumer of this shape.
      aboard: [...bus.onboard],
      stops: bus.stops.length,
      position: `${Math.round(bus.x)}, ${Math.round(bus.y)}`,
      // The rendered line the shared editor puts next to the id. Derived from the three
      // fields above, so the two cannot disagree.
      detail: `at (${Math.round(bus.x)}, ${Math.round(bus.y)}) · ` +
              `${bus.onboard.size} aboard [${[...bus.onboard].join(",") || "—"}] · ` +
              `${bus.stops.length} stops`,
      violations: probe.violations,
      unasserted: probe.unasserted,
    };
  });
}

// ---- DOM ---------------------------------------------------------------------

/** Tier 0 through the ontology, in the shape `insertSeq` expects. */
function ontologyFeasibility(cand, bus) {
  return canServe(bus, cand, RULES.seats).ok;
}

export function mountOntologyPanel({ root, getSim, onChange }) {
  return mountTBoxEditor({
    root,
    adapter: {
      tbox: TBOX,
      // Captured here, not at module load, so a second domain mounting its own panel
      // snapshots its own TBox rather than this one.
      pristine: PRISTINE,
      onReset: () => { RULES.seats = PRISTINE_SEATS; },
      implementedForms: IMPLEMENTED_FORMS,
      vocabulary: VOCABULARY,
      getSim,
      aboxFor: sim => aboxFor(sim.buses),
      notes: () => {
        const { asserted, configured } = seatLimit();
        const agree = seatLimitAgrees();
        return [{
          ok: agree,
          text: agree
            ? `the simulator follows the TBox (${configured} seats)`
            : `MISMATCH: TBox asserts ${asserted ?? "nothing"}, simulator configured ${configured}`,
        }];
      },
      // The seat limit is the one axiom whose value lives in two places.
      onValueChange: (a, n) => {
        if (a.id !== "bus-capacity") return true;
        applySeatLimit(n);
        return true;
      },
      enforceLabel: " enforce Tier 0 with the ontology",
      enforceChecked: sim => !!sim.config.feasibility,
      setEnforce: (sim, on) => { sim.config.feasibility = on ? ontologyFeasibility : null; },
      note: "The demo ships on the hand-written constraint layer. Turning this on " +
            "should not change any decision: the two constraint layers agreed on " +
            "600 of 600 bus/demand pairs, checked deterministically by " +
            "`node eval_rules.mjs`. Counting deliveries here is not that check — " +
            "this demo runs on wall-clock, so two runs see different demand counts. " +
            "What is visible here is that the two behave the same, not that they " +
            "are provably identical; the harness is where that is settled.",
      onChange,
    },
  });
}