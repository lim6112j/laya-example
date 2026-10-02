// Live TBox editor for the fleet demo.
//
// The reasoner exists and is tested, but a reader of the merit doc had to take it on
// trust: the TBox was a module constant nobody could reach from a page, and the demo ran
// the hand-written constraint layer. This makes both visible — edit an axiom and the
// running simulation changes, and the constraint layer can be switched to the ontology
// to check the two agree.
//
// The pure helpers below are exported separately from the DOM so they can be tested
// without a browser; `mountOntologyPanel` is the only part that touches a document.

import { TBOX, VOCABULARY, axiom, checkTBox, buildABox, reason, canServe } from "./ontology.js";
import { RULES } from "./fleet_sim.js";

// Snapshot at load, so `resetTBox` is exact rather than a hand-copied duplicate.
// TBOX is mutated in place, so without this a stray edit is sticky until reload.
const PRISTINE = TBOX.map(a => ({ ...a, value: Array.isArray(a.value) ? [...a.value] : a.value }));
const PRISTINE_SEATS = RULES.seats;

const forms = [
  { value: "maxCardinality", label: "at most (maxCardinality)", numeric: true },
  { value: "minCardinality", label: "at least (minCardinality)", numeric: true },
  { value: "range", label: "values are of class (range)", numeric: false },
];

export const axiomFormOptions = () => forms.map(f => ({ ...f }));
export const classOptions = () => [...VOCABULARY.classes];
export const propertyOptions = () => [...VOCABULARY.properties];

/** A readable rendering of one axiom, for the list and for the TBox self-check text. */
export function describeAxiom(a) {
  if (!a) return "(removed)";
  if (a.form === "maxCardinality") return `${a.subject} ⊑ ≤${a.value} ${a.property}`;
  if (a.form === "minCardinality") return `${a.subject} ⊑ ≥${a.value} ${a.property}`;
  if (a.form === "range") return `${a.subject}.${a.property} ⊑ ${a.value}`;
  if (a.form === "disjoint") return `${a.value.join(" ⊥ ")}`;
  return JSON.stringify(a);
}

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

export function addAxiom({ form, subject, property, value }) {
  // Compared on form+subject+property, not on id: the shipped axioms carry hand-written
  // ids ("bus-capacity", not "maxCardinality-Bus-onboardPassenger"), so an id comparison
  // would let a semantic duplicate straight through.
  const clash = TBOX.find(a =>
    a.form === form && a.subject === subject && a.property === property);
  if (clash) return { ok: false, error: `${clash.id} already asserts that` };
  const numeric = forms.find(f => f.value === form)?.numeric;
  const parsed = numeric ? Math.floor(Number(value)) : value;
  if (parsed === undefined || parsed === null || (numeric && Number.isNaN(parsed))) {
    return { ok: false, error: "that value is not valid for this axiom form" };
  }
  TBOX.push({ id: `${form}-${subject}-${property}`, form, subject, property, value: parsed });
  return { ok: true, id: `${form}-${subject}-${property}` };
}

export function removeAxiom(id) {
  const i = TBOX.findIndex(a => a.id === id);
  if (i < 0) return { ok: false, error: "no such axiom" };
  TBOX.splice(i, 1);
  return { ok: true, id };
}

export function resetTBox() {
  TBOX.splice(0, TBOX.length, ...PRISTINE.map(a => ({ ...a })));
  RULES.seats = PRISTINE_SEATS;
  return { ok: true };
}

/** Consistency of the axiom set, plus whether the declared axioms are all present. */
export function tboxStatus() {
  const check = checkTBox(TBOX);
  const missing = PRISTINE.filter(p => !TBOX.some(a => a.id === p.id))
    .map(p => ({ id: p.id, text: describeAxiom(p) }));
  return { ...check, missing };
}

/** What the ontology currently believes about each bus, for the live ABox view. */
export function aboxFor(buses) {
  return buses.map(bus => {
    const facts = buildABox(bus, bus.stops, RULES.seats);
    const probe = reason(facts);
    return {
      id: bus.label,
      aboard: [...bus.onboard],
      stops: bus.stops.length,
      position: `${Math.round(bus.x)}, ${Math.round(bus.y)}`,
      violations: probe.violations,
      unasserted: probe.unasserted,
    };
  });
}

// ---- DOM ---------------------------------------------------------------------

const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  // `append(null)` inserts the string "null" rather than skipping, and the conditional
  // children below are all `cond ? node : null`.
  for (const k of kids) { if (k !== null && k !== undefined) node.append(k); }
  return node;
};

/**
 * Mount the editor into `root`. `getSim` supplies the live sim and `onChange` is called
 * after every mutation so the host can restart the run and re-render the fleet.
 */
export function mountOntologyPanel({ root, getSim, onChange }) {
  // Created once, outside rerender(), so it survives an axiom edit and can be refreshed
  // independently. The ABox is a live view of the running sim; the axiom list is a form.
  const aboxHost = el("div", { className: "abox" });

  /** Repaint only the ABox. Cheap enough to call a few times a second. */
  const refreshABox = () => {
    const rows = aboxFor(getSim().buses);
    aboxHost.replaceChildren(...rows.map(b => el("div", { className: "abrow" },
      el("strong", { textContent: b.id }),
      ` at (${b.position}) · ${b.aboard.length} aboard [${b.aboard.join(",") || "—"}] · ${b.stops} stops`,
      b.violations.length
        ? el("span", { className: "bad", textContent: ` · ${b.violations.length} violation(s)` })
        : null,
      b.unasserted.length ? el("span", { className: "warn", textContent: " · unasserted" }) : null,
    )));
  };

  const rerender = () => {
    const sim = getSim();
    const status = tboxStatus();
    const agree = seatLimitAgrees();
    const { asserted, configured } = seatLimit();

    root.replaceChildren();

    // --- declared axioms
    const list = el("div", { className: "axioms" });
    for (const a of [...TBOX]) {
      const numeric = forms.some(f => f.value === a.form && f.numeric);
      // A disjoint axiom's value is its class list, which `describeAxiom` already
      // renders — a second, truncated copy of it in the value column helps nobody.
      const value = a.form === "disjoint"
        ? el("span", { className: "axnone", textContent: "declared" })
        : numeric
        ? el("input", { type: "number", min: "0", value: String(a.value), className: "axval" })
        : el("span", { className: "axval axfixed", textContent: String(a.value) });
      if (numeric) {
        value.addEventListener("change", () => {
          // Only the seat limit has a second home (RULES.seats); everything else is
          // declared once and read from here.
          if (a.id === "bus-capacity") applySeatLimit(value.value);
          else a.value = Math.max(0, Math.floor(Number(value.value)));
          onChange(a.id);
        });
      }
      list.append(el("div", { className: "axiom" },
        el("code", { className: "axtext", textContent: describeAxiom(a) }),
        value,
        el("button", {
          className: "axdel", title: "remove this axiom", textContent: "×",
          onclick: () => { removeAxiom(a.id); onChange(a.id); },
        }),
        a.comment ? el("div", { className: "axnote", textContent: a.comment }) : null,
      ));
    }
    root.append(el("h2", { textContent: `Ontology — TBox (${TBOX.length} axioms)` }), list);

    if (status.missing.length) {
      root.append(el("div", { className: "warn" },
        `not asserted: ${status.missing.map(m => m.id).join(", ")} — ` +
        `the ontology no longer enforces ${status.missing.map(m => m.text).join("; ")} ` +
        `(shown as declared by default)`));
    }
    root.append(el("div", { className: status.satisfiable ? "ok" : "bad" },
      status.satisfiable
        ? "consistency: satisfiable — the axiom set is coherent"
        : "UNSATISFIABLE: " + status.unsatisfiable.map(u => `${u.id} (${u.reason})`).join("; ")));

    const seatNote = agreed => agreed
      ? `the simulator follows the TBox (${configured} seats)`
      : `MISMATCH: TBox asserts ${asserted ?? "nothing"}, simulator configured ${configured}`;
    root.append(el("div", { className: agree ? "ok" : "bad", textContent: seatNote(agree) }));

    // --- add an axiom
    const sel = (options, value) => {
      const s = el("select");
      for (const o of options) {
        const opt = typeof o === "string" ? { value: o, label: o } : o;
        s.append(el("option", { value: opt.value, textContent: opt.label, selected: opt.value === value }));
      }
      return s;
    };
    const fSel = sel(axiomFormOptions(), "minCardinality");
    const sSel = sel(classOptions(), "Bus");
    const pSel = sel(propertyOptions(), "onboardPassenger");
    const vIn = el("input", { type: "number", min: "0", value: "6", className: "axval" });
    const syncValueForForm = () => {
      const f = forms.find(x => x.value === fSel.value);
      if (f.numeric) { vIn.type = "number"; vIn.className = "axval"; }
      else { vIn.type = "text"; vIn.className = "axval axfixed"; vIn.value = "GridCell"; }
    };
    fSel.addEventListener("change", syncValueForForm);
    const msg = el("span", { className: "axmsg" });
    root.append(el("div", { className: "axadd" },
      el("span", { className: "axaddlbl", textContent: "add" }),
      fSel, sSel, pSel, vIn,
      el("button", {
        className: "axbtn", textContent: "+",
        onclick: () => {
          const r = addAxiom({
            form: fSel.value, subject: sSel.value, property: pSel.value,
            value: forms.find(x => x.value === fSel.value)?.numeric ? Number(vIn.value) : vIn.value,
          });
          msg.textContent = r.ok ? `added ${r.id}` : r.error;
          if (r.ok) onChange(r.id);
        },
      }),
      msg,
    ));

    // --- what the ontology currently believes
    // aboxHost is created once and reused across every rerender, so it can be refreshed
    // on a timer without touching the axiom rows — rebuilding the whole panel on a tick
    // would steal focus from the seat-limit input mid-edit.
    root.append(el("h2", { textContent: "Ontology — ABox (what it currently believes)" }),
                aboxHost);

    // --- enforcement toggle
    const toggle = el("input", { type: "checkbox" });
    toggle.checked = !!sim.config.feasibility;
    toggle.addEventListener("change", () => onChange(toggle.checked ? "enforce" : "unenforce"));
    root.append(el("div", { className: "axenforce" }, toggle,
      el("label", { textContent: " enforce Tier 0 with the ontology" })));
    root.append(el("div", { className: "axnote",
      textContent: "The demo ships on the hand-written constraint layer. Turning this on " +
                    "should not change any decision: the two constraint layers agreed on " +
                    "600 of 600 bus/demand pairs, checked deterministically by " +
                    "`node eval_rules.mjs`. Counting deliveries here is not that check — " +
                    "this demo runs on wall-clock, so two runs see different demand counts. " +
                    "What is visible here is that the two behave the same, not that they " +
                    "are provably identical; the harness is where that is settled." }));

    root.append(el("div", { className: "axreset" },
      el("button", { className: "axbtn", textContent: "reset to declared defaults",
                     onclick: () => { resetTBox(); onChange("reset"); } })));
  };

  rerender();
  return { rerender, refreshABox };
}
