#!/usr/bin/env python3
"""
neurological_triage_webhook.py
─────────────────────────────
Dialogflow ES webhook for the Neurological Triage Chatbot.

Queries neurological_triage.owl using rdflib to answer questions about:
  - Symptoms of Alzheimer's, Parkinson's, and ALS
  - Risk factors, protective factors, and genetic factors
  - Differential triage based on reported symptoms
  - Symptom-to-disease typicality mapping

Fixes applied over original version
─────────────────────────────────────
  FIX-1  resolve_disease / resolve_symptom: Dialogflow ES always sends
         entity params as lists even for a single match. Both resolvers
         now unwrap the list and extract the first non-empty string before
         calling .strip(), eliminating the
         "'list' object has no attribute 'strip'" crash.

  FIX-2  handle_differentiate: when the user asks "How is PD different
         from AD?", Dialogflow puts BOTH disease names into the single
         'disease' list param and leaves 'disease2' empty. The handler
         now reads both values from the 'disease' list, so it only
         compares the two requested diseases instead of comparing the
         first disease against every other disease in the ontology.

  FIX-3  SYMPTOM_SYNONYMS map: bridges the gap between natural language
         phrases Dialogflow extracts ("trouble sleeping", "lose sleep")
         and the exact individual names in the ontology
         ("sleep_disturbance"). Add new entries here whenever a new
         Dialogflow entity synonym does not match the ontology URI.

  FIX-4  handle_report_symptoms: normalises symptom param to a list and
         processes each item individually so multi-symptom utterances
         work correctly.

  FIX-5  handle_get_disease_from_symptom: handles list symptom param,
         gracefully returns a no-result message when no symptoms match
         the ontology (e.g. chest pain, shortness of breath), and adds a
         combined scoring summary when multiple symptoms are provided.

  FIX-6  Session symptom accumulation: uses .extend() instead of
         .append() so the reported_symptoms list stays flat across turns.

Setup:
    pip install flask rdflib

Run locally:
    python neurological_triage_webhook.py

Expose with ngrok for Dialogflow:
    ngrok http 5000
    Set webhook URL in Dialogflow Fulfillment to:
    https://<ngrok-id>.ngrok.io/webhook
"""

from flask import Flask, request, jsonify
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from collections import defaultdict
import inspect
import os

app = Flask(__name__)

# ─── Ontology Setup ────────────────────────────────────────────────────────────
OWL_PATH = os.path.join(os.path.dirname(__file__), "neurological_triage.owl")
TRIAGE   = Namespace("http://www.semanticweb.org/neurological-triage#")

g = Graph()
g.parse(OWL_PATH, format="xml")

# Pre-build label -> URI map for fast lookup
label_to_uri: dict = {}
for s, _, o in g.triples((None, RDFS.label, None)):
    label_to_uri[str(o).lower()] = s

# ─── Disease lookup tables ─────────────────────────────────────────────────────
DISEASE_URIS = {
    "alzheimer":                     TRIAGE["alzheimers_disease"],
    "alzheimers":                    TRIAGE["alzheimers_disease"],
    "alzheimer's":                   TRIAGE["alzheimers_disease"],
    "alzheimer's disease":           TRIAGE["alzheimers_disease"],
    "alzheimer disease":             TRIAGE["alzheimers_disease"],
    "ad":                            TRIAGE["alzheimers_disease"],
    "dementia":                      TRIAGE["alzheimers_disease"],
    "parkinson":                     TRIAGE["parkinson_disease"],
    "parkinson's":                   TRIAGE["parkinson_disease"],
    "parkinsons":                    TRIAGE["parkinson_disease"],
    "parkinson's disease":           TRIAGE["parkinson_disease"],
    "parkinson disease":             TRIAGE["parkinson_disease"],
    "pd":                            TRIAGE["parkinson_disease"],
    "als":                           TRIAGE["als_disease"],
    "amyotrophic lateral sclerosis": TRIAGE["als_disease"],
    "motor neuron disease":          TRIAGE["als_disease"],
    "mnd":                           TRIAGE["als_disease"],
    "lou gehrig's disease":          TRIAGE["als_disease"],
}

DISEASE_NAMES = {
    str(TRIAGE["alzheimers_disease"]): "Alzheimer's Disease",
    str(TRIAGE["parkinson_disease"]):  "Parkinson's Disease",
    str(TRIAGE["als_disease"]):        "ALS (Amyotrophic Lateral Sclerosis)",
}

DISEASE_SHORT = {
    str(TRIAGE["alzheimers_disease"]): "Alzheimer's",
    str(TRIAGE["parkinson_disease"]):  "Parkinson's",
    str(TRIAGE["als_disease"]):        "ALS",
}

ALL_DISEASE_URIS = [
    TRIAGE["alzheimers_disease"],
    TRIAGE["parkinson_disease"],
    TRIAGE["als_disease"],
]

# ─── FIX-3: Symptom synonym map ───────────────────────────────────────────────
# Maps Dialogflow entity values (lowercased + underscore-normalised) to the
# exact local name used in the ontology URI or rdfs:label.
# Add new entries here whenever a Dialogflow synonym does not resolve
# automatically to the correct ontology individual.
SYMPTOM_SYNONYMS = {
    # Sleep
    "sleep_disturbance":        "sleep_disturbance",
    "trouble_sleeping":         "sleep_disturbance",
    "can't_sleep":              "sleep_disturbance",
    "cant_sleep":               "sleep_disturbance",
    "lose_sleep":               "sleep_disturbance",
    "losing_sleep":             "sleep_disturbance",
    "insomnia":                 "sleep_disturbance",
    "poor_sleep":               "sleep_disturbance",
    "sleep_problems":           "sleep_disturbance",
    "restless_sleep":           "sleep_disturbance",
    "disrupted_sleep":          "sleep_disturbance",
    "acting_out_dreams":        "sleep_disturbance",
    "act_out_dreams":           "sleep_disturbance",
    "wake_up_at_night":         "sleep_disturbance",
    "difficulty_sleeping":      "sleep_disturbance",
    # Memory
    "memory_loss":              "memory_loss_symptom",
    "forgetting_things":        "memory_loss_symptom",
    "forgetfulness":            "memory_loss_symptom",
    "forget_things":            "memory_loss_symptom",
    "episodic_memory_loss":     "memory_loss_symptom",
    "memory_problems":          "memory_loss_symptom",
    # Tremor
    "resting_tremor":           "resting_tremor_symptom",
    "tremor":                   "resting_tremor_symptom",
    "shaking":                  "resting_tremor_symptom",
    "hand_tremor":              "resting_tremor_symptom",
    "hand_shaking":             "resting_tremor_symptom",
    "shaky_hands":              "resting_tremor_symptom",
    # Muscle / motor
    "muscle_weakness":          "progressive_muscle_weakness_symptom",
    "limb_weakness":            "progressive_muscle_weakness_symptom",
    "weak_muscles":             "progressive_muscle_weakness_symptom",
    "weakness":                 "progressive_muscle_weakness_symptom",
    "muscle_wasting":           "muscle_wasting_symptom",
    "muscle_twitching":         "muscle_twitching_symptom",
    "fasciculations":           "muscle_twitching_symptom",
    # Speech / swallowing
    "slurred_speech":           "slurred_speech_symptom",
    "trouble_speaking":         "slurred_speech_symptom",
    "difficulty_speaking":      "slurred_speech_symptom",
    "speech_problems":          "slurred_speech_symptom",
    "dysarthria":               "dysarthria_symptom",
    "dysphagia":                "dysphagia_symptom",
    "trouble_swallowing":       "dysphagia_symptom",
    "difficulty_swallowing":    "dysphagia_symptom",
    # Cognitive
    "confusion":                "confusion_symptom",
    "getting_confused":         "confusion_symptom",
    "disorientation":           "disorientation_symptom",
    "getting_lost":             "disorientation_symptom",
    "poor_judgement":           "poor_judgement_symptom",
    "impaired_reasoning":       "poor_judgement_symptom",
    # Balance / movement
    "balance_problems":         "balance_problem_symptom",
    "balance_issues":           "balance_problem_symptom",
    "falling":                  "balance_problem_symptom",
    "bradykinesia":             "bradykinesia_symptom",
    "slow_movement":            "bradykinesia_symptom",
    "slowness":                 "bradykinesia_symptom",
    "rigidity":                 "rigidity_symptom",
    "stiffness":                "rigidity_symptom",
    "muscle_rigidity":          "rigidity_symptom",
    # Smell
    "loss_of_smell":            "olfactory_dysfunction_symptom",
    "smell_loss":               "olfactory_dysfunction_symptom",
    "anosmia":                  "olfactory_dysfunction_symptom",
    "cant_smell":               "olfactory_dysfunction_symptom",
    # Mood
    "depression":               "depression_symptom",
    "depressed":                "depression_symptom",
    "mood_changes":             "depression_symptom",
    "anxiety":                  "depression_symptom",
    # Breathing
    "breathing_problems":       "respiratory_impairment_symptom",
    "shortness_of_breath":      "respiratory_impairment_symptom",
    "respiratory_impairment":   "respiratory_impairment_symptom",
    "breathlessness":           "respiratory_impairment_symptom",
}


# ─── Helper utilities ──────────────────────────────────────────────────────────

def _unwrap_param(value) -> str:
    """
    Dialogflow ES sends entity params as lists even for a single match.
    Extract the first non-empty string so callers can safely use .strip().
    Always returns a str (possibly empty), never a list.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        candidates = [v for v in value if v]
        return candidates[0] if candidates else ""
    return value


def _unwrap_param_list(value) -> list:
    """
    Like _unwrap_param but returns ALL non-empty values as a flat list.
    Used when we genuinely need every entity value (e.g. two disease names
    in DifferentiateByDisease).
    """
    if not value:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


def get_label(uri) -> str:
    """Return rdfs:label for a URI, falling back to the local name."""
    for _, _, o in g.triples((uri, RDFS.label, None)):
        return str(o)
    return str(uri).split("#")[-1].replace("_", " ")


def get_definition(uri) -> str:
    """Return triage:definition for a URI if present, otherwise empty string."""
    for _, _, o in g.triples((uri, TRIAGE["definition"], None)):
        return str(o)
    return ""


def resolve_disease(raw_text):
    """
    Map a Dialogflow @Disease entity value to an ontology URI.
    Handles list inputs (FIX-1).
    """
    raw_text = _unwrap_param(raw_text)
    if not raw_text:
        return None
    clean = raw_text.strip().lower()
    if clean in DISEASE_URIS:
        return DISEASE_URIS[clean]
    for key, uri in DISEASE_URIS.items():
        if key in clean or clean in key:
            return uri
    return None


def resolve_symptom(raw_text):
    """
    Map a @Symptom entity value to an ontology URI.
    Resolution order (FIX-1, FIX-3):
      1. Unwrap list to string
      2. Check SYMPTOM_SYNONYMS map
      3. Check rdfs:label index
      4. Try direct TRIAGE[underscore_form] URI construction
    """
    raw_text = _unwrap_param(raw_text)
    if not raw_text:
        return None

    clean      = raw_text.strip().lower()
    normalised = clean.replace(" ", "_")

    # FIX-3: synonym map
    mapped = SYMPTOM_SYNONYMS.get(normalised) or SYMPTOM_SYNONYMS.get(clean)
    if mapped:
        candidate_label = mapped.replace("_", " ")
        if candidate_label in label_to_uri:
            return label_to_uri[candidate_label]
        candidate_uri = TRIAGE[mapped]
        if (candidate_uri, RDF.type, None) in g:
            return candidate_uri

    if clean in label_to_uri:
        return label_to_uri[clean]

    candidate_uri = TRIAGE[normalised]
    if (candidate_uri, RDF.type, None) in g:
        return candidate_uri

    return None


def get_symptoms_of_disease(disease_uri, prop: str) -> list:
    return list(g.objects(disease_uri, TRIAGE[prop]))


def get_symptom_display_list(symptom_uris) -> str:
    labels = sorted(get_label(s) for s in symptom_uris)
    return "\n".join(f"• {l}" for l in labels)


def get_factors_of_disease(disease_uri, prop: str) -> list:
    return list(g.subjects(TRIAGE[prop], disease_uri))


def score_symptoms(reported_symptom_uris: list) -> tuple:
    """
    Score each disease against reported symptom URIs.
    Weights: hasPrimarySymptom=3, hasSymptom=1, hasOverlappingSymptom=0.3,
             moreTypicalOf bonus=2.
    Returns (scores dict, matched dict).
    """
    scores:  dict = defaultdict(float)
    matched: dict = defaultdict(list)

    for s_uri in reported_symptom_uris:
        for disease in g.objects(s_uri, TRIAGE["moreTypicalOf"]):
            scores[str(disease)] += 2.0
            matched[str(disease)].append((get_label(s_uri), "highly typical"))

    for disease in ALL_DISEASE_URIS:
        primary = set(g.objects(disease, TRIAGE["hasPrimarySymptom"]))
        has     = set(g.objects(disease, TRIAGE["hasSymptom"]))
        overlap = set(g.objects(disease, TRIAGE["hasOverlappingSymptom"]))
        for s_uri in reported_symptom_uris:
            if s_uri in primary:
                scores[str(disease)]  += 3.0
                matched[str(disease)].append((get_label(s_uri), "primary symptom"))
            elif s_uri in has:
                scores[str(disease)]  += 1.0
                matched[str(disease)].append((get_label(s_uri), "associated symptom"))
            elif s_uri in overlap:
                scores[str(disease)]  += 0.3
                matched[str(disease)].append((get_label(s_uri), "overlapping symptom"))

    return scores, matched


# ─── Intent Handlers ──────────────────────────────────────────────────────────

def handle_start_triage(params):
    return (
        "Starting neurological triage. I'll guide you through key symptoms to "
        "differentiate between Alzheimer's disease, Parkinson's disease, and ALS.\n\n"
        "Let's begin. Does the patient have any tremor or shaking, particularly at rest "
        "when the limb is not being used?"
    )


def handle_report_symptoms(params, session_params):
    """FIX-4: normalise symptom param to a list; process each item individually."""
    sym_val = params.get("symptom", "")
    if isinstance(sym_val, str):
        sym_val = [sym_val] if sym_val else []
    sym_val = [s for s in sym_val if s]

    if not sym_val:
        return (
            "Could you describe the symptom in more detail? "
            "For example: resting tremor, limb weakness, memory loss, or trouble sleeping."
        )

    responses = []
    for single_sym in sym_val:
        s_uri = resolve_symptom(single_sym)
        if not s_uri:
            responses.append(
                f"I noted '{single_sym}' but couldn't find an exact match in the ontology. "
                "Could you rephrase? For example: resting tremor, limb weakness, or episodic memory loss."
            )
            continue

        label      = get_label(s_uri)
        defn       = get_definition(s_uri)
        typical_of = list(g.objects(s_uri, TRIAGE["moreTypicalOf"]))

        typical_str = ""
        if typical_of:
            names = [DISEASE_SHORT.get(str(d), get_label(d)) for d in typical_of]
            typical_str = f" This symptom is most typical of {', '.join(names)}."

        appearances = []
        for disease_uri in ALL_DISEASE_URIS:
            primary = set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"]))
            has     = set(g.objects(disease_uri, TRIAGE["hasSymptom"]))
            overlap = set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
            dname   = DISEASE_SHORT[str(disease_uri)]
            if s_uri in primary:
                appearances.append(f"{dname} (primary)")
            elif s_uri in has:
                appearances.append(f"{dname} (associated)")
            elif s_uri in overlap:
                appearances.append(f"{dname} (overlapping)")

        appear_str = f" Seen in: {', '.join(appearances)}." if appearances else ""
        entry = f"Noted: {label}.{typical_str}{appear_str}"
        if defn:
            entry += f"\n\nDefinition: {defn}"
        responses.append(entry)

    combined  = "\n\n".join(responses)
    combined += "\n\nShall I continue triage? Tell me another symptom or say 'give me the assessment'."
    return combined


def handle_get_primary_symptoms(params):
    disease_uri = resolve_disease(params.get("disease", ""))
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."
    primary = get_symptoms_of_disease(disease_uri, "hasPrimarySymptom")
    dname   = DISEASE_NAMES.get(str(disease_uri), "")
    if not primary:
        return f"No primary symptoms found for {dname} in the ontology."
    return f"Primary symptoms of {dname}:\n{get_symptom_display_list(primary)}"


def handle_get_all_symptoms(params):
    disease_uri = resolve_disease(params.get("disease", ""))
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."
    primary     = get_symptoms_of_disease(disease_uri, "hasPrimarySymptom")
    associated  = get_symptoms_of_disease(disease_uri, "hasSymptom")
    overlapping = get_symptoms_of_disease(disease_uri, "hasOverlappingSymptom")
    dname       = DISEASE_NAMES.get(str(disease_uri), "")
    parts = []
    if primary:
        parts.append(f"Primary symptoms:\n{get_symptom_display_list(primary)}")
    if associated:
        parts.append(f"Associated symptoms:\n{get_symptom_display_list(associated)}")
    if overlapping:
        parts.append(f"Overlapping symptoms (also in other diseases):\n{get_symptom_display_list(overlapping)}")
    if not parts:
        return f"No symptoms found for {dname}."
    return f"Symptom profile for {dname}:\n\n" + "\n\n".join(parts)


def handle_get_symptoms_by_category(params):
    cat_val     = _unwrap_param(params.get("symptomCategory", "")).lower()
    disease_uri = resolve_disease(params.get("disease", ""))

    cat_map = {
        "motor":                   TRIAGE["MotorSymptom"],
        "cognitive":               TRIAGE["CognitiveSymptom"],
        "behavioural":             TRIAGE["BehaviouralPsychiatricSymptom"],
        "behavioural_psychiatric": TRIAGE["BehaviouralPsychiatricSymptom"],
        "psychiatric":             TRIAGE["BehaviouralPsychiatricSymptom"],
        "autonomic":               TRIAGE["AutonomicSymptom"],
        "speech":                  TRIAGE["SpeechSwallowingSymptom"],
        "swallowing":              TRIAGE["SpeechSwallowingSymptom"],
        "speech_swallowing":       TRIAGE["SpeechSwallowingSymptom"],
        "respiratory":             TRIAGE["RespiratorySymptom"],
        "sleep":                   TRIAGE["SleepSymptom"],
        "sensory":                 TRIAGE["SensorySymptom"],
    }

    cat_uri = next((uri for key, uri in cat_map.items() if key in cat_val), None)
    if not cat_uri:
        return (
            f"I don't recognise the category '{cat_val}'. "
            "Try: motor, cognitive, autonomic, speech/swallowing, respiratory, sleep, behavioural, or sensory."
        )

    cat_label         = get_label(cat_uri)
    matching_symptoms = list(g.subjects(TRIAGE["belongsToSymptomCategory"], cat_uri))

    if not matching_symptoms:
        return f"No {cat_label} symptoms found in the ontology."

    if disease_uri:
        all_disease_syms = (
            set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"])) |
            set(g.objects(disease_uri, TRIAGE["hasSymptom"]))         |
            set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
        )
        matching_symptoms = [s for s in matching_symptoms if s in all_disease_syms]
        dname  = DISEASE_SHORT.get(str(disease_uri), "")
        prefix = f"{cat_label} symptoms of {dname}"
    else:
        prefix = f"{cat_label} symptoms across all three diseases"

    if not matching_symptoms:
        return f"No {cat_label} symptoms found for {DISEASE_SHORT.get(str(disease_uri), '')}."

    return f"{prefix}:\n{get_symptom_display_list(matching_symptoms)}"


def handle_get_disease_from_symptom(params):
    """FIX-5: list normalisation, graceful no-result, combined scoring."""
    sym_val = params.get("symptom", "")
    if isinstance(sym_val, str):
        sym_val = [sym_val] if sym_val else []
    sym_val = [s for s in sym_val if s]

    if not sym_val:
        return (
            "Please describe a symptom. "
            "For example: resting tremor, limb weakness, or episodic memory loss."
        )

    resolved   = []
    unresolved = []
    for s in sym_val:
        uri = resolve_symptom(s)
        if uri:
            resolved.append((s, uri))
        else:
            unresolved.append(s)

    if not resolved:
        symptom_list = ", ".join(f"'{s}'" for s in sym_val)
        return (
            f"I couldn't find any specific neurological information for {symptom_list} "
            "in the ontology. These symptoms are not associated with Alzheimer's Disease, "
            "Parkinson's Disease, or ALS in our knowledge base. "
            "Please consult a general practitioner for an assessment."
        )

    lines = []
    for _raw, s_uri in resolved:
        label      = get_label(s_uri)
        defn       = get_definition(s_uri)
        typical_of = list(g.objects(s_uri, TRIAGE["moreTypicalOf"]))
        appearances = []
        for disease_uri in ALL_DISEASE_URIS:
            primary = set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"]))
            has     = set(g.objects(disease_uri, TRIAGE["hasSymptom"]))
            overlap = set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
            dname   = DISEASE_SHORT[str(disease_uri)]
            if s_uri in primary:
                appearances.append(f"• {dname}: primary symptom")
            elif s_uri in has:
                appearances.append(f"• {dname}: associated symptom")
            elif s_uri in overlap:
                appearances.append(f"• {dname}: overlapping symptom")

        entry = label
        if defn:
            entry += f"\n{defn}"
        if typical_of:
            names  = [DISEASE_SHORT.get(str(d), get_label(d)) for d in typical_of]
            entry += f"\n\nMost typical of: {', '.join(names)}."
        if appearances:
            entry += "\n\nAppears in:\n" + "\n".join(appearances)
        elif not typical_of:
            entry += "\n\nThis symptom was not found linked to any of the three diseases."
        lines.append(entry)

    if unresolved:
        lines.append(
            "Note: I couldn't find these symptoms in the ontology: "
            + ", ".join(f"'{s}'" for s in unresolved)
        )

    if len(resolved) > 1:
        sym_uris        = [uri for _, uri in resolved]
        scores, matched = score_symptoms(sym_uris)
        ranked          = sorted(ALL_DISEASE_URIS, key=lambda d: scores.get(str(d), 0), reverse=True)
        top_score       = scores.get(str(ranked[0]), 0)
        if top_score > 0:
            rank_labels = ["Most consistent with", "Second consideration", "Less consistent with"]
            summary     = ["\n-- Combined Assessment --"]
            for i, d_uri in enumerate(ranked):
                score = scores.get(str(d_uri), 0)
                if score == 0:
                    continue
                summary.append(f"{rank_labels[i]}: {DISEASE_NAMES[str(d_uri)]} (score: {score:.1f})")
            summary.append(
                "\n(!) This is an informational aid only. "
                "Please consult a neurologist for clinical assessment."
            )
            lines.append("\n".join(summary))

    return "\n\n".join(lines)


def handle_get_overlapping(params):
    disease_uri = resolve_disease(params.get("disease", ""))

    if disease_uri:
        overlapping = get_symptoms_of_disease(disease_uri, "hasOverlappingSymptom")
        dname       = DISEASE_NAMES.get(str(disease_uri), "")
        if not overlapping:
            return f"No overlapping symptoms documented for {dname}."
        return (
            f"Overlapping symptoms for {dname} (also seen in other diseases):\n"
            + get_symptom_display_list(overlapping)
        )

    result_lines = []
    for d in ALL_DISEASE_URIS:
        overlapping = get_symptoms_of_disease(d, "hasOverlappingSymptom")
        if overlapping:
            dname  = DISEASE_SHORT[str(d)]
            labels = sorted(get_label(s) for s in overlapping)
            result_lines.append(f"{dname}: {', '.join(labels)}")

    return (
        "Overlapping symptoms (appear in multiple diseases, "
        "making differential diagnosis harder):\n\n"
        + "\n".join(result_lines)
    )


def handle_differentiate(params):
    """
    Collect disease values from both 'disease' and 'disease2' params,
    deduplicate, then generate ALL unique pairwise comparisons using
    itertools.combinations.

    1 disease  -> compare against all other two (2 pairs total)
    2 diseases -> compare only those two (1 pair)
    3 diseases -> compare every unique pair: A vs B, A vs C, B vs C (3 pairs)

    The previous version anchored on d1 and compared d1 vs each other disease,
    producing PD vs ALS and PD vs AD but missing ALS vs AD when all three
    were provided.
    """
    from itertools import combinations

    all_vals = (
        _unwrap_param_list(params.get("disease",  "")) +
        _unwrap_param_list(params.get("disease2", ""))
    )

    # Resolve and deduplicate while preserving order
    seen: set = set()
    resolved_uris: list = []
    for val in all_vals:
        uri = resolve_disease(val)
        if uri and str(uri) not in seen:
            seen.add(str(uri))
            resolved_uris.append(uri)

    if not resolved_uris:
        return "Please specify at least one disease to compare: Alzheimer's, Parkinson's, or ALS."

    # Build the list of pairs to compare
    if len(resolved_uris) == 1:
        # Only one disease given — compare it against all others
        d1 = resolved_uris[0]
        compare_pairs = [(d1, d) for d in ALL_DISEASE_URIS if d != d1]
    else:
        # 2 or 3 diseases — generate every unique pair
        # combinations preserves order and never repeats (A,B) as (B,A)
        compare_pairs = list(combinations(resolved_uris, 2))

    parts = []
    for d1_uri, d2_uri in compare_pairs:
        n1 = DISEASE_SHORT.get(str(d1_uri), get_label(d1_uri))
        n2 = DISEASE_SHORT.get(str(d2_uri), get_label(d2_uri))

        primary1 = set(g.objects(d1_uri, TRIAGE["hasPrimarySymptom"]))
        primary2 = set(g.objects(d2_uri, TRIAGE["hasPrimarySymptom"]))
        unique1  = primary1 - primary2
        unique2  = primary2 - primary1
        shared   = primary1 & primary2

        section = f"-- {n1} vs {n2} --"
        if unique1:
            section += f"\n{n1} distinctive: {', '.join(sorted(get_label(s) for s in unique1))}"
        if unique2:
            section += f"\n{n2} distinctive: {', '.join(sorted(get_label(s) for s in unique2))}"
        if shared:
            section += f"\nShared primary symptoms: {', '.join(sorted(get_label(s) for s in shared))}"
        if not unique1 and not unique2:
            section += f"\nNo primary symptom differences found between {n1} and {n2}."
        parts.append(section)

    return "\n\n".join(parts)


def handle_get_risk_factors(params):
    disease_uri = resolve_disease(params.get("disease", ""))
    cat_val     = _unwrap_param(params.get("factorCategory", "")).lower()

    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."

    factors = get_factors_of_disease(disease_uri, "isRiskFactorFor")
    dname   = DISEASE_NAMES.get(str(disease_uri), "")

    if not factors:
        return f"No risk factors found for {dname} in the ontology."

    if cat_val:
        cat_class_map = {
            "genetic":         TRIAGE["GeneticFactor"],
            "lifestyle":       TRIAGE["LifestyleFactor"],
            "epidemiological": TRIAGE["EpidemiologicalFactor"],
        }
        cat_class = next((v for k, v in cat_class_map.items() if k in cat_val), None)
        if cat_class:
            factors = [f for f in factors if (f, TRIAGE["belongsToFactorCategory"], cat_class) in g]
            dname  += f" ({cat_val})"

    labels = sorted(get_label(f) for f in factors)
    return f"Risk factors for {dname}:\n" + "\n".join(f"• {l}" for l in labels)


def handle_get_protective_factors(params):
    disease_uri = resolve_disease(params.get("disease", ""))
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."
    factors = get_factors_of_disease(disease_uri, "isProtectiveFactorFor")
    dname   = DISEASE_NAMES.get(str(disease_uri), "")
    if not factors:
        return f"No protective factors documented for {dname} in the ontology."
    labels = sorted(get_label(f) for f in factors)
    return f"Protective / beneficial factors for {dname}:\n" + "\n".join(f"• {l}" for l in labels)


def handle_get_genetic_factors(params):
    disease_uri = resolve_disease(params.get("disease", ""))
    factor_val  = _unwrap_param(params.get("influencingFactor", ""))
    gene_cat    = TRIAGE["GeneticFactor"]

    if factor_val:
        f_uri = resolve_symptom(factor_val) or label_to_uri.get(factor_val.lower())
        if f_uri:
            label       = get_label(f_uri)
            defn        = get_definition(f_uri)
            risk_for    = list(g.objects(f_uri, TRIAGE["isRiskFactorFor"]))
            protect_for = list(g.objects(f_uri, TRIAGE["isProtectiveFactorFor"]))
            resp = label
            if defn:
                resp += f"\n\n{defn}"
            if risk_for:
                resp += "\n\nRisk factor for: " + ", ".join(
                    DISEASE_SHORT.get(str(d), get_label(d)) for d in risk_for)
            if protect_for:
                resp += "\nProtective for: " + ", ".join(
                    DISEASE_SHORT.get(str(d), get_label(d)) for d in protect_for)
            return resp

    all_factors = get_factors_of_disease(disease_uri, "isRiskFactorFor") if disease_uri else []
    genetic     = [f for f in all_factors if (f, TRIAGE["belongsToFactorCategory"], gene_cat) in g]

    if not genetic and disease_uri:
        return f"No genetic risk factors found for {DISEASE_NAMES.get(str(disease_uri), '')}."

    if not genetic:
        genetic = list(g.subjects(TRIAGE["belongsToFactorCategory"], gene_cat))

    dname  = DISEASE_NAMES.get(str(disease_uri), "all three diseases") if disease_uri else "all three diseases"
    labels = sorted(get_label(f) for f in genetic)
    return f"Genetic risk factors for {dname}:\n" + "\n".join(f"• {l}" for l in labels)


def handle_get_lifestyle_factors(params):
    disease_uri   = resolve_disease(params.get("disease", ""))
    lifestyle_cat = TRIAGE["LifestyleFactor"]

    if disease_uri:
        all_risk  = get_factors_of_disease(disease_uri, "isRiskFactorFor")
        lifestyle = [f for f in all_risk if (f, TRIAGE["belongsToFactorCategory"], lifestyle_cat) in g]
        protective= get_factors_of_disease(disease_uri, "isProtectiveFactorFor")
        l_protect = [f for f in protective if (f, TRIAGE["belongsToFactorCategory"], lifestyle_cat) in g]
        dname     = DISEASE_NAMES.get(str(disease_uri), "")
        parts     = []
        if lifestyle:
            parts.append("Lifestyle risk factors:\n" + "\n".join(
                f"• {get_label(f)}" for f in sorted(lifestyle, key=get_label)))
        if l_protect:
            parts.append("Lifestyle protective factors:\n" + "\n".join(
                f"• {get_label(f)}" for f in sorted(l_protect, key=get_label)))
        if not parts:
            return f"No lifestyle factors found for {dname}."
        return f"Lifestyle factors for {dname}:\n\n" + "\n\n".join(parts)

    all_lifestyle = list(g.subjects(TRIAGE["belongsToFactorCategory"], lifestyle_cat))
    labels        = sorted(get_label(f) for f in all_lifestyle)
    return "Lifestyle influencing factors in the ontology:\n" + "\n".join(f"• {l}" for l in labels)


def handle_factor_detail(params):
    factor_val = _unwrap_param(params.get("influencingFactor", ""))
    if not factor_val:
        return "Which factor would you like details on?"

    f_uri = label_to_uri.get(factor_val.lower())
    if not f_uri:
        candidate = TRIAGE[factor_val.replace(" ", "_")]
        f_uri = candidate if (candidate, RDF.type, None) in g else None

    if not f_uri:
        return f"I couldn't find '{factor_val}' in the ontology."

    label         = get_label(f_uri)
    defn          = get_definition(f_uri)
    risk_for      = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["isRiskFactorFor"])]
    protect_for   = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["isProtectiveFactorFor"])]
    contradictory = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["hasContradictoryEvidenceFor"])]

    resp = label
    if defn:
        resp += f"\n\n{defn}"
    if risk_for:
        resp += f"\n\nRisk factor for: {', '.join(risk_for)}"
    if protect_for:
        resp += f"\nProtective for: {', '.join(protect_for)}"
    if contradictory:
        resp += f"\nContradictory evidence for: {', '.join(contradictory)} (studies conflict)"
    return resp


def handle_triage_result(params, session_params):
    reported = session_params.get("reported_symptoms", [])
    if not reported:
        return (
            "No symptoms have been reported yet. "
            "Please tell me about the patient's symptoms first, "
            "or say 'start triage' to begin the guided assessment."
        )

    sym_uris = [uri for sym_val in reported for uri in [resolve_symptom(sym_val)] if uri]

    if not sym_uris:
        return "I couldn't match the reported symptoms to the ontology. Could you rephrase them?"

    scores, matched = score_symptoms(sym_uris)
    ranked          = sorted(ALL_DISEASE_URIS, key=lambda d: scores.get(str(d), 0), reverse=True)
    rank_labels     = ["Most consistent with", "Second consideration", "Less consistent with"]

    lines = ["Triage Assessment\n" + "-" * 30]
    for i, d_uri in enumerate(ranked):
        score        = scores.get(str(d_uri), 0)
        dname        = DISEASE_NAMES[str(d_uri)]
        match_details= matched.get(str(d_uri), [])
        detail       = ", ".join(f"{sym} ({typ})" for sym, typ in match_details[:4])
        lines.append(f"{rank_labels[i]}: {dname} (score: {score:.1f})")
        if detail:
            lines.append(f"  Matched: {detail}")

    lines.append(
        "\n(!) This is an ontology-based triage aid only. "
        "Clinical judgement and diagnostic testing are required for diagnosis."
    )
    return "\n".join(lines)


# ─── Intent -> handler registry ───────────────────────────────────────────────
INTENT_HANDLERS = {
    "StartTriage":             handle_start_triage,
    "ReportSymptoms":          handle_report_symptoms,
    "GetPrimarySymptoms":      handle_get_primary_symptoms,
    "GetAllSymptoms":          handle_get_all_symptoms,
    "GetSymptomsByCategory":   handle_get_symptoms_by_category,
    "GetDiseaseFromSymptom":   handle_get_disease_from_symptom,
    "GetOverlappingSymptoms":  handle_get_overlapping,
    "DifferentiateByDisease":  handle_differentiate,
    "GetRiskFactors":          handle_get_risk_factors,
    "GetProtectiveFactors":    handle_get_protective_factors,
    "GetGeneticRiskFactors":   handle_get_genetic_factors,
    "GetLifestyleRiskFactors": handle_get_lifestyle_factors,
    "GetFactorDetail":         handle_factor_detail,
    "GetTriageResult":         handle_triage_result,
    "SymptomDetail.followup":  lambda p, sp: "I'll expand on that symptom in the triage context.",
}


# ─── Webhook endpoint ──────────────────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True, force=True) or {}

    query_result = req.get("queryResult", {})
    intent_name  = query_result.get("intent", {}).get("displayName", "")
    params       = query_result.get("parameters", {})

    # Flatten all output context parameters into one session dict
    session_p: dict = {}
    for ctx in query_result.get("outputContexts", []):
        session_p.update(ctx.get("parameters", {}))

    # FIX-6: accumulate reported symptoms as a flat list across turns
    if intent_name == "ReportSymptoms" and params.get("symptom"):
        existing = session_p.get("reported_symptoms", [])
        if isinstance(existing, str):
            existing = [existing]
        new_syms = params["symptom"]
        if isinstance(new_syms, str):
            new_syms = [new_syms]
        existing.extend(s for s in new_syms if s)
        session_p["reported_symptoms"] = existing

    # Dispatch
    handler = INTENT_HANDLERS.get(intent_name)
    if handler:
        try:
            sig = inspect.signature(handler)
            response_text = (
                handler(params, session_p)
                if len(sig.parameters) == 2
                else handler(params)
            )
        except Exception as e:
            response_text = f"An error occurred while processing your request: {e}"
    else:
        response_text = (
            f"Intent '{intent_name}' is not yet handled. "
            "Try asking about symptoms, risk factors, or say 'start triage'."
        )

    # Persist session state
    session_id      = req.get("session", "")
    output_contexts = []
    if session_p.get("reported_symptoms"):
        output_contexts.append({
            "name":          f"{session_id}/contexts/triage-active",
            "lifespanCount": 20,
            "parameters":    {"reported_symptoms": session_p["reported_symptoms"]},
        })

    return jsonify({
        "fulfillmentText":     response_text,
        "fulfillmentMessages": [{"text": {"text": [response_text]}}],
        "outputContexts":      output_contexts,
        "source":              "neurological-triage-webhook",
    })


# ─── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "ontology_triples": len(g),
        "individuals":      sum(1 for _ in g.subjects(RDF.type, OWL.NamedIndividual)),
    })


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loaded ontology: {len(g)} triples")
    print("Webhook running on http://localhost:5000/webhook")
    app.run(debug=True, host="0.0.0.0", port=5000)
