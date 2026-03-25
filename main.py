#!/usr/bin/env python3
"""
neurological_triage_webhook.py
─────────────────────────────
Dialogflow CX/ES webhook for the Neurological Triage Chatbot.

Queries neurological_triage.owl using rdflib to answer questions about:
- Symptoms of Alzheimer's, Parkinson's, and ALS
- Risk factors, protective factors, and genetic factors
- Differential triage based on reported symptoms
- Symptom-to-disease typicality mapping

Setup:
    pip install flask rdflib

Run locally:
    python neurological_triage_webhook.py

Expose with ngrok for Dialogflow:
    ngrok http 5000
    Then set webhook URL in Dialogflow Fulfillment to: https://<ngrok-id>.ngrok.io/webhook
"""

from flask import Flask, request, jsonify
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from rdflib.namespace import NamespaceManager
from collections import defaultdict
import os, re

app = Flask(__name__)

# ─── Ontology Setup ────────────────────────────────────────────────────────────
OWL_PATH = os.path.join(os.path.dirname(__file__), "neurological_triage.owl")
TRIAGE = Namespace("http://www.semanticweb.org/neurological-triage#")

g = Graph()
g.parse(OWL_PATH, format="xml")

# Pre-build label → URI map for fast lookup
label_to_uri = {}
for s, _, o in g.triples((None, RDFS.label, None)):
    label_to_uri[str(o).lower()] = s

# Canonical disease URIs
DISEASE_URIS = {
    "alzheimer": TRIAGE["alzheimers_disease"],
    "alzheimers": TRIAGE["alzheimers_disease"],
    "alzheimer's": TRIAGE["alzheimers_disease"],
    "alzheimer disease": TRIAGE["alzheimers_disease"],
    "ad": TRIAGE["alzheimers_disease"],
    "dementia": TRIAGE["alzheimers_disease"],
    "parkinson": TRIAGE["parkinson_disease"],
    "parkinson's": TRIAGE["parkinson_disease"],
    "parkinsons": TRIAGE["parkinson_disease"],
    "parkinson disease": TRIAGE["parkinson_disease"],
    "pd": TRIAGE["parkinson_disease"],
    "als": TRIAGE["als_disease"],
    "amyotrophic lateral sclerosis": TRIAGE["als_disease"],
    "motor neuron disease": TRIAGE["als_disease"],
    "mnd": TRIAGE["als_disease"],
    "lou gehrig's disease": TRIAGE["als_disease"],
}

DISEASE_NAMES = {
    str(TRIAGE["alzheimers_disease"]): "Alzheimer's Disease",
    str(TRIAGE["parkinson_disease"]): "Parkinson's Disease",
    str(TRIAGE["als_disease"]): "ALS (Amyotrophic Lateral Sclerosis)",
}

DISEASE_SHORT = {
    str(TRIAGE["alzheimers_disease"]): "Alzheimer's",
    str(TRIAGE["parkinson_disease"]): "Parkinson's",
    str(TRIAGE["als_disease"]): "ALS",
}

# ─── Helper Functions ──────────────────────────────────────────────────────────

def get_label(uri):
    """Return the rdfs:label of a URI, or the local name if not found."""
    for _, _, o in g.triples((uri, RDFS.label, None)):
        return str(o)
    local = str(uri).split("#")[-1].replace("_", " ")
    return local


def get_definition(uri):
    """Return the triage:definition of a URI if present."""
    for _, _, o in g.triples((uri, TRIAGE["definition"], None)):
        return str(o)
    return ""


def resolve_disease(raw_text):
    """Map a Dialogflow @Disease entity value to an ontology URI."""
    if not raw_text:
        return None
    clean = raw_text.strip().lower()
    # Try direct match first
    if clean in DISEASE_URIS:
        return DISEASE_URIS[clean]
    # Try partial match — handles "Parkinson's Disease" → "parkinson"
    for key, uri in DISEASE_URIS.items():
        if key in clean or clean in key:
            return uri
    return None


def resolve_symptom(raw_text):
    """Map a @Symptom entity value to an ontology URI."""
    if not raw_text:
        return None
    # Try exact label match first
    clean = raw_text.strip().lower()
    if clean in label_to_uri:
        return label_to_uri[clean]
    # Try underscore variant (entity value form)
    uri = TRIAGE[raw_text.replace(" ", "_")]
    if (uri, RDF.type, None) in g:
        return uri
    return None


def get_symptoms_of_disease(disease_uri, prop):
    """Return list of symptom URIs for a disease via the given property."""
    return list(g.objects(disease_uri, TRIAGE[prop]))


def get_symptom_display_list(symptom_uris):
    """Format a list of symptom URIs as a readable bulleted string."""
    labels = sorted([get_label(s) for s in symptom_uris])
    return "\n".join(f"• {l}" for l in labels)


def get_factors_of_disease(disease_uri, prop):
    """Return list of factor URIs for a disease via the given property."""
    factors = []
    for factor_uri in g.subjects(TRIAGE[prop], disease_uri):
        factors.append(factor_uri)
    return factors


def score_symptoms(reported_symptom_uris):
    """
    Triage scoring: given a list of symptom URIs the patient presents with,
    score each disease based on:
      - hasPrimarySymptom match = 3 points
      - hasSymptom match = 1 point
      - hasOverlappingSymptom match = 0.5 points (reduces, doesn't help differentiate)
      - moreTypicalOf = 2 bonus points
    Returns dict: {disease_uri: score}
    """
    all_diseases = [
        TRIAGE["alzheimers_disease"],
        TRIAGE["parkinson_disease"],
        TRIAGE["als_disease"]
    ]
    scores = defaultdict(float)
    matched = defaultdict(list)

    for s_uri in reported_symptom_uris:
        # moreTypicalOf
        for disease in g.objects(s_uri, TRIAGE["moreTypicalOf"]):
            scores[str(disease)] += 2.0
            matched[str(disease)].append((get_label(s_uri), "highly typical"))

    for disease in all_diseases:
        primary = set(g.objects(disease, TRIAGE["hasPrimarySymptom"]))
        has = set(g.objects(disease, TRIAGE["hasSymptom"]))
        overlap = set(g.objects(disease, TRIAGE["hasOverlappingSymptom"]))

        for s_uri in reported_symptom_uris:
            if s_uri in primary:
                scores[str(disease)] += 3.0
                matched[str(disease)].append((get_label(s_uri), "primary symptom"))
            elif s_uri in has:
                scores[str(disease)] += 1.0
                matched[str(disease)].append((get_label(s_uri), "associated symptom"))
            elif s_uri in overlap:
                scores[str(disease)] += 0.3
                matched[str(disease)].append((get_label(s_uri), "overlapping symptom"))

    return scores, matched


# ─── Intent Handlers ──────────────────────────────────────────────────────────

def handle_start_triage(params):
    return ("Starting neurological triage. I'll guide you through key symptoms to "
            "differentiate between Alzheimer's disease, Parkinson's disease, and ALS.\n\n"
            "Let's begin. Does the patient have any tremor or shaking, particularly at rest "
            "when the limb is not being used?")


def handle_report_symptoms(params, session_params):
    sym_val = params.get("symptom", "")
    if not sym_val:
        return "Could you describe the symptom in more detail? For example: resting tremor, limb weakness, memory loss, or breathing problems."

    s_uri = resolve_symptom(sym_val)
    if not s_uri:
        return (f"I noted '{sym_val}' but couldn't find an exact match in the ontology. "
                "Could you rephrase? For example: resting tremor, limb weakness, or episodic memory loss.")

    label = get_label(s_uri)
    defn = get_definition(s_uri)

    # Check typicality
    typical_of = list(g.objects(s_uri, TRIAGE["moreTypicalOf"]))
    typical_str = ""
    if typical_of:
        names = [DISEASE_SHORT.get(str(d), get_label(d)) for d in typical_of]
        typical_str = f" This symptom is most typical of {', '.join(names)}."

    # Check which diseases it appears in
    appearances = []
    for disease_uri in [TRIAGE["alzheimers_disease"], TRIAGE["parkinson_disease"], TRIAGE["als_disease"]]:
        primary = set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"]))
        has = set(g.objects(disease_uri, TRIAGE["hasSymptom"]))
        overlap = set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
        dname = DISEASE_SHORT[str(disease_uri)]
        if s_uri in primary:
            appearances.append(f"{dname} (primary)")
        elif s_uri in has:
            appearances.append(f"{dname} (associated)")
        elif s_uri in overlap:
            appearances.append(f"{dname} (overlapping)")

    appear_str = ""
    if appearances:
        appear_str = f" Seen in: {', '.join(appearances)}."

    response = f"Noted: {label}.{typical_str}{appear_str}"
    if defn:
        response += f"\n\nDefinition: {defn}"
    response += "\n\nShall I continue triage? Tell me another symptom or say 'give me the assessment'."
    return response


def handle_get_primary_symptoms(params):
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val)
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."

    primary = get_symptoms_of_disease(disease_uri, "hasPrimarySymptom")
    dname = DISEASE_NAMES.get(str(disease_uri), disease_val)

    if not primary:
        return f"No primary symptoms found for {dname} in the ontology."

    symptom_list = get_symptom_display_list(primary)
    return f"Primary symptoms of {dname}:\n{symptom_list}"


def handle_get_all_symptoms(params):
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val)
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."

    primary = get_symptoms_of_disease(disease_uri, "hasPrimarySymptom")
    associated = get_symptoms_of_disease(disease_uri, "hasSymptom")
    overlapping = get_symptoms_of_disease(disease_uri, "hasOverlappingSymptom")
    dname = DISEASE_NAMES.get(str(disease_uri), disease_val)

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
    cat_val = params.get("symptomCategory", "").lower()
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val) if disease_val else None

    # Map category value to SymptomCategory class URI
    cat_map = {
        "motor": TRIAGE["MotorSymptom"],
        "cognitive": TRIAGE["CognitiveSymptom"],
        "behavioural": TRIAGE["BehaviouralPsychiatricSymptom"],
        "behavioural_psychiatric": TRIAGE["BehaviouralPsychiatricSymptom"],
        "psychiatric": TRIAGE["BehaviouralPsychiatricSymptom"],
        "autonomic": TRIAGE["AutonomicSymptom"],
        "speech": TRIAGE["SpeechSwallowingSymptom"],
        "swallowing": TRIAGE["SpeechSwallowingSymptom"],
        "speech_swallowing": TRIAGE["SpeechSwallowingSymptom"],
        "respiratory": TRIAGE["RespiratorySymptom"],
        "sleep": TRIAGE["SleepSymptom"],
        "sensory": TRIAGE["SensorySymptom"],
    }

    cat_uri = None
    for key, uri in cat_map.items():
        if key in cat_val:
            cat_uri = uri
            break

    if not cat_uri:
        return f"I don't recognise the category '{cat_val}'. Try: motor, cognitive, autonomic, speech/swallowing, respiratory, sleep, behavioural, or sensory."

    cat_label = get_label(cat_uri)

    # Find symptom individuals that belong to this category
    matching_symptoms = []
    for sym_uri in g.subjects(TRIAGE["belongsToSymptomCategory"], cat_uri):
        matching_symptoms.append(sym_uri)

    if not matching_symptoms:
        return f"No {cat_label} symptoms found in the ontology."

    if disease_uri:
        # Filter to those the disease actually has
        all_disease_syms = (
            set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"])) |
            set(g.objects(disease_uri, TRIAGE["hasSymptom"])) |
            set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
        )
        matching_symptoms = [s for s in matching_symptoms if s in all_disease_syms]
        dname = DISEASE_SHORT.get(str(disease_uri), disease_val)
        prefix = f"{cat_label} symptoms of {dname}"
    else:
        prefix = f"{cat_label} symptoms across all three diseases"

    if not matching_symptoms:
        return f"No {cat_label} symptoms found for {DISEASE_SHORT.get(str(disease_uri), '')}."

    return f"{prefix}:\n{get_symptom_display_list(matching_symptoms)}"


def handle_get_disease_from_symptom(params):
    sym_val = params.get("symptom", "")
    s_uri = resolve_symptom(sym_val)
    if not s_uri:
        return f"I couldn't find '{sym_val}' in the ontology. Try: resting tremor, limb weakness, or episodic memory loss."

    label = get_label(s_uri)
    defn = get_definition(s_uri)

    typical_of = list(g.objects(s_uri, TRIAGE["moreTypicalOf"]))
    appearances = []
    for disease_uri in [TRIAGE["alzheimers_disease"], TRIAGE["parkinson_disease"], TRIAGE["als_disease"]]:
        primary = set(g.objects(disease_uri, TRIAGE["hasPrimarySymptom"]))
        has = set(g.objects(disease_uri, TRIAGE["hasSymptom"]))
        overlap = set(g.objects(disease_uri, TRIAGE["hasOverlappingSymptom"]))
        dname = DISEASE_SHORT[str(disease_uri)]
        if s_uri in primary: appearances.append(f"• {dname}: primary symptom")
        elif s_uri in has: appearances.append(f"• {dname}: associated symptom")
        elif s_uri in overlap: appearances.append(f"• {dname}: overlapping symptom")

    response = f"{label}"
    if defn:
        response += f"\n{defn}"
    if typical_of:
        names = [DISEASE_SHORT.get(str(d), get_label(d)) for d in typical_of]
        response += f"\n\nMost typical of: {', '.join(names)}."
    if appearances:
        response += f"\n\nAppears in:\n" + "\n".join(appearances)
    if not appearances and not typical_of:
        response += "\n\nThis symptom was not found linked to any of the three diseases in the ontology."
    return response


def handle_get_overlapping(params):
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val) if disease_val else None

    result_lines = []
    all_diseases = [TRIAGE["alzheimers_disease"], TRIAGE["parkinson_disease"], TRIAGE["als_disease"]]

    if disease_uri:
        overlapping = get_symptoms_of_disease(disease_uri, "hasOverlappingSymptom")
        dname = DISEASE_NAMES.get(str(disease_uri), disease_val)
        if not overlapping:
            return f"No overlapping symptoms documented for {dname}."
        return (f"Overlapping symptoms for {dname} (also seen in other diseases):\n"
                + get_symptom_display_list(overlapping))

    # All three diseases
    for d in all_diseases:
        overlapping = get_symptoms_of_disease(d, "hasOverlappingSymptom")
        if overlapping:
            dname = DISEASE_SHORT[str(d)]
            labels = sorted([get_label(s) for s in overlapping])
            result_lines.append(f"{dname}: {', '.join(labels)}")

    return ("Overlapping symptoms (symptoms that appear in multiple diseases, "
            "making differential diagnosis harder):\n\n" + "\n".join(result_lines))


def handle_differentiate(params):
    disease_val = params.get("disease", "")
    disease2_val = params.get("disease2", "")
    d1 = resolve_disease(disease_val)
    d2 = resolve_disease(disease2_val)

    if not d1:
        return "Please specify at least one disease to compare."

    all_diseases = [TRIAGE["alzheimers_disease"], TRIAGE["parkinson_disease"], TRIAGE["als_disease"]]
    compare_diseases = [d for d in all_diseases if d != d1] if not d2 else [d2]

    parts = []
    for d2_uri in compare_diseases:
        n1 = DISEASE_SHORT.get(str(d1))
        n2 = DISEASE_SHORT.get(str(d2_uri))
        primary1 = set(g.objects(d1, TRIAGE["hasPrimarySymptom"]))
        primary2 = set(g.objects(d2_uri, TRIAGE["hasPrimarySymptom"]))
        unique1 = primary1 - primary2
        unique2 = primary2 - primary1
        shared = primary1 & primary2

        section = f"── {n1} vs {n2} ──"
        if unique1:
            section += f"\n{n1} distinctive: {', '.join(sorted([get_label(s) for s in unique1]))}"
        if unique2:
            section += f"\n{n2} distinctive: {', '.join(sorted([get_label(s) for s in unique2]))}"
        if shared:
            section += f"\nShared primary: {', '.join(sorted([get_label(s) for s in shared]))}"
        parts.append(section)

    return "\n\n".join(parts)


def handle_get_risk_factors(params):
    disease_val = params.get("disease", "")
    cat_val = params.get("factorCategory", "").lower()
    disease_uri = resolve_disease(disease_val)
    if not disease_uri:
        return "Please specify a disease: Alzheimer's, Parkinson's, or ALS."

    factors = get_factors_of_disease(disease_uri, "isRiskFactorFor")
    dname = DISEASE_NAMES.get(str(disease_uri), disease_val)

    if not factors:
        return f"No risk factors found for {dname} in the ontology."

    # Optionally filter by category
    if cat_val:
        cat_class_map = {
            "genetic": TRIAGE["GeneticFactor"],
            "lifestyle": TRIAGE["LifestyleFactor"],
            "epidemiological": TRIAGE["EpidemiologicalFactor"],
        }
        cat_class = next((v for k,v in cat_class_map.items() if k in cat_val), None)
        if cat_class:
            filtered = []
            for f_uri in factors:
                if (f_uri, TRIAGE["belongsToFactorCategory"], cat_class) in g:
                    filtered.append(f_uri)
            factors = filtered
            dname += f" ({cat_val})"

    labels = sorted([get_label(f) for f in factors])
    return f"Risk factors for {dname}:\n" + "\n".join(f"• {l}" for l in labels)


def handle_get_protective_factors(params):
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val)
    if not disease_uri:
        return "Please specify a disease."

    factors = get_factors_of_disease(disease_uri, "isProtectiveFactorFor")
    dname = DISEASE_NAMES.get(str(disease_uri), disease_val)

    if not factors:
        return f"No protective factors documented for {dname} in the ontology."

    labels = sorted([get_label(f) for f in factors])
    return (f"Protective / beneficial factors for {dname}:\n"
            + "\n".join(f"• {l}" for l in labels))


def handle_get_genetic_factors(params):
    disease_val = params.get("disease", "")
    factor_val = params.get("influencingFactor", "")
    disease_uri = resolve_disease(disease_val) if disease_val else None

    if factor_val:
        # Specific gene query
        f_uri = resolve_symptom(factor_val)  # reuse URI resolver
        if not f_uri:
            f_uri = label_to_uri.get(factor_val.lower())
        if f_uri:
            label = get_label(f_uri)
            defn = get_definition(f_uri)
            risk_for = list(g.objects(f_uri, TRIAGE["isRiskFactorFor"]))
            protect_for = list(g.objects(f_uri, TRIAGE["isProtectiveFactorFor"]))
            response = f"{label}"
            if defn:
                response += f"\n\n{defn}"
            if risk_for:
                response += f"\n\nRisk factor for: {', '.join([DISEASE_SHORT.get(str(d), get_label(d)) for d in risk_for])}"
            if protect_for:
                response += f"\nProtective for: {', '.join([DISEASE_SHORT.get(str(d), get_label(d)) for d in protect_for])}"
            return response

    # All genetic factors for a disease
    gene_cat = TRIAGE["GeneticFactor"]
    all_factors = get_factors_of_disease(disease_uri, "isRiskFactorFor") if disease_uri else []
    genetic = [f for f in all_factors if (f, TRIAGE["belongsToFactorCategory"], gene_cat) in g]

    if not genetic and disease_uri:
        return f"No genetic risk factors found for {DISEASE_NAMES.get(str(disease_uri))}."

    if not genetic:
        # Return all genetic factors in ontology
        genetic = list(g.subjects(TRIAGE["belongsToFactorCategory"], gene_cat))

    dname = DISEASE_NAMES.get(str(disease_uri), "all diseases") if disease_uri else "all three diseases"
    labels = sorted([get_label(f) for f in genetic])
    return (f"Genetic risk factors for {dname}:\n"
            + "\n".join(f"• {l}" for l in labels))


def handle_get_lifestyle_factors(params):
    disease_val = params.get("disease", "")
    disease_uri = resolve_disease(disease_val) if disease_val else None
    lifestyle_cat = TRIAGE["LifestyleFactor"]

    if disease_uri:
        all_factors = get_factors_of_disease(disease_uri, "isRiskFactorFor")
        lifestyle = [f for f in all_factors if (f, TRIAGE["belongsToFactorCategory"], lifestyle_cat) in g]
        protective = get_factors_of_disease(disease_uri, "isProtectiveFactorFor")
        lifestyle_protect = [f for f in protective if (f, TRIAGE["belongsToFactorCategory"], lifestyle_cat) in g]
        dname = DISEASE_NAMES.get(str(disease_uri), disease_val)
        parts = []
        if lifestyle:
            parts.append("Lifestyle risk factors:\n" + "\n".join(f"• {get_label(f)}" for f in sorted(lifestyle, key=get_label)))
        if lifestyle_protect:
            parts.append("Lifestyle protective factors:\n" + "\n".join(f"• {get_label(f)}" for f in sorted(lifestyle_protect, key=get_label)))
        if not parts:
            return f"No lifestyle factors found for {dname}."
        return f"Lifestyle factors for {dname}:\n\n" + "\n\n".join(parts)

    all_lifestyle = list(g.subjects(TRIAGE["belongsToFactorCategory"], lifestyle_cat))
    labels = sorted([get_label(f) for f in all_lifestyle])
    return "Lifestyle influencing factors in the ontology:\n" + "\n".join(f"• {l}" for l in labels)


def handle_factor_detail(params):
    factor_val = params.get("influencingFactor", "")
    if not factor_val:
        return "Which factor would you like details on?"

    # Try to find by label
    f_uri = label_to_uri.get(factor_val.lower())
    if not f_uri:
        # Try underscore form
        f_uri = TRIAGE[factor_val.replace(" ", "_")]
        if (f_uri, RDF.type, None) not in g:
            f_uri = None

    if not f_uri:
        return f"I couldn't find '{factor_val}' in the ontology."

    label = get_label(f_uri)
    defn = get_definition(f_uri)
    risk_for = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["isRiskFactorFor"])]
    protect_for = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["isProtectiveFactorFor"])]
    contradictory = [DISEASE_SHORT.get(str(d), get_label(d)) for d in g.objects(f_uri, TRIAGE["hasContradictoryEvidenceFor"])]

    response = f"{label}"
    if defn:
        response += f"\n\n{defn}"
    if risk_for:
        response += f"\n\nRisk factor for: {', '.join(risk_for)}"
    if protect_for:
        response += f"\nProtective for: {', '.join(protect_for)}"
    if contradictory:
        response += f"\nContradictory evidence for: {', '.join(contradictory)} (studies conflict)"
    return response


def handle_triage_result(params, session_params):
    reported = session_params.get("reported_symptoms", [])
    if not reported:
        return ("No symptoms have been reported yet. Please tell me about the patient's symptoms first, "
                "or say 'start triage' to begin the guided assessment.")

    sym_uris = []
    for sym_val in reported:
        uri = resolve_symptom(sym_val)
        if uri:
            sym_uris.append(uri)

    if not sym_uris:
        return "I couldn't match the reported symptoms to the ontology. Could you rephrase them?"

    scores, matched = score_symptoms(sym_uris)
    all_diseases = [TRIAGE["alzheimers_disease"], TRIAGE["parkinson_disease"], TRIAGE["als_disease"]]
    ranked = sorted(all_diseases, key=lambda d: scores.get(str(d), 0), reverse=True)

    lines = ["Triage Assessment\n" + "─" * 30]
    for i, d_uri in enumerate(ranked):
        score = scores.get(str(d_uri), 0)
        dname = DISEASE_NAMES[str(d_uri)]
        match_details = matched.get(str(d_uri), [])
        rank_label = ["Most consistent with", "Second consideration", "Less consistent with"][i]
        detail = ", ".join([f"{sym} ({typ})" for sym, typ in match_details[:4]])
        lines.append(f"{rank_label}: {dname} (score: {score:.1f})")
        if detail:
            lines.append(f"  Matched: {detail}")

    lines.append("\n⚠ This is an ontology-based triage aid only. Clinical judgement and diagnostic testing are required for diagnosis.")
    return "\n".join(lines)


# ─── Main Webhook Router ───────────────────────────────────────────────────────

INTENT_HANDLERS = {
    "StartTriage": handle_start_triage,
    "ReportSymptoms": handle_report_symptoms,
    "GetPrimarySymptoms": handle_get_primary_symptoms,
    "GetAllSymptoms": handle_get_all_symptoms,
    "GetSymptomsByCategory": handle_get_symptoms_by_category,
    "GetDiseaseFromSymptom": handle_get_disease_from_symptom,
    "GetOverlappingSymptoms": handle_get_overlapping,
    "DifferentiateByDisease": handle_differentiate,
    "GetRiskFactors": handle_get_risk_factors,
    "GetProtectiveFactors": handle_get_protective_factors,
    "GetGeneticRiskFactors": handle_get_genetic_factors,
    "GetLifestyleRiskFactors": handle_get_lifestyle_factors,
    "GetFactorDetail": handle_factor_detail,
    "GetTriageResult": handle_triage_result,
    "SymptomDetail.followup": lambda p, sp: "I'll expand on that symptom in the triage context.",
}


@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True, force=True)

    intent_name = req.get("queryResult", {}).get("intent", {}).get("displayName", "")
    params = req.get("queryResult", {}).get("parameters", {})
    session_params = req.get("queryResult", {}).get("outputContexts", [{}])

    # Accumulate reported symptoms in session context
    session_p = {}
    for ctx in req.get("queryResult", {}).get("outputContexts", []):
        session_p.update(ctx.get("parameters", {}))

    # Track reported symptoms across turns
    if intent_name == "ReportSymptoms" and params.get("symptom"):
        existing = session_p.get("reported_symptoms", [])
        if isinstance(existing, str):
            existing = [existing]
        existing.append(params["symptom"])
        session_p["reported_symptoms"] = existing

    handler = INTENT_HANDLERS.get(intent_name)
    if handler:
        try:
            import inspect
            sig = inspect.signature(handler)
            if len(sig.parameters) == 2:
                response_text = handler(params, session_p)
            else:
                response_text = handler(params)
        except Exception as e:
            response_text = f"An error occurred while processing your request: {str(e)}"
    else:
        response_text = (f"Intent '{intent_name}' is not yet handled. "
                        "Try asking about symptoms, risk factors, or say 'start triage'.")

    # Build output contexts to persist session data
    session_id = req.get("session", "")
    output_contexts = []
    if session_p.get("reported_symptoms"):
        output_contexts.append({
            "name": f"{session_id}/contexts/triage-active",
            "lifespanCount": 20,
            "parameters": {"reported_symptoms": session_p["reported_symptoms"]}
        })

    return jsonify({
        "fulfillmentText": response_text,
        "fulfillmentMessages": [{"text": {"text": [response_text]}}],
        "outputContexts": output_contexts,
        "source": "neurological-triage-webhook"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ontology_triples": len(g), "individuals": sum(1 for _ in g.subjects(RDF.type, OWL.NamedIndividual))})


if __name__ == "__main__":
    print(f"Loaded ontology: {len(g)} triples")
    print(f"Webhook running on http://localhost:5000/webhook")
    app.run(debug=True, host="0.0.0.0", port=5000)
