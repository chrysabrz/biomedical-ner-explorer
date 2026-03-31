"""
Canonical biomedical entity-type labels for the NER tools spreadsheet.
Used when loading data and when normalizing the CSV on disk.

Merges obvious synonyms (anatomy variants, adverse drug, disease phrasing, species/organism,
variants, lab/procedure labels, etc.). Gene vs protein and chemical vs drug are **not** merged.
Stanza BioNLP13CG anatomical fine types still map to Anatomy via BIONLP13CG_ANATOMY_MAP.
"""

from __future__ import annotations

# Any tool: merge obvious synonyms / UMLS-style variants
GLOBAL_ENTITY_MAP: dict[str, str] = {
    # Anatomy (same concept, different wording)
    "Anatomy (16 subtypes)": "Anatomy",
    "Anatomy / Body site": "Anatomy",
    "Anatomy / Body structure": "Anatomy",
    "Anatomy (bpoc, tisu, cell)": "Anatomy",
    "Anatomical system": "Anatomy",
    "Anatomy modifier": "Anatomy",
    # Adverse drug (MetaMap / MedCAT / CLAMP)
    "Adverse drug effect (inpo)": "Adverse drug effect",
    "Adverse drug reaction": "Adverse drug effect",
    # Disease (clinical vs biomedical phrasing; keep Disease / Cancer as its own label)
    "Disease / Disorder": "Disease",
    "Problem / Disease": "Disease",
    # Species / taxon (single-word export label → explicit)
    "Species": "Species / Organism",
    # Variants
    "Mutation / Variant": "Variant / Mutation",
    "Variant": "Variant / Mutation",
    # MetaMap semantic types in parentheses → short labels
    "Disease / Disorder (dsyn)": "Disease",
    "Sign / Symptom (sosy, T184)": "Sign / Symptom",
    "Procedure (topp, diap, lbpr)": "Procedure",
    "Lab test / Finding (lbpr)": "Lab test / Finding",
    "Species / Organism (orgm)": "Species / Organism",
    # Labs / tests
    "Lab test + value": "Lab test / Finding",
    "Test / Lab": "Lab test / Finding",
    # Procedures
    "Treatment / Procedure": "Procedure",
    # Clinical findings
    "Observation / Finding": "Sign / Symptom",
    # Oncology / pathology text-mining (same disease entity family)
    "Cancer": "Disease",
    "Pathological formation": "Disease",
}

# Stanza BioNLP13CG fine-grained anatomical categories → one bucket
BIONLP13CG_ANATOMY_MAP: dict[str, str] = {
    "Organ": "Anatomy",
    "Tissue": "Anatomy",
    "Organism subdivision": "Anatomy",
    "Multi-tissue structure": "Anatomy",
    "Developing anatomical structure": "Anatomy",
    "Immaterial anatomical entity": "Anatomy",
    "Organism substance": "Anatomy",
    "Cell": "Anatomy",
    "Anatomical system": "Anatomy",
}


def _is_stanza_bionlp13cg(tool: str | float | None) -> bool:
    if tool is None or (isinstance(tool, float) and str(tool) == "nan"):
        return False
    t = str(tool)
    return "bionlp13cg" in t.lower()


def canonicalize_entity_type(
    tool: str | float | None, entity_type: str | float | None
) -> str | float | None:
    """Return canonical entity label; pass through NaN / empty."""
    if entity_type is None or (isinstance(entity_type, float) and str(entity_type) == "nan"):
        return entity_type
    et = str(entity_type).strip()
    if not et:
        return entity_type

    if _is_stanza_bionlp13cg(tool) and et in BIONLP13CG_ANATOMY_MAP:
        return BIONLP13CG_ANATOMY_MAP[et]

    return GLOBAL_ENTITY_MAP.get(et, et)


def normalize_csv_entity_column(
    path: str, output_path: str | None = None, encoding: str = "utf-8"
) -> None:
    """Rewrite CSV with canonical Entity type values (in-place if output_path is None)."""
    import pandas as pd

    out = output_path or path
    df = pd.read_csv(path, encoding=encoding)
    df.columns = df.columns.str.strip()
    df["Tool"] = df["Tool"].ffill()
    if "Entity type" not in df.columns:
        raise ValueError("CSV must have an 'Entity type' column")

    def _row_et(r):
        raw = r["Entity type"]
        if pd.isna(raw):
            return raw
        return canonicalize_entity_type(r["Tool"], str(raw).strip())

    df["Entity type"] = df.apply(_row_et, axis=1)
    df.to_csv(out, index=False, encoding=encoding)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    default = here / "Tools for Recognition and Normalization - Sheet3 (1).csv"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    normalize_csv_entity_column(str(target))
    print(f"Updated: {target}")
