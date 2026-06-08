"""
Regenerate benchmark_summary.csv (wide) and benchmark_summary_by_entity.csv (long)
from a single in-script source of truth.

The original 5 corpora (BC5CDR, BC2GM, NCBI Disease, BioRED, JNLPBA) are preserved
verbatim; the remaining corpora were researched per the agent reports (sources cited
in the chat). Split sizes are document/sentence counts at the stated granularity;
a blank split means no official split of that kind exists.
"""
import pandas as pd

WIDE_COLS = [
    "corpus", "granularity", "entity_types", "num_entity_types",
    "has_normalization", "normalization_dbs", "has_relations", "relation_types",
    "train_rows", "validation_rows", "test_rows", "hf_dataset",
]
LONG_COLS = [
    "corpus", "granularity", "entity_type", "is_normalized", "normalization_dbs",
    "corpus_has_relations", "corpus_relation_types",
    "train_rows", "validation_rows", "test_rows", "hf_dataset", "notes",
]


def join(parts):
    return "; ".join(parts)


# Each corpus: wide metadata + per-entity (entity_type, is_normalized, norm_dbs, notes).
# Repeated long-row fields (granularity, relations, splits, hf) are filled from `wide`.
CORPORA = []


def corpus(wide, ents):
    CORPORA.append({"wide": wide, "ents": ents})


# ---- Original 5 (verbatim) -------------------------------------------------
corpus(
    dict(corpus="BC5CDR", granularity="document",
         entity_types=["Chemical", "Disease"], has_normalization="yes",
         normalization_dbs=["MeSH"], has_relations="yes",
         relation_types=["CID (chemical induces disease)"],
         train="500", val="500", test="500", hf="bigbio/bc5cdr (bc5cdr_bigbio_kb)"),
    [("Chemical", "yes", ["MeSH"], ""), ("Disease", "yes", ["MeSH"], "")],
)
corpus(
    dict(corpus="BC2GM", granularity="sentence",
         entity_types=["GENE"], has_normalization="no", normalization_dbs=[],
         has_relations="no", relation_types=[],
         train="12500", val="2500", test="5000", hf="spyysalo/bc2gm_corpus"),
    [("GENE", "no", [], "IOB-tagged sentences; no document grouping.")],
)
corpus(
    dict(corpus="NCBI Disease", granularity="document",
         entity_types=["SpecificDisease", "DiseaseClass", "CompositeMention", "Modifier"],
         has_normalization="yes", normalization_dbs=["MeSH", "OMIM"],
         has_relations="no", relation_types=[],
         train="592", val="100", test="100", hf="bigbio/ncbi_disease (ncbi_disease_bigbio_kb)"),
    [("SpecificDisease", "yes", ["MeSH", "OMIM"], "Single specific disease (e.g. 'adenomatous polyposis coli')."),
     ("DiseaseClass", "yes", ["MeSH", "OMIM"], "Disease class or general term (e.g. 'cancer'; 'tumours')."),
     ("CompositeMention", "yes", ["MeSH", "OMIM"], "One surface mention covering 2+ diseases; multiple normalized IDs."),
     ("Modifier", "yes", ["MeSH", "OMIM"], "Disease term used adjectivally on another concept.")],
)
_biored_rel = ["Association", "Positive_Correlation", "Negative_Correlation", "Bind",
               "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
corpus(
    dict(corpus="BioRED", granularity="document",
         entity_types=["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
                       "OrganismTaxon", "SequenceVariant", "CellLine"],
         has_normalization="yes",
         normalization_dbs=["NCBIGene", "MeSH", "OMIM", "NCBITaxon", "dbSNP", "Cellosaurus"],
         has_relations="yes", relation_types=_biored_rel,
         train="400", val="100", test="100", hf="bigbio/biored (biored_bigbio_kb)"),
    [("GeneOrGeneProduct", "yes", ["NCBIGene"], ""),
     ("DiseaseOrPhenotypicFeature", "yes", ["MeSH", "OMIM"], ""),
     ("ChemicalEntity", "yes", ["MeSH"], ""),
     ("OrganismTaxon", "yes", ["NCBITaxon"], ""),
     ("SequenceVariant", "yes", ["dbSNP", "tmVar"], "Variants normalized to dbSNP rs IDs or tmVar."),
     ("CellLine", "yes", ["Cellosaurus"], "")],
)
_jnlpba_note = "Token-as-entity rows derived from IOB; merge consecutive B-/I- to recover spans."
corpus(
    dict(corpus="JNLPBA", granularity="sentence (token-as-entity)",
         entity_types=["DNA", "RNA", "protein", "cell_line", "cell_type"],
         has_normalization="no", normalization_dbs=[], has_relations="no", relation_types=[],
         train="18546", val="3856", test="", hf="bigbio/jnlpba (jnlpba_bigbio_kb)"),
    [(e, "no", [], _jnlpba_note) for e in ["DNA", "RNA", "protein", "cell_line", "cell_type"]],
)

# ---- Researched additions --------------------------------------------------
corpus(
    dict(corpus="AnatEM", granularity="document",
         entity_types=["Organism subdivision", "Anatomical system", "Organ", "Multi-tissue structure",
                       "Tissue", "Cell", "Developing anatomical structure", "Cellular component",
                       "Organism substance", "Immaterial anatomical entity", "Pathological formation", "Cancer"],
         has_normalization="no", normalization_dbs=[], has_relations="no", relation_types=[],
         train="606", val="202", test="404", hf="bigbio/anat_em (anat_em_bigbio_kb)"),
    [("Organism subdivision", "no", [], "Anatomical types defined w.r.t. the Common Anatomy Reference Ontology (CARO)."),
     ("Anatomical system", "no", [], ""), ("Organ", "no", [], ""),
     ("Multi-tissue structure", "no", [], ""), ("Tissue", "no", [], ""), ("Cell", "no", [], ""),
     ("Developing anatomical structure", "no", [], ""), ("Cellular component", "no", [], ""),
     ("Organism substance", "no", [], ""), ("Immaterial anatomical entity", "no", [], ""),
     ("Pathological formation", "no", [], ""),
     ("Cancer", "no", [], "Cancer-focused extension type beyond the base AnEM types.")],
)
corpus(
    dict(corpus="BC4CHEMD", granularity="document",
         entity_types=["SYSTEMATIC", "TRIVIAL", "FORMULA", "ABBREVIATION", "FAMILY", "MULTIPLE", "IDENTIFIER"],
         has_normalization="no", normalization_dbs=[], has_relations="no", relation_types=[],
         train="3500", val="3500", test="3000", hf="bigbio/chemdner (chemdner_bigbio_kb)"),
    [("SYSTEMATIC", "no", [], "CHEMDNER mention class (IUPAC-style systematic name). Many setups collapse all 7 classes to one CHEMICAL label."),
     ("TRIVIAL", "no", [], "Trivial/common name."), ("FORMULA", "no", [], "Molecular formula."),
     ("ABBREVIATION", "no", [], "Abbreviation/acronym."), ("FAMILY", "no", [], "Chemical family/class."),
     ("MULTIPLE", "no", [], "Non-contiguous/multiple-name mention."),
     ("IDENTIFIER", "no", [], "Database/registry identifier string (mention-type label, not a DB link).")],
)
_genia = ["protein_molecule", "protein_family_or_group", "protein_complex", "protein_subunit",
          "protein_substructure", "protein_domain_or_region", "protein_NA",
          "DNA_molecule", "DNA_family_or_group", "DNA_domain_or_region", "DNA_substructure", "DNA_NA",
          "RNA_molecule", "RNA_family_or_group", "RNA_domain_or_region", "RNA_substructure", "RNA_NA",
          "cell_type", "cell_line", "cell_component", "tissue", "body_part",
          "organism", "mono_cell", "virus", "other_organism",
          "lipid", "carbohydrate", "nucleotide", "amino_acid_monomer", "peptide",
          "inorganic", "atom", "other_name"]
corpus(
    dict(corpus="GENIA", granularity="document", entity_types=_genia,
         has_normalization="no", normalization_dbs=[], has_relations="no", relation_types=[],
         train="1999", val="", test="", hf="bigbio/genia_term_corpus (genia_term_corpus_bigbio_kb)"),
    [(e, "no", [], "GENIA ontology fine-grained class (nested annotations); distributed as a single train split.")
     for e in _genia],
)
_cg_ents = ["Anatomical_system", "Cancer", "Cell", "Cellular_component", "Developing_anatomical_structure",
            "Gene_or_gene_product", "Immaterial_anatomical_entity", "Multi-tissue_structure", "Organ",
            "Organism", "Organism_subdivision", "Organism_substance", "Pathological_formation",
            "Simple_chemical", "Tissue", "Amino_acid", "Anatomical_entity", "DNA_domain_or_region"]
_cg_rel = ["Development", "Growth", "Cell proliferation", "Cell death", "Carcinogenesis", "Metastasis",
           "Mutation", "Gene expression", "Transcription", "Translation", "Phosphorylation", "Binding",
           "Localization", "Regulation", "Positive regulation", "Negative regulation", "Planned process"]
corpus(
    dict(corpus="BioNLP13CG", granularity="document", entity_types=_cg_ents,
         has_normalization="no", normalization_dbs=[], has_relations="yes", relation_types=_cg_rel,
         train="300", val="100", test="200", hf="bigbio/bionlp_st_2013_cg (bionlp_st_2013_cg_bigbio_kb)"),
    [(_cg_ents[0], "no", [], "BioNLP Shared Task 2013 Cancer Genetics; event extraction (40 event types) + coreference. Entities are not linked to external IDs.")]
    + [(e, "no", [], "") for e in _cg_ents[1:]],
)
corpus(
    dict(corpus="CRAFT", granularity="document",
         entity_types=["CHEBI", "CL", "GO_BP", "GO_CC", "GO_MF", "MONDO", "MOP",
                       "NCBITaxon", "PR", "SO", "UBERON"],
         has_normalization="yes",
         normalization_dbs=["ChEBI", "Cell Ontology (CL)", "Gene Ontology (GO)", "MONDO",
                            "Molecular Process Ontology (MOP)", "NCBI Taxonomy",
                            "Protein Ontology (PR)", "Sequence Ontology (SO)", "UBERON"],
         has_relations="no", relation_types=[],
         train="60", val="7", test="30", hf="bigbio/craft (craft_bigbio_kb)"),
    [("CHEBI", "yes", ["ChEBI"], "Each class is an ontology; mentions carry that ontology's concept IDs. Splits are full-text article counts (dev=validation)."),
     ("CL", "yes", ["Cell Ontology (CL)"], ""),
     ("GO_BP", "yes", ["Gene Ontology - Biological Process"], ""),
     ("GO_CC", "yes", ["Gene Ontology - Cellular Component"], ""),
     ("GO_MF", "yes", ["Gene Ontology - Molecular Function"], ""),
     ("MONDO", "yes", ["MONDO Disease Ontology"], ""),
     ("MOP", "yes", ["Molecular Process Ontology (MOP)"], ""),
     ("NCBITaxon", "yes", ["NCBI Taxonomy"], ""),
     ("PR", "yes", ["Protein Ontology (PR)"], ""),
     ("SO", "yes", ["Sequence Ontology (SO)"], ""),
     ("UBERON", "yes", ["UBERON"], "")],
)
_cpr = ["CPR:1", "CPR:2", "CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:7", "CPR:8", "CPR:9", "CPR:10"]
corpus(
    dict(corpus="ChemProt", granularity="document",
         entity_types=["CHEMICAL", "GENE-Y", "GENE-N"], has_normalization="no", normalization_dbs=[],
         has_relations="yes", relation_types=_cpr,
         train="1020", val="612", test="800", hf="bigbio/chemprot (chemprot_bigbio_kb)"),
    [("CHEMICAL", "no", [], "Splits are PubMed abstract counts. Shared-task evaluation uses CPR groups 3,4,5,6,9."),
     ("GENE-Y", "no", [], "Gene/protein normalizable in the original annotation."),
     ("GENE-N", "no", [], "Gene/protein not normalizable in the original annotation.")],
)
corpus(
    dict(corpus="Linnaeus", granularity="document",
         entity_types=["Species"], has_normalization="yes", normalization_dbs=["NCBI Taxonomy"],
         has_relations="no", relation_types=[],
         train="100", val="", test="", hf="bigbio/linnaeus"),
    [("Species", "yes", ["NCBI Taxonomy"],
      "100 full-text PMC articles (2,988 species annotations); distributed as a single set with no official split.")],
)
corpus(
    dict(corpus="Species-800", granularity="sentence",
         entity_types=["Species"], has_normalization="yes", normalization_dbs=["NCBI Taxonomy"],
         has_relations="no", relation_types=[],
         train="560", val="80", test="160", hf="spyysalo/species_800"),
    [("Species", "yes", ["NCBI Taxonomy"],
      "Also known as S800. 800 PubMed abstracts; split counts are document (abstract) counts from the official split files.")],
)
corpus(
    dict(corpus="BioID", granularity="sentence (token-as-entity)",
         entity_types=["Gene", "Protein", "miRNA", "Cellular component", "Cell type", "Cell line",
                       "Tissue/Organ", "Organism/Species", "Small molecule/Chemical"],
         has_normalization="yes",
         normalization_dbs=["NCBI Gene", "Uniprot", "Rfam", "GO", "CL", "Cellosaurus",
                            "Uberon", "NCBI Taxonomy", "ChEBI", "PubChem", "Corum"],
         has_relations="no", relation_types=[],
         train="13573", val="", test="4310", hf="bigbio/bioid"),
    [("Gene", "yes", ["NCBI Gene"],
      "BioCreative VI Bio-ID track; units are figure-panel captions. No official validation split."),
     ("Protein", "yes", ["Uniprot", "Corum"], "Protein complexes normalized to Corum."),
     ("miRNA", "yes", ["Rfam"], ""),
     ("Cellular component", "yes", ["GO"], ""),
     ("Cell type", "yes", ["CL (Cell Ontology)"], ""),
     ("Cell line", "yes", ["Cellosaurus"], ""),
     ("Tissue/Organ", "yes", ["Uberon"], ""),
     ("Organism/Species", "yes", ["NCBI Taxonomy"], ""),
     ("Small molecule/Chemical", "yes", ["ChEBI", "PubChem"], "")],
)
corpus(
    dict(corpus="NLM-Chem", granularity="document",
         entity_types=["Chemical"], has_normalization="yes", normalization_dbs=["MeSH"],
         has_relations="no", relation_types=[],
         train="80", val="20", test="50", hf="bigbio/nlmchem"),
    [("Chemical", "yes", ["MeSH"],
      "150 full-text PMC articles, 80/20/50 split; chemical mentions normalized to MeSH.")],
)
corpus(
    dict(corpus="NLM-Gene", granularity="document",
         entity_types=["Gene"], has_normalization="yes", normalization_dbs=["NCBI Gene"],
         has_relations="no", relation_types=[],
         train="450", val="", test="100", hf="bigbio/nlm_gene"),
    [("Gene", "yes", ["NCBI Gene"],
      "550 PubMed articles (multi-species), 450/100 split; no official validation split.")],
)
corpus(
    dict(corpus="GNormPlus", granularity="document",
         entity_types=["Gene", "FamilyName", "DomainMotif"], has_normalization="yes",
         normalization_dbs=["NCBI Gene"], has_relations="no", relation_types=[],
         train="432", val="", test="262", hf="bigbio/gnormplus"),
    [("Gene", "yes", ["NCBI Gene"], "Gene mentions normalized to NCBI (Entrez) Gene. Train = BC2GNtrain (281) + NLMIAT (151); test = BC2GNtest (262)."),
     ("FamilyName", "no", [], "Gene family annotations; not normalized to NCBI Gene."),
     ("DomainMotif", "no", [], "Protein domain/motif annotations; not normalized.")],
)
corpus(
    dict(corpus="tmVar (v3)", granularity="document",
         entity_types=["DNAMutation", "ProteinMutation", "SNP", "DNAAllele", "ProteinAllele",
                       "AcidChange", "OtherMutation", "CopyNumberVariant", "Gene", "Species", "CellLine"],
         has_normalization="yes",
         normalization_dbs=["dbSNP", "tmVar", "ClinGen Allele Registry", "NCBI Gene", "NCBI Taxonomy"],
         has_relations="no", relation_types=[],
         train="", val="", test="500", hf="bigbio/tmvar_v3"),
    [("DNAMutation", "yes", ["dbSNP", "tmVar", "ClinGen Allele Registry"],
      "500 PubMed articles; no official train/dev split (loader exposes a single test split)."),
     ("ProteinMutation", "yes", ["dbSNP", "tmVar", "ClinGen Allele Registry"], ""),
     ("SNP", "yes", ["dbSNP"], "rs identifiers."),
     ("DNAAllele", "yes", ["tmVar", "ClinGen Allele Registry"], "Allele/CNV recognition new in tmVar 3.0."),
     ("ProteinAllele", "yes", ["tmVar", "ClinGen Allele Registry"], ""),
     ("AcidChange", "yes", ["tmVar"], ""),
     ("OtherMutation", "yes", ["tmVar"], ""),
     ("CopyNumberVariant", "yes", ["tmVar"], "New variant class in tmVar 3.0."),
     ("Gene", "yes", ["NCBI Gene"], ""),
     ("Species", "yes", ["NCBI Taxonomy"], ""),
     ("CellLine", "no", [], "")],
)
corpus(
    dict(corpus="SCAI Chemical", granularity="document",
         entity_types=["IUPAC", "TRIVIAL", "TRIVIALVAR", "PARTIUPAC", "FAMILY", "ABBREVIATION", "MODIFIER", "SUM"],
         has_normalization="no", normalization_dbs=[], has_relations="no", relation_types=[],
         train="100", val="", test="", hf="bigbio/scai_chemical"),
    [("IUPAC", "no", [], "100 MEDLINE abstracts; single set, no official split."),
     ("TRIVIAL", "no", [], ""), ("TRIVIALVAR", "no", [], ""),
     ("PARTIUPAC", "no", [], "Partial IUPAC name."), ("FAMILY", "no", [], ""),
     ("ABBREVIATION", "no", [], ""), ("MODIFIER", "no", [], ""),
     ("SUM", "no", [], "Sum-formula type (original SCAI tagset).")],
)
corpus(
    dict(corpus="SCAI Disease", granularity="document",
         entity_types=["DISEASE", "ADVERSE"], has_normalization="no", normalization_dbs=[],
         has_relations="no", relation_types=[],
         train="400", val="", test="", hf="bigbio/scai_disease"),
    [("DISEASE", "no", [], "400 MEDLINE abstracts; single set, no official split or concept normalization."),
     ("ADVERSE", "no", [], "Adverse-effect mentions.")],
)
_dimb_rel = ["affects", "improves", "worsens", "associated_with", "pos_associated_with",
             "neg_associated_with", "interacts_with", "increases", "decreases", "causes",
             "prevents", "predisposes", "has_component"]
corpus(
    dict(corpus="DiMB-RE", granularity="document",
         entity_types=["Food", "Nutrient", "DietPattern", "Microorganism", "DiversityMetric",
                       "Metabolite", "Physiology", "Disease", "Measurement", "Enzyme", "Gene",
                       "Chemical", "Methodology", "Population", "Biospecimen"],
         has_normalization="yes",
         normalization_dbs=["FoodOn", "MeSH", "NCI Thesaurus", "ChEBI", "NCBI Taxonomy", "NCBI Gene", "OCHV"],
         has_relations="yes", relation_types=_dimb_rel,
         train="109", val="19", test="37", hf=""),
    [("Food", "yes", ["FoodOn", "MeSH", "NCI Thesaurus", "OCHV"],
      "Diet-microbiome relation corpus; splits are document counts from the official release. Not on Hugging Face."),
     ("Nutrient", "yes", ["MeSH", "ChEBI", "NCI Thesaurus", "OCHV"], ""),
     ("DietPattern", "yes", ["MeSH", "NCI Thesaurus", "OCHV"], ""),
     ("Microorganism", "yes", ["NCBI Taxonomy"], ""),
     ("DiversityMetric", "no", [], ""),
     ("Metabolite", "yes", ["ChEBI", "MeSH"], ""),
     ("Physiology", "yes", ["MeSH", "NCI Thesaurus", "OCHV"], ""),
     ("Disease", "yes", ["MeSH", "NCI Thesaurus", "OCHV"], ""),
     ("Measurement", "no", [], ""),
     ("Enzyme", "yes", ["NCBI Gene"], ""),
     ("Gene", "yes", ["NCBI Gene"], ""),
     ("Chemical", "yes", ["ChEBI", "MeSH"], ""),
     ("Methodology", "no", [], "Part of the full 15-type schema; absent from the reduced NER split files."),
     ("Population", "no", [], ""),
     ("Biospecimen", "no", [], "")],
)


# ---- Emit ------------------------------------------------------------------
wide_rows, long_rows = [], []
for c in CORPORA:
    w = c["wide"]
    wide_rows.append({
        "corpus": w["corpus"], "granularity": w["granularity"],
        "entity_types": join(w["entity_types"]), "num_entity_types": len(w["entity_types"]),
        "has_normalization": w["has_normalization"], "normalization_dbs": join(w["normalization_dbs"]),
        "has_relations": w["has_relations"], "relation_types": join(w["relation_types"]),
        "train_rows": w["train"], "validation_rows": w["val"], "test_rows": w["test"],
        "hf_dataset": w["hf"],
    })
    rel_long = join(w["relation_types"]) if w["has_relations"] == "yes" else ""
    for ent, is_norm, norm_dbs, notes in c["ents"]:
        long_rows.append({
            "corpus": w["corpus"], "granularity": w["granularity"], "entity_type": ent,
            "is_normalized": is_norm, "normalization_dbs": join(norm_dbs),
            "corpus_has_relations": w["has_relations"], "corpus_relation_types": rel_long,
            "train_rows": w["train"], "validation_rows": w["val"], "test_rows": w["test"],
            "hf_dataset": w["hf"], "notes": notes,
        })

pd.DataFrame(wide_rows, columns=WIDE_COLS).to_csv("benchmark_summary.csv", index=False)
pd.DataFrame(long_rows, columns=LONG_COLS).to_csv("benchmark_summary_by_entity.csv", index=False)
print(f"wrote {len(wide_rows)} corpora, {len(long_rows)} entity rows")
