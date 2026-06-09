"""
Biomedical NER/NEN Explorer
Interactive visualization of entity coverage, training corpora, and ID outputs

Usage:
    pip install -r requirements.txt
    streamlit run biomedical_ner_explorer.py

Loads the bundled CSV next to this script by default; optional upload overrides.
"""

from __future__ import annotations

import io
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from entity_canonicalization import canonicalize_entity_type
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Purple-led palette (primary accent + complementary hues for chart series)
_NS_PURPLE = "#A855F7"
_NS_VIOLET = "#8B5CF6"
_NS_PURPLE_DEEP = "#6D28D9"
_NS_LAVENDER = "#C4B5FD"
_NS_FUCHSIA = "#D946EF"
_NS_BLUE = "#1A7BD4"
_NS_CYAN = "#06B6D4"
_NS_AMBER = "#F59E0B"
_NS_GREEN = "#10B981"
_NS_ROSE = "#F43F5E"
_NS_ORANGE = "#F97316"
_NS_SKY = "#0EA5E9"
_NS_RED = "#EF4444"
_NS_TEAL = "#14B8A6"
_NS_GRAY = "#6B7280"

NS_DISCRETE_PALETTE = [
    _NS_PURPLE, _NS_VIOLET, _NS_FUCHSIA, _NS_PURPLE_DEEP, _NS_LAVENDER,
    _NS_AMBER, _NS_ROSE, _NS_GREEN, _NS_CYAN, _NS_ORANGE, _NS_SKY, _NS_TEAL,
]

# Network node roles - same family as NeuroSight relationship colouring
NETWORK_NODE_COLORS = {
    "tool": _NS_PURPLE,
    "entity": _NS_VIOLET,
    "corpus": _NS_CYAN,
    "id_output": _NS_GREEN,
}

NS_HEATMAP_SCALE = [
    [0.0, "#faf5ff"],
    [0.35, "#ddd6fe"],
    [0.65, _NS_PURPLE],
    [1.0, _NS_PURPLE_DEEP],
]

NS_NORM_BAR_SCALE = [
    [0.0, _NS_RED],
    [0.5, _NS_AMBER],
    [1.0, _NS_GREEN],
]

# Light-lavender surfaces (purple-tinted, still neutral enough for dataframes)
THEME_PAGE = "#f5f3ff"
THEME_SIDEBAR = "#ede9fe"
THEME_TEXT = "#1e1b4b"
THEME_TEXT_MUTED = "#4c1d95"
THEME_BORDER = "#c4b5fd"
THEME_PLOT_PAPER = "#faf5ff"
THEME_PLOT_BG = "#fafafa"
THEME_PLOT_FONT = "#3b0764"
THEME_HOVER_BG = "#ffffff"
THEME_HOVER_FONT = "#1e1b4b"

DEFAULT_CSV = SCRIPT_DIR / "comprehensive_biomedical_ner_tools.csv"

# Plotly: wheel zoom + toolbar (NeuroSight-style interactivity in Streamlit)
PLOTLY_NETWORK_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _truncate_label(s: str, max_len: int = 26) -> str:
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _cluster_layout_ner(G: nx.Graph) -> dict:
    """Semantic clusters (like NeuroSight Knowledge Graph), not a hairball spring layout."""
    by_type: dict[str, list] = defaultdict(list)
    for n in G.nodes():
        by_type[G.nodes[n]["type"]].append(n)
    centers = {
        "tool": (-3.15, 0.0),
        "entity": (0.0, 0.0),
        "corpus": (3.05, 1.45),
        "id_output": (3.05, -1.45),
    }
    type_phase = {"tool": 0.0, "entity": 0.9, "corpus": 0.35, "id_output": 1.7}
    pos = {}
    for ntype, nodes in by_type.items():
        cx, cy = centers.get(ntype, (0.0, 0.0))
        phase = type_phase.get(ntype, 0.0)
        n = len(nodes)
        for i, node in enumerate(sorted(nodes, key=lambda x: G.nodes[x]["label"])):
            ang = (2 * math.pi * i / max(n, 1)) + phase
            r = 0.5 + 0.24 * (i % 7)
            pos[node] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return pos


def _edge_key(n1: str, n2: str) -> tuple[str, str]:
    return tuple(sorted((n1, n2)))


def _node_hover_html(G: nx.Graph, node: str) -> str:
    meta = G.nodes[node]
    label, ntype = meta["label"], meta["type"]
    nbrs = list(G.neighbors(node))
    by_cat: dict[str, list[str]] = defaultdict(list)
    for nb in nbrs:
        by_cat[G.nodes[nb]["type"]].append(G.nodes[nb]["label"])
    lines = [
        f"<b>{label}</b>",
        f"<span style='color:#64748b'>Category:</span> {ntype.replace('_', ' ').title()}",
        f"<span style='color:#64748b'>Degree:</span> {len(nbrs)} links",
    ]
    for cat, title in [
        ("tool", "Tools"),
        ("entity", "Entity types"),
        ("corpus", "Training corpora"),
        ("id_output", "ID outputs"),
    ]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        shown = items[:10]
        extra = len(items) - len(shown)
        tail = f"<br>  … +{extra} more" if extra > 0 else ""
        lines.append(f"<b>{title}</b><br>  " + "<br>  ".join(shown) + tail)
    return "<br>".join(lines)


def build_ner_graph(filtered_df: pd.DataFrame):
    """Returns graph, positions, and edge hover lines keyed by undirected edge key."""
    G = nx.Graph()
    edge_lines: dict[tuple[str, str], list[str]] = defaultdict(list)

    def reg(n1: str, n2: str, line: str):
        edge_lines[_edge_key(n1, n2)].append(line)

    for _, row in filtered_df.iterrows():
        tool = row["Tool"]
        entity = row["Entity type"]
        corpus_raw = row["NER training corpus"]
        id_output = row["ID output"]
        task = row.get("Task", "")
        norm_tool = row.get("Normalization tool", "")

        tk = f"tool:{tool}"
        ek = f"entity:{entity}"
        G.add_node(tk, type="tool", label=tool)
        G.add_node(ek, type="entity", label=entity)
        G.add_edge(tk, ek)
        nt = f"<br>Normalization: {norm_tool}" if pd.notna(norm_tool) and str(norm_tool).strip() else ""
        reg(tk, ek, f"<b>{tool}</b> ↔ <b>{entity}</b><br>Task: {task}{nt}")

        if pd.notna(corpus_raw):
            for corpus in [c.strip() for c in str(corpus_raw).split(",") if c.strip()]:
                ck = f"corpus:{corpus}"
                G.add_node(ck, type="corpus", label=corpus)
                G.add_edge(tk, ck)
                reg(tk, ck, f"<b>{tool}</b> trains on<br><i>{corpus}</i>")

        if pd.notna(id_output) and str(id_output).strip() not in ("", "Span only"):
            ik = f"id:{id_output}"
            G.add_node(ik, type="id_output", label=str(id_output))
            G.add_edge(tk, ik)
            reg(tk, ik, f"<b>{tool}</b> outputs<br><i>{id_output}</i>")

    pos = _cluster_layout_ner(G) if len(G) else {}
    return G, pos, edge_lines

# Page config
st.set_page_config(
    page_title="Biomedical NER/NEN Explorer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Biomedical NER/NEN Explorer🧬")
st.markdown("**Interactive exploration of entity coverage, training corpora, and ID outputs**")

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
        background-color: """ + THEME_PAGE + """ !important;
        color: """ + THEME_TEXT + """ !important;
    }
    /* Top header bar (Deploy / hamburger toolbar) stayed white - tint it to match the page */
    [data-testid="stHeader"], header[data-testid="stHeader"] {
        background: """ + THEME_PAGE + """ !important;
    }
    [data-testid="stHeader"] * {
        color: """ + THEME_TEXT + """ !important;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background: """ + THEME_SIDEBAR + """ !important;
        border-right: 1px solid """ + THEME_BORDER + """;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p {
        color: """ + THEME_TEXT + """ !important;
    }
    h1, h2, h3 { color: """ + THEME_TEXT + """ !important; }
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li { color: """ + THEME_TEXT_MUTED + """ !important; }
    /* Selectbox menus inside tabs were clipped (list looked like it ended at “Chemical”) */
    [data-testid="stTabs"] [role="tabpanel"] { overflow: visible !important; }
    [data-testid="stTabs"] [data-testid="stVerticalBlock"] { overflow: visible !important; }
    div[data-baseweb="popover"],
    ul[role="listbox"] {
        z-index: 100000 !important;
        max-height: min(70vh, 520px) !important;
        overflow-y: auto !important;
    }
    [data-testid="stTabs"] button { color: """ + THEME_TEXT + """ !important; }
    [data-testid="stMetricValue"] { color: """ + THEME_TEXT + """ !important; }
    [data-testid="stMetricLabel"] { color: """ + THEME_TEXT_MUTED + """ !important; }
    /* Multiselect chips: dark purple with white text instead of Streamlit's default red */
    span[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: """ + _NS_PURPLE_DEEP + """ !important;
        color: #ffffff !important;
    }
    span[data-baseweb="tag"] *,
    div[data-baseweb="tag"] * {
        color: #ffffff !important;
    }
    span[data-baseweb="tag"] svg,
    div[data-baseweb="tag"] svg {
        fill: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for file upload and filters
st.sidebar.header(" Data Input")
if DEFAULT_CSV.exists():
    st.sidebar.caption(f"Default: `{DEFAULT_CSV.name}`")
else:
    st.sidebar.warning(f"Default CSV not found: `{DEFAULT_CSV.name}` - upload a file below.")

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload a different CSV",
    type=["csv"],
    help="Overrides the bundled spreadsheet when provided.",
)


def _norm_text_cell(val):
    """Strip, NFKC unicode, collapse spaces - avoids duplicate Entity type / Tool labels in the UI."""
    if pd.isna(val):
        return val
    t = str(val).replace("\u00a0", " ").replace("\u200b", "")
    t = " ".join(t.split())
    t = unicodedata.normalize("NFKC", t).strip()
    return t


def _corpus_tokens(cell) -> list[str]:
    if pd.isna(cell):
        return []
    out: list[str] = []
    for part in str(cell).split(","):
        t = _norm_text_cell(part)
        if pd.notna(t) and str(t).strip():
            out.append(str(t).strip())
    return out


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Map normalized CSV column names to the names used throughout the app
    col_map = {
        "Entity_Type_Standardized": "Entity type",
        "NER_Training_Corpus": "NER training corpus",
        "Normalization_Tool": "Normalization tool",
        "ID_Output": "ID output",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # Drop columns we don't need
    df = df.drop(columns=[c for c in ("Institution", "Year") if c in df.columns])
    if "Unnamed: 6" in df.columns:
        df = df.rename(columns={"Unnamed: 6": "URL"})
    if "url" in df.columns and "URL" not in df.columns:
        df = df.rename(columns={"url": "URL"})
    df["Tool"] = df["Tool"].ffill()
    df["Tool"] = df["Tool"].map(_norm_text_cell)
    df = df.dropna(subset=["Entity type"])
    df["Entity type"] = df["Entity type"].map(_norm_text_cell)
    df = df[df["Entity type"].astype(str).str.strip() != ""]
    df["Entity type"] = df.apply(
        lambda r: canonicalize_entity_type(r["Tool"], r["Entity type"]),
        axis=1,
    )
    return df


_PREPARE_VERSION = 2  # bump to bust Streamlit cache after prepare_df changes


@st.cache_data
def load_data_from_bytes(data: bytes, _v: int = _PREPARE_VERSION) -> pd.DataFrame:
    return prepare_df(pd.read_csv(io.BytesIO(data)))


@st.cache_data
def load_data_from_path(path_str: str, _mtime: float, _v: int = _PREPARE_VERSION) -> pd.DataFrame:
    return prepare_df(pd.read_csv(path_str))


df = None
if uploaded_file is not None:
    df = load_data_from_bytes(uploaded_file.getvalue())
elif DEFAULT_CSV.exists():
    mtime = DEFAULT_CSV.stat().st_mtime
    df = load_data_from_path(str(DEFAULT_CSV), mtime)

if df is not None:

    def _all_corpus_names(frame: pd.DataFrame) -> list[str]:
        names: set[str] = set()
        if "NER training corpus" not in frame.columns:
            return []
        for raw in frame["NER training corpus"].dropna():
            names.update(_corpus_tokens(raw))
        return sorted(names, key=str.lower)

    # Sidebar filters
    st.sidebar.header("Filters")

    tool_names = sorted(df["Tool"].dropna().unique(), key=str.lower)
    selected_tools = st.sidebar.multiselect(
        "Tools",
        options=tool_names,
        default=tool_names,
        help="Restrict views to specific NER/NEN systems.",
    )

    entity_types = sorted(df["Entity type"].unique(), key=str.lower)
    selected_entities = st.sidebar.multiselect(
        "Entity types",
        options=entity_types,
        default=entity_types,
        help="After cleanup, each label appears once (whitespace / unicode normalized).",
    )

    tasks = sorted(df["Task"].dropna().unique(), key=str.lower)
    selected_tasks = st.sidebar.multiselect(
        "Task type",
        options=tasks,
        default=list(tasks),
    )

    corpus_options = _all_corpus_names(df)
    selected_corpora = st.sidebar.multiselect(
        "Training corpora (optional)",
        options=corpus_options,
        default=[],
        help="Keep only rows whose “NER training corpus” field mentions any selected corpus. Empty = no extra filter.",
    )

    has_normalization = st.sidebar.checkbox(
        "Only tools with ID normalization",
        help="Show only tools that output structured IDs (not just spans)",
    )

    base_mask = (
        df["Tool"].isin(selected_tools)
        & df["Entity type"].isin(selected_entities)
        & df["Task"].isin(selected_tasks)
    )
    if selected_corpora:
        want = set(selected_corpora)
        corpus_hit = df["NER training corpus"].apply(
            lambda x: bool(set(_corpus_tokens(x)) & want) if pd.notna(x) else False
        )
        base_mask = base_mask & corpus_hit

    filtered_df = df[base_mask].copy()

    if has_normalization:
        filtered_df = filtered_df[
            (filtered_df["ID output"].notna())
            & (filtered_df["ID output"].astype(str).str.strip() != "")
            & (filtered_df["ID output"].astype(str) != "Span only")
        ]

    if not selected_tools:
        st.warning("Select at least one **Tool** in the sidebar.")
    elif filtered_df.empty:
        st.info(
            "No rows match the current filters - try widening **Tools**, **Entity types**, "
            "or clearing **Training corpora**."
        )

    # Main content tabs
    tab1, tab2, tab3, tab_corpora, tab5, tab6 = st.tabs([
        " Entity Coverage Overview",
        " Entity Explorer",
        "Training Corpora Analysis",
        "Benchmark Corpora",
        " Per-entity analysis",
        " Per-tool view",
    ])
    
    with tab1:
        st.header("Entity Coverage by Tools")

        # Build coverage matrix: rows = entity types, columns = tools, cells = ✓ or empty
        pairs = filtered_df[["Tool", "Entity type"]].drop_duplicates()
        if not pairs.empty:
            tools_sorted = sorted(pairs["Tool"].unique(), key=str.lower)
            entities_sorted = sorted(pairs["Entity type"].unique(), key=str.lower)
            covered = set(zip(pairs["Tool"], pairs["Entity type"]))

            matrix_data = []
            for ent in entities_sorted:
                row = {"Entity Type": ent}
                count = 0
                for tool in tools_sorted:
                    if (tool, ent) in covered:
                        row[tool] = "✓"
                        count += 1
                    else:
                        row[tool] = ""
                row["# Tools"] = count
                matrix_data.append(row)

            coverage_table = pd.DataFrame(matrix_data)
            # Sort by number of tools descending
            coverage_table = coverage_table.sort_values("# Tools", ascending=False)

            st.dataframe(
                coverage_table,
                use_container_width=True,
                hide_index=True,
                height=min(800, 40 + 35 * len(entities_sorted)),
            )

            # Summary below the table
            st.subheader("Coverage Statistics")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                entity_coverage = filtered_df.groupby("Entity type")["Tool"].nunique().sort_values(ascending=False)
                st.markdown("**Most Covered Entities:**")
                for entity, count in entity_coverage.head(10).items():
                    st.write(f"- **{entity}**: {count} tools")
            with col_s2:
                st.markdown("**Gaps (Single Tool Coverage):**")
                gaps = entity_coverage[entity_coverage == 1]
                if gaps.empty:
                    st.caption("No single-tool gaps in current filter.")
                else:
                    for entity, _c in gaps.items():
                        tool = filtered_df[filtered_df["Entity type"] == entity]["Tool"].iloc[0]
                        st.write(f"- **{entity}**: only {tool}")
        else:
            st.warning("No data matches current filters")
    
    with tab2:
        st.header("Entity Explorer")
        st.markdown("Select an **entity type** to see every tool that recognises it, their training corpora, normalisation tools, and ID outputs.")

        explorer_entities = sorted(filtered_df["Entity type"].astype(str).unique(), key=str.lower)
        if not explorer_entities:
            st.warning("No entities match the current filters.")
        else:
            pick_ent = st.selectbox("Entity type", options=explorer_entities, key="explorer_entity_pick")
            ent_rows = filtered_df[filtered_df["Entity type"] == pick_ent].copy()

            # Sunburst: Entity → Tool → Training Corpus → ID output ──
            sun_records = []
            for _, r in ent_rows.iterrows():
                tool = r["Tool"]
                id_out = str(r["ID output"]).strip() if pd.notna(r["ID output"]) else "Span only"
                if not id_out or id_out == "nan":
                    id_out = "Span only"
                corpora_raw = r.get("NER training corpus")
                if pd.notna(corpora_raw) and str(corpora_raw).strip():
                    for corpus in [c.strip() for c in str(corpora_raw).split(",") if c.strip()]:
                        sun_records.append({"Entity": pick_ent, "Tool": tool, "Corpus": corpus, "ID Output": id_out})
                else:
                    sun_records.append({"Entity": pick_ent, "Tool": tool, "Corpus": "-", "ID Output": id_out})
            if sun_records:
                sun_df = pd.DataFrame(sun_records).drop_duplicates()
                fig_sun = px.sunburst(
                    sun_df,
                    path=["Entity", "Tool", "Corpus", "ID Output"],
                    title=f"{pick_ent}: Tools → Training Corpora → ID Outputs",
                    color_discrete_sequence=NS_DISCRETE_PALETTE,
                )
                fig_sun.update_layout(
                    height=520,
                    paper_bgcolor=THEME_PLOT_PAPER,
                    font=dict(color=THEME_PLOT_FONT),
                    margin=dict(t=50, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_sun, use_container_width=True)

            # Detail cards per tool 
            st.subheader(f"Tools recognising **{pick_ent}**")
            for tool_name in sorted(ent_rows["Tool"].unique(), key=str.lower):
                tool_slice = ent_rows[ent_rows["Tool"] == tool_name]
                with st.expander(f"**{tool_name}**", expanded=True):
                    info_cols = st.columns([1, 1, 1])
                    with info_cols[0]:
                        tasks = ", ".join(sorted(tool_slice["Task"].dropna().unique()))
                        st.markdown(f"**Task:** {tasks}" if tasks else "**Task:** -")
                        if "Entity_Type_Original" in tool_slice.columns:
                            orig = ", ".join(sorted(tool_slice["Entity_Type_Original"].dropna().unique()))
                            if orig:
                                st.markdown(f"**Original name:** {orig}")
                    with info_cols[1]:
                        corpora = set()
                        for raw in tool_slice["NER training corpus"].dropna():
                            corpora.update(c.strip() for c in str(raw).split(",") if c.strip())
                        st.markdown("**Training corpora:**")
                        if corpora:
                            for c in sorted(corpora):
                                st.write(f"  - {c}")
                        else:
                            st.caption("-")
                    with info_cols[2]:
                        ids = sorted({str(v).strip() for v in tool_slice["ID output"].dropna() if str(v).strip() and str(v).strip() != "Span only"})
                        norms = sorted({str(v).strip() for v in tool_slice["Normalization tool"].dropna() if str(v).strip()}) if "Normalization tool" in tool_slice.columns else []
                        st.markdown("**ID output:**")
                        if ids:
                            for v in ids:
                                st.write(f"  - {v}")
                        else:
                            st.caption("Span only")
                        if norms:
                            st.markdown("**Normalization tool:**")
                            for v in norms:
                                st.write(f"  - {v}")
                        url_vals = tool_slice["URL"].dropna().unique() if "URL" in tool_slice.columns else []
                        for u in url_vals:
                            u = str(u).strip()
                            if u:
                                st.markdown(f"[Link]({u})")
    
    with tab3:
        st.header("Training Corpora Analysis")

        corpus_row_hits = Counter()
        corpus_to_tools: dict[str, set] = defaultdict(set)
        for _, row in filtered_df.iterrows():
            if pd.notna(row["NER training corpus"]):
                for corpus in _corpus_tokens(row["NER training corpus"]):
                    corpus_row_hits[corpus] += 1
                    corpus_to_tools[corpus].add(row["Tool"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most used corpora")

            if corpus_row_hits:
                metrics_rows = [
                    {
                        "Corpus": corpus,
                        "Distinct tools": len(corpus_to_tools[corpus]),
                        "Row mentions": corpus_row_hits[corpus],
                        "Tools": ", ".join(sorted(corpus_to_tools[corpus])),
                    }
                    for corpus in corpus_row_hits
                ]
                metrics_rows.sort(
                    key=lambda r: (r["Distinct tools"], r["Row mentions"]),
                    reverse=True,
                )
                corpus_df = pd.DataFrame(metrics_rows[:20])

                fig_corpus = px.bar(
                    corpus_df,
                    x="Distinct tools",
                    y="Corpus",
                    title="Training corpora by number of distinct tools",
                    orientation="h",
                    height=600,
                    color="Corpus",
                    color_discrete_sequence=NS_DISCRETE_PALETTE,
                    hover_data=["Row mentions", "Tools"],
                )
                fig_corpus.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor=THEME_PLOT_PAPER,
                    plot_bgcolor=THEME_PLOT_BG,
                    font=dict(color=THEME_PLOT_FONT),
                )
                fig_corpus.update_traces(marker_line_width=0)
                st.plotly_chart(fig_corpus, use_container_width=True)
            else:
                st.warning("No corpus data available")
        
        with col2:
            st.subheader("Corpus details & filter")
            

            if corpus_row_hits:
                st.markdown("**Corpora shared across multiple tools:**")
                shown = 0
                for row in sorted(
                    (
                        {
                            "corpus": c,
                            "n_tools": len(corpus_to_tools[c]),
                            "n_rows": corpus_row_hits[c],
                            "tools": sorted(corpus_to_tools[c]),
                        }
                        for c in corpus_row_hits
                    ),
                    key=lambda r: (-r["n_tools"], -r["n_rows"], r["corpus"].lower()),
                ):
                    if row["n_tools"] < 2:
                        continue
                    st.write(
                        f"**{row['corpus']}** - **{row['n_tools']} tools**, "
                        f"{row['n_rows']} table row(s)"
                    )
                    st.write(f"   → {', '.join(row['tools'])}")
                    st.write("")
                    shown += 1
                    if shown >= 12:
                        break
                if shown == 0:
                    st.caption("No corpus appears under more than one tool in the current filter.")
            else:
                st.caption("No corpus data in the current filter.")

    with tab_corpora:
        st.header("Benchmark Corpora")
        st.markdown(
            "Every corpus referenced by the tools in this explorer. Public benchmark corpora "
            "(via Hugging Face `bigbio`/`spyysalo`) carry full metadata — entity types, "
            "normalization DBs, relations, and split sizes. Restricted clinical sets, "
            "free-text combinations, and methods/dictionaries are included for completeness "
            "with an explanatory note and blank fields where no verified data exists."
        )

        bench_wide_path = SCRIPT_DIR / "benchmark_summary.csv"
        bench_long_path = SCRIPT_DIR / "benchmark_summary_by_entity.csv"

        if not bench_wide_path.exists() or not bench_long_path.exists():
            st.warning(
                "Benchmark CSVs not found. Place `benchmark_summary.csv` and "
                "`benchmark_summary_by_entity.csv` next to this script."
            )
        else:
            bench_wide = pd.read_csv(bench_wide_path)
            bench_long = pd.read_csv(bench_long_path)

            def _fmt_count(v):
                if pd.isna(v):
                    return "-"
                return f"{int(v):,}"

            corpus_options = sorted(bench_wide["corpus"].astype(str).unique(), key=str.lower)
            if not corpus_options:
                st.warning("No corpora found in the benchmark CSVs.")
            else:
                pick_corpus = st.selectbox(
                    "Corpus",
                    options=corpus_options,
                    key="bench_corpus_pick",
                )
                row = bench_wide[bench_wide["corpus"] == pick_corpus].iloc[0]
                sub = bench_long[bench_long["corpus"] == pick_corpus].copy()

                ents = (
                    [e.strip() for e in str(row["entity_types"]).split(";") if e.strip()]
                    if pd.notna(row["entity_types"]) else []
                )
                rels = (
                    [r.strip() for r in str(row["relation_types"]).split(";") if r.strip()]
                    if pd.notna(row["relation_types"]) else []
                )
                norms = (
                    [n.strip() for n in str(row["normalization_dbs"]).split(";") if n.strip()]
                    if pd.notna(row["normalization_dbs"]) else []
                )

                note = str(row["notes"]).strip() if "notes" in row and pd.notna(row["notes"]) else ""
                if note:
                    st.info(note)

                st.subheader(f"All entity rows for {pick_corpus}")
                display_cols = [
                    "corpus", "granularity", "entity_type",
                    "is_normalized", "normalization_dbs",
                    "corpus_has_relations",
                    "train_rows", "validation_rows", "test_rows",
                    "hf_dataset", "notes",
                ]
                cols = [c for c in display_cols if c in sub.columns]
                st.dataframe(
                    sub[cols].rename(columns={
                        "corpus": "Corpus",
                        "granularity": "Granularity",
                        "entity_type": "Entity type",
                        "is_normalized": "Normalized",
                        "normalization_dbs": "Normalization DBs",
                        "corpus_has_relations": "Corpus has relations",
                        "train_rows": "Train",
                        "validation_rows": "Validation",
                        "test_rows": "Test",
                        "hf_dataset": "Hugging Face dataset",
                        "notes": "Notes",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Entity types", int(row["num_entity_types"]))
                with m2:
                    gran = str(row["granularity"]) if pd.notna(row["granularity"]) else "—"
                    st.metric("Granularity", gran)
                with m3:
                    st.metric("Normalized", "Yes" if str(row["has_normalization"]) == "yes" else "No")
                with m4:
                    st.metric("Relations", f"{len(rels)} types" if rels else "None")

                if pd.notna(row["hf_dataset"]) and str(row["hf_dataset"]).strip():
                    st.markdown(f"**Hugging Face:** `{row['hf_dataset']}`")

                agg_left, agg_right = st.columns(2)
                with agg_left:
                    st.subheader("Entity types")
                    if ents:
                        for e in ents:
                            st.write(f"- {e}")
                    else:
                        st.caption("-")
                    st.subheader("Normalization DBs")
                    if norms:
                        for n in norms:
                            st.write(f"- {n}")
                    else:
                        st.caption("None")
                with agg_right:
                    st.subheader("Splits")
                    st.markdown(
                        f"- Train: {_fmt_count(row['train_rows'])}  \n"
                        f"- Validation: {_fmt_count(row['validation_rows'])}  \n"
                        f"- Test: {_fmt_count(row['test_rows'])}"
                    )
                    st.subheader("Relations")
                    if rels:
                        for r in rels:
                            st.write(f"- {r}")
                    else:
                        st.caption("None")

    with tab5:
        st.header("Per-entity analysis")
        
        ent_options = sorted(filtered_df["Entity type"].astype(str).unique(), key=str.lower)
        if not ent_options:
            st.warning("No entities match the current filters.")
        else:
            pick = st.selectbox(
                "Entity type",
                options=ent_options,
                key="per_entity_pick_select",
            )
            sub = filtered_df[filtered_df["Entity type"] == pick].copy()

            if "Entity_Type_Original" in sub.columns:
                orig_names = (
                    sub[["Tool", "Entity_Type_Original"]]
                    .dropna(subset=["Entity_Type_Original"])
                    .drop_duplicates()
                    .sort_values("Tool")
                )
                if not orig_names.empty:
                    st.subheader("Original entity names by tool")
                    st.dataframe(
                        orig_names.rename(columns={"Entity_Type_Original": "Original Name"}),
                        use_container_width=True,
                        hide_index=True,
                    )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Tools covering this entity", sub["Tool"].nunique())
            with m2:
                st.metric("Table rows (tool configurations)", len(sub))
            with m3:
                norm_mask = (
                    sub["ID output"].notna()
                    & (sub["ID output"].astype(str).str.strip() != "")
                    & (sub["ID output"].astype(str) != "Span only")
                )
                st.metric("Tools with ID normalization", sub.loc[norm_mask, "Tool"].nunique())
            with m4:
                st.metric("Distinct tasks", sub["Task"].dropna().nunique())

            agg_left, agg_right = st.columns(2)
            with agg_left:
                st.subheader("Training corpora (aggregated)")
                corpus_parts: list[str] = []
                for raw in sub["NER training corpus"].dropna():
                    corpus_parts.extend(
                        [c.strip() for c in str(raw).split(",") if c.strip()]
                    )
                corpora_unique = sorted(set(corpus_parts), key=str.lower)
                if corpora_unique:
                    for c in corpora_unique:
                        st.write(f"- {c}")
                else:
                    st.caption("No training corpus text for this entity in the current view.")

            with agg_right:
                st.subheader("ID outputs & normalization")
                st.markdown("**ID output** (unique)")
                id_vals = sorted(
                    {
                        str(v).strip()
                        for v in sub["ID output"].dropna()
                        if str(v).strip() and str(v).strip() != "Span only"
                    }
                )
                if id_vals:
                    for v in id_vals:
                        st.write(f"• {v}")
                else:
                    st.caption("None or span-only under current filters.")

                if "Normalization tool" in sub.columns:
                    st.markdown("**Normalization tools** (unique)")
                    nt_vals = sorted(
                        {
                            str(v).strip()
                            for v in sub["Normalization tool"].dropna()
                            if str(v).strip()
                        }
                    )
                    if nt_vals:
                        for v in nt_vals:
                            st.write(f"• {v}")
                    else:
                        st.caption("-")

            st.subheader("All tool rows for this entity")
            preferred_cols = [
                "Tool",
                "Entity type",
                "Entity_Type_Original",
                "Task",
                "NER training corpus",
                "Normalization tool",
                "ID output",
                "URL",
            ]
            show_cols = [c for c in preferred_cols if c in sub.columns]
            display_sub = sub[show_cols].sort_values(
                ["Tool", "Task"] if "Task" in show_cols else ["Tool"]
            )
            if "Entity_Type_Original" in display_sub.columns:
                display_sub = display_sub.rename(columns={"Entity_Type_Original": "Original Name"})
            st.dataframe(
                display_sub,
                use_container_width=True,
                hide_index=True,
            )

    with tab6:
        st.header("Per-tool view")
        st.markdown("Select a **tool** to see every entity it covers, original entity names, corpora, normalisation, and ID outputs.")

        tool_options = sorted(filtered_df["Tool"].astype(str).unique(), key=str.lower)
        if not tool_options:
            st.warning("No tools match the current filters.")
        else:
            pick_tool = st.selectbox("Tool", options=tool_options, key="per_tool_pick")
            tool_rows = filtered_df[filtered_df["Tool"] == pick_tool].copy()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Entity types", tool_rows["Entity type"].nunique())
            with m2:
                tasks_str = ", ".join(sorted(tool_rows["Task"].dropna().unique()))
                st.metric("Tasks", tasks_str if tasks_str else "-")
            with m3:
                norm_count = tool_rows[
                    (tool_rows["ID output"].notna())
                    & (tool_rows["ID output"].astype(str).str.strip() != "")
                    & (tool_rows["ID output"].astype(str) != "Span only")
                ]["Entity type"].nunique()
                st.metric("Entities with ID normalization", norm_count)

            # URL
            urls = tool_rows["URL"].dropna().unique() if "URL" in tool_rows.columns else []
            for u in urls:
                u = str(u).strip()
                if u:
                    st.markdown(f"**URL:** [{u}]({u})")

            # Entity breakdown table
            st.subheader("Entity breakdown")
            breakdown_rows = []
            for _, r in tool_rows.iterrows():
                orig = str(r.get("Entity_Type_Original", "")).strip() if pd.notna(r.get("Entity_Type_Original")) else ""
                corpora = str(r["NER training corpus"]).strip() if pd.notna(r["NER training corpus"]) else ""
                norm = str(r["Normalization tool"]).strip() if pd.notna(r.get("Normalization tool")) else ""
                id_out = str(r["ID output"]).strip() if pd.notna(r["ID output"]) else ""
                notes = str(r["Notes"]).strip() if pd.notna(r.get("Notes")) else ""
                breakdown_rows.append({
                    "Entity Type": r["Entity type"],
                    "Original Name": orig,
                    "Training Corpora": corpora,
                    "Normalization Tool": norm,
                    "ID Output": id_out,
                    "Notes": notes,
                })
            bd = pd.DataFrame(breakdown_rows)
            # Drop empty columns
            bd = bd.loc[:, bd.astype(str).ne("").any()]
            st.dataframe(bd.sort_values("Entity Type"), use_container_width=True, hide_index=True)

    # Summary statistics at bottom
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tools in View", 
            filtered_df['Tool'].nunique(),
        )
    
    with col2:
        st.metric(
            "Entity Types",
            filtered_df['Entity type'].nunique(), 
        )
    
    with col3:
        # Count tools with normalization
        with_norm = filtered_df[
            (filtered_df['ID output'].notna()) &
            (filtered_df['ID output'] != 'Span only')
        ]['Tool'].nunique()
        
        _tn = filtered_df["Tool"].nunique()
        st.metric(
            "Tools with Normalization",
            with_norm,
        )
    
    with col4:
        # Count unique corpora
        all_corpora = set()
        for corpus_str in filtered_df['NER training corpus'].dropna():
            corpora = [c.strip() for c in str(corpus_str).split(',') if c.strip()]
            all_corpora.update(corpora)
        
        st.metric(
            "Training Corpora",
            len(all_corpora)
        )

else:
    st.markdown("""
    ### 👋 Welcome to the Biomedical NER Tool Explorer!
    
    This app helps you explore the ecosystem of biomedical named entity recognition tools, their entity coverage, training data, and output formats.
    
    **To get started:**
    1. Place the bundled CSV next to this script, or upload a file in the sidebar
    2. Use filters to focus on specific tool categories or entity types  
    3. Explore the different views:
       - **Entity Coverage**: See which tools cover which entities
       - **Entity Explorer**: Pick an entity to see every tool that recognises it
       - **Training Corpora**: Discover shared training datasets across tools
       - **Benchmark Corpora**: 5 annotated corpora (BC5CDR, BC2GM, NCBI Disease, BioRED, JNLPBA), 18 entity types, normalization DBs, relations, splits
       - **Per-entity analysis**: One entity at a time - tables, corpora, IDs, charts
       - **Per-tool view**: Drill into a single tool's entities and outputs
    
    **Key insights you can discover:**
    - Which entity types are well-covered vs gaps with single-tool coverage
    - Which training corpora are shared across multiple tools (like BioRED)
    - Which tools converge on the same ID formats (like UMLS CUI)
    - Clinical vs biomedical tool specialization patterns
    
    Perfect for planning annotation pipelines, identifying complementary tools, and understanding the biomedical NER landscape.
    """)
    
    # example of expected CSV format
    st.subheader("Expected CSV Format")
    example_data = pd.DataFrame({
        'Tool': ['PubTator3', 'PubTator3', 'BERN2'],
        'Task': ['NER+NEN+RE', 'NER+NEN+RE', 'NER+NEN'],
        'Entity type': ['Gene / Protein', 'Disease', 'Gene / Protein'],
        'NER training corpus': ['NLM-Gene, BioRED', 'NCBI Disease, BC5CDR', 'BC2GM'],
        'Normalization tool': ['GNorm2', 'TaggerOne', 'BioSyn hybrid'],
        'ID output': ['NCBI Gene ID', 'MeSH ID', 'NCBI Gene ID']
    })
    st.dataframe(example_data)
