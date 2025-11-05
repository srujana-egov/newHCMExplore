import streamlit as st
import json
import requests
from typing import List, Dict, Any, Tuple, Optional, Set
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()

# Components
from search_utils import GraphSearcher
from data import NODES, EDGES
from slack_integration import send_slack_review_request
from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------
# Initialization
# ---------------------------
@st.cache_resource
def get_searcher():
    return GraphSearcher(nodes=NODES, edges=EDGES)

searcher = get_searcher()

# ---------------------------
# Graph indices (roots, children, parents) + end_user options
# ---------------------------
@st.cache_resource
def _build_index(nodes: List[Dict], edges: List[Dict]):
    node_map = {n["id"]: n for n in nodes}
    children_map: Dict[str, List[str]] = {}
    parents_map: Dict[str, List[str]] = {}

    indegree: Dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        src, tgt = e["source"], e["target"]
        children_map.setdefault(src, []).append(tgt)
        parents_map.setdefault(tgt, []).append(src)
        indegree[tgt] = indegree.get(tgt, 0) + 1
        indegree.setdefault(src, indegree.get(src, 0))

    roots = [nid for nid, d in indegree.items() if d == 0]

    # unique end_user values (strings only)
    end_users = sorted({str(n.get("end_user")) for n in nodes if n.get("end_user") not in (None, "", [])})
    return node_map, children_map, parents_map, roots, end_users

NODE_MAP, CHILDREN_MAP, PARENTS_MAP, ROOT_IDS, END_USER_OPTIONS = _build_index(NODES, EDGES)

# ---------------------------
# Session state
# ---------------------------
def _ensure_state():
    if "visible_nodes" not in st.session_state:
        st.session_state.visible_nodes: Set[str] = set(ROOT_IDS)
    if "visible_edges" not in st.session_state:
        st.session_state.visible_edges: Set[Tuple[str, str]] = set()
    if "highlight_ids" not in st.session_state:
        st.session_state.highlight_ids: Set[str] = set()  # search-based highlights (teal)
    if "role_highlight_ids" not in st.session_state:
        st.session_state.role_highlight_ids: Set[str] = set()  # end_user-based highlights (orange)
    if "focus_node_id" not in st.session_state:
        st.session_state.focus_node_id = None
    if "last_query_text" not in st.session_state:
        st.session_state.last_query_text = ""
    if "last_similar_nodes" not in st.session_state:
        st.session_state.last_similar_nodes: List[Dict[str, Any]] = []
    if "last_rag_response" not in st.session_state:
        st.session_state.last_rag_response = None
    if "details_node_id" not in st.session_state:
        st.session_state.details_node_id = None  # which node’s details to show inline
    if "filter_end_users" not in st.session_state:
        st.session_state.filter_end_users: List[str] = []  # chosen end_user values
    if "prev_filter_snapshot" not in st.session_state:
        st.session_state.prev_filter_snapshot = tuple()

# ---------------------------
# Expand/Collapse
# ---------------------------
def _expand_node(node_id: str):
    for child_id in CHILDREN_MAP.get(node_id, []):
        st.session_state.visible_nodes.add(child_id)
        st.session_state.visible_edges.add((node_id, child_id))

def _collect_descendants(root_id: str) -> Set[str]:
    seen: Set[str] = set()
    q = deque(CHILDREN_MAP.get(root_id, []))
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        for c in CHILDREN_MAP.get(nid, []):
            if c not in seen:
                q.append(c)
    return seen

def _collapse_subtree(root_id: str):
    descendants = _collect_descendants(root_id)
    collapsing_starts = {root_id} | descendants
    to_remove_edges = {(u, v) for (u, v) in st.session_state.visible_edges if u in collapsing_starts}
    st.session_state.visible_edges.difference_update(to_remove_edges)

    still_visible_edges = st.session_state.visible_edges
    parents_of = {}
    for (u, v) in still_visible_edges:
        parents_of[v] = parents_of.get(v, 0) + 1

    to_hide_nodes: Set[str] = set()
    for nid in descendants:
        if parents_of.get(nid, 0) == 0:
            to_hide_nodes.add(nid)

    st.session_state.visible_nodes.difference_update(to_hide_nodes)
    st.session_state.visible_edges = {(u, v) for (u, v) in st.session_state.visible_edges if v not in to_hide_nodes}

def _expand_all():
    st.session_state.visible_nodes = {n["id"] for n in NODES}
    st.session_state.visible_edges = {(e["source"], e["target"]) for e in EDGES}

def _expand_to_node(node_id: str):
    st.session_state.visible_nodes.add(node_id)
    frontier = deque([node_id])
    seen = set()
    while frontier:
        cur = frontier.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for p in PARENTS_MAP.get(cur, []):
            st.session_state.visible_nodes.add(p)
            st.session_state.visible_edges.add((p, cur))
            frontier.append(p)

# ---------------------------
# Event helper (click id)
# ---------------------------
def _get_clicked_node_id(graph_event: Any) -> Optional[str]:
    if not graph_event:
        return None
    if isinstance(graph_event, str):
        return graph_event
    if isinstance(graph_event, list) and graph_event and isinstance(graph_event[0], str):
        return graph_event[0]
    if isinstance(graph_event, dict):
        sel = graph_event.get("selected") or graph_event.get("selection")
        if isinstance(sel, dict):
            nodes = sel.get("nodes") or sel.get("nodeIds")
            if nodes:
                return nodes[0]
        if graph_event.get("type") in ("click", "select") and isinstance(graph_event.get("id"), str):
            return graph_event["id"]
        if isinstance(graph_event.get("nodes"), list) and graph_event["nodes"]:
            return graph_event["nodes"][0]
    return None

# ---------------------------
# Details panel rendering (inline under the graph)
# ---------------------------
def _node_label(n: Dict[str, Any], default_id: str) -> str:
    return n.get("label") or default_id

def _node_content(n: Dict[str, Any]) -> str:
    body = n.get("content")
    if isinstance(body, str) and body.strip():
        return body.strip()
    slim = {k: v for k, v in n.items() if k not in {"id", "label", "url"}}
    try:
        return "```json\n" + json.dumps(slim, indent=2) + "\n```"
    except Exception:
        return "_No additional content available._"

def _node_url(n: Dict[str, Any]) -> Optional[str]:
    u = n.get("url")
    return u.strip() if isinstance(u, str) and u.strip() else None

def _render_details_panel(node: Dict[str, Any]):
    st.markdown(
        """
        <style>
          .kg-card { border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px; }
          .kg-title { font-size: 20px; margin: 0 0 6px 0; }
          .kg-meta { opacity: 0.75; font-size: 13px; margin-bottom: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    title = _node_label(node, node.get("id", "Node"))
    body = _node_content(node)
    url = _node_url(node)

    with st.container():
        st.markdown(f"<div class='kg-card'><div class='kg-title'>{title}</div>", unsafe_allow_html=True)
        end_user = node.get("end_user")
        if end_user:
            st.markdown(f"<div class='kg-meta'>End user: {end_user}</div>", unsafe_allow_html=True)

        if body.strip().startswith("```"):
            st.markdown(body)
        else:
            st.markdown(body)

        st.markdown("</div>", unsafe_allow_html=True)
        if url:
            st.link_button("Open node ↗", url, use_container_width=False)

# ---------------------------
# Graph search (GraphSearcher) with threshold
# ---------------------------
def _normalize_result_item(item: Any) -> Optional[Dict[str, Any]]:
    def _get_attr(o, names, default=None):
        for n in names:
            if isinstance(o, dict) and n in o:
                return o[n]
            if hasattr(o, n):
                return getattr(o, n)
        return default
    def _coerce_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if isinstance(item, dict):
        node_id = _get_attr(item, ["id", "node_id", "nid"])
        score = _get_attr(item, ["score", "similarity", "sim"])
        distance = _get_attr(item, ["distance"])
        label = _get_attr(item, ["label"])
        if score is None and distance is not None:
            d = _coerce_float(distance)
            score = (1.0 / (1.0 + d)) if (d is not None and d >= 0) else None
        score = _coerce_float(score)
        if node_id is None or score is None:
            return None
        if label is None:
            label = NODE_MAP.get(node_id, {}).get("label", node_id)
        return {"node_id": str(node_id), "label": str(label), "score": float(score)}

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        node_id = item[0]
        score = _coerce_float(item[1])
        if score is None:
            score = _coerce_float(_get_attr(item, ["score", "similarity", "sim"]))
        if score is None:
            d = _coerce_float(_get_attr(item, ["distance"]))
            if d is not None and d >= 0:
                score = 1.0 / (1.0 + d)
        if score is None:
            return None
        label = item[2] if len(item) >= 3 else NODE_MAP.get(node_id, {}).get("label", node_id)
        return {"node_id": str(node_id), "label": str(label), "score": float(score)}

    node_id = _get_attr(item, ["id", "node_id", "nid"])
    if node_id is None:
        node_obj = _get_attr(item, ["node"])
        node_id = _get_attr(node_obj, ["id"]) if node_obj is not None else None
    score = _get_attr(item, ["score", "similarity", "sim"])
    if score is None:
        d = _get_attr(item, ["distance"])
        d = _coerce_float(d)
        if d is not None and d >= 0:
            score = 1.0 / (1.0 + d)
    score = _coerce_float(score)
    label = _get_attr(item, ["label"])
    if label is None:
        node_obj = _get_attr(item, ["node"])
        if node_obj is not None:
            label = _get_attr(node_obj, ["label", "name"])
    if node_id is None or score is None:
        return None
    if label is None:
        label = NODE_MAP.get(node_id, {}).get("label", node_id)
    return {"node_id": str(node_id), "label": str(label), "score": float(score)}

def find_similar_nodes_with_searcher(query: str, k: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
    results_raw = None
    for method_name in ["search", "similar", "query", "knn", "find"]:
        if hasattr(searcher, method_name):
            try:
                method = getattr(searcher, method_name)
                try:
                    results_raw = method(query, k=k)  # type: ignore
                except TypeError:
                    results_raw = method(query)       # type: ignore
                break
            except Exception:
                continue
    if results_raw is None:
        return []
    if isinstance(results_raw, dict) and "results" in results_raw:
        results_iter = results_raw["results"]
    elif hasattr(results_raw, "results"):
        results_iter = getattr(results_raw, "results")
    else:
        results_iter = results_raw

    normalized: List[Dict[str, Any]] = []
    for item in results_iter:
        norm = _normalize_result_item(item)
        if norm is not None:
            normalized.append(norm)

    filtered = [n for n in normalized if n["score"] is not None and float(n["score"]) >= threshold]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:k]

# ---------------------------
# Graph rendering (expand/collapse + highlight + focus)
# ---------------------------
def render_graph() -> Any:
    # Visual priority: focus (pink) > role_highlight (orange) > search_highlight (teal) > default
    search_high = st.session_state.highlight_ids or set()
    role_high = st.session_state.role_highlight_ids or set()
    focus_id = st.session_state.focus_node_id

    a_nodes = []
    for nid in st.session_state.visible_nodes:
        n = NODE_MAP[nid]
        base_label = n.get("label", nid)

        children = CHILDREN_MAP.get(nid, [])
        hidden_kids = any((cid not in st.session_state.visible_nodes) for cid in children)
        label = f"+ {base_label}" if children and hidden_kids else (f"– {base_label}" if children else base_label)

        # Determine color/size by priority
        if nid == focus_id:
            color = "#E91E63"   # pink
            size = 34
            font_size = 14
        elif nid in role_high:
            color = "#FFA500"   # orange
            size = 28
            font_size = 13
        elif nid in search_high:
            color = "#2EC4B6"   # teal
            size = 28
            font_size = 13
        else:
            color = "#324563"
            size = 24
            font_size = 12

        a_nodes.append(
            Node(
                id=nid,
                label=label,
                size=size,
                shape="dot",
                color=color,
                font={"size": font_size, "color": "#FFFFFF", "face": "Arial"},
                borderWidth=0,
                borderWidthSelected=3,
            )
        )

    # Edge color logic: orange if touches role-highlight; else gold-ish if touches focus/search; else grey
    a_edges = []
    for (src, tgt) in st.session_state.visible_edges:
        if src in role_high or tgt in role_high:
            e_color = "#FFA500"  # orange
            e_width = 1.2
        elif src == focus_id or tgt == focus_id or src in search_high or tgt in search_high:
            e_color = "#FF9F1C"  # gold highlight
            e_width = 1.2
        else:
            e_color = "#8a8a8a"
            e_width = 0.9
        a_edges.append(
            Edge(
                source=src,
                target=tgt,
                type="CURVE_SMOOTH",
                width=e_width,
                color=e_color,
                smooth={"type": "curvedCW", "roundness": 0.2},
            )
        )

    config = Config(
        width="100%",
        height=700,
        directed=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        layout={
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 250,
                "nodeSpacing": 200,
                "treeSpacing": 250,
                "direction": "UD",
                "sortMethod": "hubsize",
                "shakeTowards": "roots",
            }
        },
        physics={"enabled": False},
        node={
            "labelProperty": "label",
            "font": {"size": 12, "face": "Arial", "color": "#FFFFFF"},
            "size": 24,
            "borderWidth": 0,
            "borderWidthSelected": 3,
            "chosen": {
                "node": (
                    "function(values, id, selected, hovering) {"
                    "values.size = 29;"
                    'values.borderColor = "#FF6B6B";'
                    "}"
                )
            },
        },
        link={
            "highlightColor": "#FF9F1C",
            "renderLabel": False,
            "type": "curve",
            "width": 1.2,
            "color": {"color": "#8a8a8a", "highlight": "#FF9F1C", "hover": "#2EC4B6"},
        },
        interaction={
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
            "hideEdgesOnZoom": False,
            "multiselect": False,  # single-select for cleaner events
            "selectConnectedEdges": False,
        },
    )

    return agraph(nodes=a_nodes, edges=a_edges, config=config)

# ---------------------------
# RAG helpers
# ---------------------------
def render_pretty_answer(text: str) -> None:
    if not text:
        st.write("")
        return
    try:
        st.json(json.loads(text))
    except json.JSONDecodeError:
        st.markdown(text)

def query_rag_api(query: str) -> Tuple[bool, str]:
    try:
        url = "http://localhost:8000/query"
        headers = {"Content-Type": "application/json"}
        payload = {"query": query}
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        try:
            result = response.json()
            return True, result.get("answer", json.dumps(result, indent=2))
        except ValueError:
            return True, response.text
    except Exception as e:
        return False, f"Error querying RAG API: {str(e)}"

# ---------------------------
# Main
# ---------------------------
def main():
    st.set_page_config(page_title="Knowledge Graph Query", page_icon="🌐", layout="wide")
    _ensure_state()

    # --- Sidebar: end_user highlighter ---
    with st.sidebar:
        st.header("Highlight by End User")
        selected = st.multiselect("End user", END_USER_OPTIONS, default=st.session_state.filter_end_users)
        if tuple(selected) != st.session_state.prev_filter_snapshot:
            st.session_state.filter_end_users = list(selected)
            st.session_state.prev_filter_snapshot = tuple(selected)

            # Compute highlight set (orange) but DON'T change visibility
            selected_set = set(selected)
            st.session_state.role_highlight_ids = {
                n["id"] for n in NODES if str(n.get("end_user")) in selected_set
            }
            # No rerun needed; but to ensure consistent updates with some Streamlit/iframe combos, we can rerun safely:
            st.rerun()

        # Quick legend
        st.caption("Legend:")
        st.markdown(
            "- **Pink**: Focus\n"
            "- **Orange**: End user selected\n"
            "- **Teal**: Search matches\n"
            "- **Grey**: Others"
        )

    # --- Graph (click expands/collapses and shows details inline) ---
    graph_event = render_graph()

    clicked_id = _get_clicked_node_id(graph_event)
    if clicked_id:
        # Show details for clicked node
        st.session_state.details_node_id = clicked_id

        did_change = False
        children = CHILDREN_MAP.get(clicked_id, [])
        if children:
            any_hidden = any((cid not in st.session_state.visible_nodes) for cid in children)
            if any_hidden:
                _expand_node(clicked_id)
                did_change = True
            else:
                _collapse_subtree(clicked_id)
                did_change = True

        if did_change:
            st.rerun()

    # Render inline details (if any)
    if st.session_state.details_node_id and st.session_state.details_node_id in NODE_MAP:
        _render_details_panel(NODE_MAP[st.session_state.details_node_id])

    # --- Search (graph first; optional RAG) ---
    st.markdown("---")
    st.header("Search the Graph")

    with st.form("graph_first_form"):
        query = st.text_area(
            "Your Query (Graph Search first):",
            placeholder="Type your question here...",
            help="We search the knowledge graph first. If not found, you can ask AI (RAG).",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_graph_btn = st.form_submit_button("🔎 Search Graph", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.form_submit_button("🧹 Clear", type="secondary", use_container_width=True)

    if clear_btn:
        st.session_state.highlight_ids = set()
        st.session_state.role_highlight_ids = set()
        st.session_state.focus_node_id = None
        st.session_state.last_query_text = ""
        st.session_state.last_similar_nodes = []
        st.session_state.last_rag_response = None
        st.session_state.details_node_id = None
        st.toast("Cleared.", icon="🧼")
        st.rerun()

    if submit_graph_btn and query.strip():
        st.session_state.last_query_text = query.strip()

        best_match, all_matches = searcher.search(st.session_state.last_query_text)

        # Threshold filter ≥ 0.5
        similar = [r for r in all_matches if getattr(r, "score", 0) >= 0.5]

        if similar:
            _expand_all()
            st.session_state.highlight_ids = {r.node_id for r in similar}  # teal highlights
            st.session_state.focus_node_id = similar[0].node_id
            st.session_state.last_similar_nodes = [
                {
                    "node_id": r.node_id,
                    "label": (getattr(r, "node_data", {}) or {}).get("label", r.node_id),
                    "score": r.score,
                }
                for r in similar
            ]
            st.session_state.last_rag_response = None
            st.rerun()
        else:
            with st.spinner("🤖 Asking AI (no strong graph match found)..."):
                ok, rag = query_rag_api(st.session_state.last_query_text)
            st.session_state.last_rag_response = rag if ok else rag
            st.session_state.highlight_ids = set()
            st.session_state.focus_node_id = None
            st.session_state.last_similar_nodes = []
            st.rerun()

    # Show graph search results (if any)
    if st.session_state.last_similar_nodes:
        st.subheader("Graph Matches (similarity ≥ 0.5)")
        cols = st.columns(min(3, len(st.session_state.last_similar_nodes)))
        for idx, node in enumerate(st.session_state.last_similar_nodes):
            with cols[idx % 3]:
                with st.container(border=True):
                    nid2 = node["node_id"]
                    label = node["label"]
                    score = node["score"]
                    st.markdown(f"**{label}**  \n*Score: {score:.2f}*")
                    if st.button("Focus", key=f"focus_{nid2}"):
                        _expand_to_node(nid2)
                        st.session_state.focus_node_id = nid2
                        st.rerun()

        # Offer RAG only if user clicks
        if st.button("Not there? Ask AI (RAG) 🤖", use_container_width=True):
            with st.spinner("🤖 Asking AI..."):
                ok, rag = query_rag_api(st.session_state.last_query_text)
            st.session_state.last_rag_response = rag if ok else rag
            try:
                send_slack_review_request(
                    query=st.session_state.last_query_text,
                    rag_response=st.session_state.last_rag_response,
                    similar_nodes=st.session_state.last_similar_nodes,
                )
                st.toast("✅ Sent for review on Slack", icon="📤")
            except Exception as e:
                st.warning(f"⚠️ Could not send to Slack: {str(e)}")
            st.rerun()

    # Show RAG response (if any)
    if st.session_state.last_rag_response is not None:
        st.subheader("AI Answer")
        render_pretty_answer(st.session_state.last_rag_response)

    elif submit_graph_btn and not query.strip():
        st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
