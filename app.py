import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Smart Feedback & Review Dashboard",
    page_icon="🧩",
    layout="wide",
)

DATA_DIR = Path("data")


@st.cache_data
def load_data():
    documents = pd.read_csv(DATA_DIR / "documents.csv")
    review_events = pd.read_csv(DATA_DIR / "review_events.csv")
    comments = pd.read_csv(DATA_DIR / "comments.csv")

    documents["created_date"] = pd.to_datetime(documents["created_date"])
    review_events["event_date"] = pd.to_datetime(review_events["event_date"])
    comments["resolved"] = (
        comments["resolved"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    )

    if "repeated_flag" in comments.columns:
        comments["repeated_flag"] = (
            comments["repeated_flag"].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False}
            )
        )
    else:
        comments["repeated_flag"] = False

    return documents, review_events, comments


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "sop": "procedure",
        "qa": "quality review",
        "doc": "document",
        "xref": "cross reference",
        "cross reference": "reference",
        "referenced": "reference",
        "referencing": "reference",
        "formatting": "format",
        "formatted": "format",
        "ambiguity": "ambiguous",
        "unclear": "ambiguous",
        "missing": "absent",
        "incomplete": "absent",
        "delay": "late",
        "delayed": "late",
        "timeline": "schedule",
        "workflow": "process flow",
        "traceability": "linkage",
        "linking": "linkage",
    }

    tokens = [replacements.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def build_comment_features(comments: pd.DataFrame) -> pd.DataFrame:
    df = comments.copy()
    df["normalized_issue_text"] = df["issue_text"].apply(normalize_text)
    df["combined_text"] = (
        df["issue_category"].fillna("").astype(str).str.lower().str.strip()
        + " "
        + df["normalized_issue_text"].fillna("")
    ).str.strip()
    return df


def assign_similarity_clusters(comments: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    df = comments.copy()

    if df.empty:
        df["cluster_id"] = []
        df["cluster_size"] = []
        df["auto_repeated_flag"] = []
        return df

    text_series = df["combined_text"].fillna("").astype(str)

    if len(df) == 1:
        df["cluster_id"] = 1
        df["cluster_size"] = 1
        df["auto_repeated_flag"] = False
        return df

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(text_series)
    sim = cosine_similarity(tfidf)

    cluster_ids = [-1] * len(df)
    current_cluster = 0

    for i in range(len(df)):
        if cluster_ids[i] != -1:
            continue

        current_cluster += 1
        cluster_ids[i] = current_cluster

        for j in range(i + 1, len(df)):
            same_doc = df.iloc[i]["doc_id"] == df.iloc[j]["doc_id"]
            same_category = df.iloc[i]["issue_category"] == df.iloc[j]["issue_category"]

            adjusted_threshold = threshold
            if same_doc and same_category:
                adjusted_threshold = threshold - 0.08
            elif same_category:
                adjusted_threshold = threshold - 0.04

            if sim[i, j] >= adjusted_threshold:
                cluster_ids[j] = current_cluster

    df["cluster_id"] = cluster_ids
    cluster_sizes = df.groupby("cluster_id").size().to_dict()
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes)
    df["auto_repeated_flag"] = df["cluster_size"] >= 2
    return df


def compute_revision_burden(doc_id, review_events, comments):
    doc_events = review_events[review_events["doc_id"] == doc_id].copy()
    doc_comments = comments[comments["doc_id"] == doc_id].copy()

    total_comments = len(doc_comments)
    total_rounds = int(doc_events["round_no"].max()) if not doc_events.empty else 0
    repeated_issues = int(doc_comments["auto_repeated_flag"].sum()) if not doc_comments.empty else 0
    unresolved = int((doc_comments["resolved"] == False).sum()) if not doc_comments.empty else 0
    delay_days = float(doc_events["duration_days"].sum()) if not doc_events.empty else 0.0
    reviewer_count = doc_comments["reviewer"].nunique() if not doc_comments.empty else 0

    score = (
        total_comments * 1.0
        + total_rounds * 2.2
        + repeated_issues * 2.5
        + unresolved * 1.7
        + delay_days * 0.55
        + reviewer_count * 0.8
    )

    if score < 15:
        label = "Low"
    elif score < 30:
        label = "Moderate"
    else:
        label = "High"

    return round(score, 1), label


def make_summary_metrics(documents, review_events, comments):
    total_docs = len(documents)
    active_comments = int((comments["resolved"] == False).sum())
    avg_rounds = round(documents["total_rounds"].mean(), 1) if not documents.empty else 0.0
    delayed_items = int(documents["delay_risk"].isin(["Medium", "High"]).sum())
    recurrent_clusters = int(comments[comments["auto_repeated_flag"]]["cluster_id"].nunique())

    return total_docs, active_comments, avg_rounds, delayed_items, recurrent_clusters


def build_document_overview(documents, review_events, comments):
    rows = []

    for _, row in documents.iterrows():
        doc_id = row["doc_id"]
        doc_comments = comments[comments["doc_id"] == doc_id]

        open_comments = int((doc_comments["resolved"] == False).sum())
        reviewer_count = doc_comments["reviewer"].nunique()
        repeated_clusters = int(
            doc_comments[doc_comments["auto_repeated_flag"]]["cluster_id"].nunique()
        )
        burden_score, burden_label = compute_revision_burden(doc_id, review_events, comments)

        rows.append(
            {
                "doc_id": doc_id,
                "title": row["title"],
                "document_type": row["document_type"],
                "team": row["team"],
                "status": row["current_status"],
                "review_rounds": row["total_rounds"],
                "reviewers": reviewer_count,
                "open_comments": open_comments,
                "repeated_clusters": repeated_clusters,
                "delay_risk": row["delay_risk"],
                "burden_score": burden_score,
                "burden_level": burden_label,
            }
        )

    return pd.DataFrame(rows)


def generate_pattern_summary(selected_doc, selected_comments, selected_events, burden_score, burden_label):
    total_rounds = int(selected_doc["total_rounds"])
    unresolved = int((selected_comments["resolved"] == False).sum()) if not selected_comments.empty else 0
    repeated_clusters = (
        selected_comments[selected_comments["auto_repeated_flag"]]["cluster_id"].nunique()
        if not selected_comments.empty
        else 0
    )

    if selected_comments.empty:
        top_issue = "no dominant issue pattern"
    else:
        top_issue_df = (
            selected_comments.groupby("issue_category")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        top_issue = top_issue_df.iloc[0]["issue_category"]

    if selected_events.empty:
        bottleneck_stage = "no bottleneck stage detected"
    else:
        bottleneck_df = (
            selected_events.groupby("event_type", as_index=False)["duration_days"]
            .sum()
            .sort_values("duration_days", ascending=False)
        )
        bottleneck_stage = bottleneck_df.iloc[0]["event_type"]

    if repeated_clusters >= 2:
        pattern_line = "The review pattern suggests recurring structural friction rather than isolated one-time comments."
    elif unresolved >= 3:
        pattern_line = "The current review state still appears unstable, with unresolved issues accumulating across rounds."
    else:
        pattern_line = "The review pattern appears relatively contained, though some friction remains visible."

    summary = (
        f"This document is currently '{selected_doc['current_status']}' after {total_rounds} review round(s). "
        f"The revision burden is {burden_label.lower()} (score: {burden_score}). "
        f"The most common issue category is '{top_issue}', and the longest delay is concentrated in '{bottleneck_stage}'. "
        f"There are {unresolved} unresolved comment(s) and {repeated_clusters} repeated issue cluster(s). "
        f"{pattern_line}"
    )
    return summary


def find_cluster_examples(comments: pd.DataFrame) -> pd.DataFrame:
    repeated = comments[comments["auto_repeated_flag"]].copy()
    if repeated.empty:
        return pd.DataFrame()

    cluster_summary = (
        repeated.groupby(["cluster_id", "issue_category"])
        .agg(
            count=("comment_id", "count"),
            example=("issue_text", "first"),
        )
        .reset_index()
        .sort_values(["count", "cluster_id"], ascending=[False, True])
    )
    return cluster_summary


def main():
    documents, review_events, comments = load_data()
    comments = build_comment_features(comments)
    comments = assign_similarity_clusters(comments, threshold=0.35)

    st.title("Smart Feedback & Review Dashboard")
    st.caption(
        "AI-assisted prototype for exploring review friction, feedback loops, and revision burden in regulated document workflows."
    )

    st.sidebar.header("Filters")

    doc_type_options = ["All"] + sorted(documents["document_type"].dropna().unique().tolist())
    team_options = ["All"] + sorted(documents["team"].dropna().unique().tolist())
    status_options = ["All"] + sorted(documents["current_status"].dropna().unique().tolist())
    reviewer_options = ["All"] + sorted(comments["reviewer"].dropna().unique().tolist())

    selected_doc_type = st.sidebar.selectbox("Document type", doc_type_options)
    selected_team = st.sidebar.selectbox("Team", team_options)
    selected_status = st.sidebar.selectbox("Review status", status_options)
    selected_reviewer = st.sidebar.selectbox("Reviewer", reviewer_options)

    filtered_documents = documents.copy()

    if selected_doc_type != "All":
        filtered_documents = filtered_documents[filtered_documents["document_type"] == selected_doc_type]

    if selected_team != "All":
        filtered_documents = filtered_documents[filtered_documents["team"] == selected_team]

    if selected_status != "All":
        filtered_documents = filtered_documents[filtered_documents["current_status"] == selected_status]

    if selected_reviewer != "All":
        reviewer_doc_ids = comments[comments["reviewer"] == selected_reviewer]["doc_id"].unique()
        filtered_documents = filtered_documents[filtered_documents["doc_id"].isin(reviewer_doc_ids)]

    filtered_doc_ids = filtered_documents["doc_id"].unique()
    filtered_events = review_events[review_events["doc_id"].isin(filtered_doc_ids)].copy()
    filtered_comments = comments[comments["doc_id"].isin(filtered_doc_ids)].copy()

    total_docs, active_comments, avg_rounds, delayed_items, recurrent_clusters = make_summary_metrics(
        filtered_documents, filtered_events, filtered_comments
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents under review", total_docs)
    c2.metric("Active comments", active_comments)
    c3.metric("Avg. revision rounds", avg_rounds)
    c4.metric("Delayed items", delayed_items)
    c5.metric("Repeated issue clusters", recurrent_clusters)

    st.markdown("---")

    overview_df = build_document_overview(filtered_documents, filtered_events, filtered_comments)

    st.subheader("Document Overview")
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    if overview_df.empty:
        st.warning("No documents match the current filters.")
        return

    selected_doc_id = st.selectbox(
        "Select a document to inspect",
        overview_df["doc_id"].tolist(),
        format_func=lambda x: f"{x} — {overview_df.loc[overview_df['doc_id'] == x, 'title'].iloc[0]}",
    )

    selected_doc = filtered_documents[filtered_documents["doc_id"] == selected_doc_id].iloc[0]
    selected_events = filtered_events[filtered_events["doc_id"] == selected_doc_id].copy()
    selected_comments = filtered_comments[filtered_comments["doc_id"] == selected_doc_id].copy()

    burden_score, burden_label = compute_revision_burden(
        selected_doc_id, filtered_events, filtered_comments
    )

    st.markdown("---")
    st.subheader("Selected Document Detail")

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("#### Document Summary")
        st.write(f"Document ID: {selected_doc['doc_id']}")
        st.write(f"Title: {selected_doc['title']}")
        st.write(f"Type: {selected_doc['document_type']}")
        st.write(f"Owner: {selected_doc['owner']}")
        st.write(f"Team: {selected_doc['team']}")
        st.write(f"Current status: {selected_doc['current_status']}")
        st.write(f"Final outcome: {selected_doc['final_outcome']}")
        st.write(f"Revision burden score: {burden_score} ({burden_label})")

        st.markdown("#### AI-like Pattern Summary")
        st.info(
            generate_pattern_summary(
                selected_doc,
                selected_comments,
                selected_events,
                burden_score,
                burden_label,
            )
        )

    with right:
        st.markdown("#### Repeated Issue Clusters")
        cluster_examples = find_cluster_examples(selected_comments)

        if cluster_examples.empty:
            st.info("No repeated issue clusters detected for this document.")
        else:
            fig_cluster = px.bar(
                cluster_examples,
                x="issue_category",
                y="count",
                hover_data=["cluster_id", "example"],
                title="Auto-detected repeated issue clusters",
            )
            fig_cluster.update_layout(
                xaxis_title="Issue category",
                yaxis_title="Cluster size",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feedback Flow Timeline")
        if selected_events.empty:
            st.info("No review events available.")
        else:
            timeline_df = (
                selected_events.groupby(["round_no", "event_type"], as_index=False)["duration_days"]
                .sum()
                .sort_values(["round_no", "event_type"])
            )
            fig_timeline = px.bar(
                timeline_df,
                x="round_no",
                y="duration_days",
                color="event_type",
                barmode="group",
                title="Review activity by round",
            )
            fig_timeline.update_layout(
                xaxis_title="Review round",
                yaxis_title="Duration (days)",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        st.markdown("#### Bottleneck by Review Stage")
        if selected_events.empty:
            st.info("No review events available.")
        else:
            stage_df = (
                selected_events.groupby("event_type", as_index=False)["duration_days"]
                .sum()
                .sort_values("duration_days", ascending=False)
            )
            fig_stage = px.bar(
                stage_df,
                x="event_type",
                y="duration_days",
                title="Time spent by stage",
            )
            fig_stage.update_layout(
                xaxis_title="Stage",
                yaxis_title="Total duration (days)",
            )
            st.plotly_chart(fig_stage, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Reviewer Pattern Comparison")
    if selected_comments.empty:
        st.info("No reviewer data available for this document.")
    else:
        reviewer_df = (
            selected_comments.groupby(["reviewer", "issue_category"])
            .size()
            .reset_index(name="count")
            .sort_values(["reviewer", "count"], ascending=[True, False])
        )
        fig_reviewer = px.bar(
            reviewer_df,
            x="reviewer",
            y="count",
            color="issue_category",
            barmode="stack",
            title="Issue distribution by reviewer",
        )
        fig_reviewer.update_layout(
            xaxis_title="Reviewer",
            yaxis_title="Comment count",
        )
        st.plotly_chart(fig_reviewer, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Comment Threads with Auto Pattern Flags")
    if selected_comments.empty:
        st.info("No comments available for this document.")
    else:
        comment_display = selected_comments[
            [
                "round_no",
                "reviewer",
                "issue_category",
                "severity",
                "resolved",
                "cluster_id",
                "cluster_size",
                "auto_repeated_flag",
                "issue_text",
            ]
        ].sort_values(["round_no", "reviewer", "cluster_id"])
        st.dataframe(comment_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### Global Repeated Cluster Explorer")
    global_clusters = find_cluster_examples(filtered_comments)

    if global_clusters.empty:
        st.info("No repeated clusters detected in the filtered dataset.")
    else:
        st.dataframe(
            global_clusters.rename(
                columns={
                    "cluster_id": "Cluster ID",
                    "issue_category": "Issue Category",
                    "count": "Cluster Size",
                    "example": "Example Comment",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
