import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from googleapiclient.discovery import build
from langdetect import detect

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# CENTER LAYOUT
# ===============================
_, main_col, _ = st.columns([1, 4, 1])

with main_col:

    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    # ===============================
    # COLORS (3 class)
    # ===============================
    NEG_COLOR = "#ef4444"   # Red
    NEU_COLOR = "#2563eb"   # Blue
    POS_COLOR = "#f97316"   # Orange
    COLORS = [NEG_COLOR, NEU_COLOR, POS_COLOR]

    # ===============================
    # LOAD MODELS (3 CLASS)
    # ===============================
    @st.cache_resource
    def load_models():
        # English (3-class)
        sentiment_en = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

        # Tamil/Hindi (3-class multilingual)
        sentiment_multi = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        return sentiment_en, sentiment_multi

    sentiment_en, sentiment_multi = load_models()

    # ===============================
    # YOUTUBE API
    # ===============================
    youtube = build(
        "youtube",
        "v3",
        developerKey=st.secrets["YOUTUBE_API_KEY"]
    )

    # ===============================
    # LABEL MAPS (SAFE)
    # ===============================
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",

        # in case model returns string labels
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
    }

    def detect_language(text: str) -> str:
        try:
            return detect(text)
        except:
            return "en"

    # ===============================
    # FAST BATCH PREDICTION (CPU SAFE)
    # ===============================
    def predict_sentiments_batch(texts, batch_size=16):
        """
        Returns:
        sentiments: list[str]
        confidences: list[float]
        """
        sentiments, confidences = [], []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # detect language quickly for each text
            langs = []
            for t in batch:
                langs.append(detect_language(t))

            # split batch into english vs multi
            en_texts = [t[:512] for t, lg in zip(batch, langs) if lg == "en"]
            multi_texts = [t[:512] for t, lg in zip(batch, langs) if lg != "en"]

            en_out = []
            multi_out = []

            if en_texts:
                en_out = sentiment_en(
                    en_texts,
                    truncation=True,
                    max_length=512
                )

            if multi_texts:
                multi_out = sentiment_multi(
                    multi_texts,
                    truncation=True,
                    max_length=512
                )

            # merge outputs back in same order
            en_idx = 0
            multi_idx = 0

            for lg in langs:
                if lg == "en":
                    out = en_out[en_idx]
                    en_idx += 1
                else:
                    out = multi_out[multi_idx]
                    multi_idx += 1

                lab = out.get("label", "")
                score = float(out.get("score", 0.0))

                lab = label_map.get(lab, label_map.get(str(lab).lower(), "Neutral"))

                sentiments.append(lab)
                confidences.append(score)

        return sentiments, confidences

    # ===============================
    # ASPECTS (PRODUCT ONLY)
    # ===============================
    PRODUCT_ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "performance", "build"],
        "Camera": ["camera", "photo", "video"],
        "Battery": ["battery", "charge", "backup"]
    }

    def aspect_based_sentiment(texts, sentiments):
        """
        Uses already predicted sentiments (FAST)
        """
        rows = []
        for t, s in zip(texts, sentiments):
            tl = str(t).lower()
            for asp, keys in PRODUCT_ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({"Aspect": asp, "Sentiment": s})
        return pd.DataFrame(rows)

    # ===============================
    # YOUTUBE FUNCTIONS
    # ===============================
    def search_videos(query, limit=10):
        res = youtube.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res.get("items", [])]

    def fetch_comments(video_id, limit=100):
        try:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100
            ).execute()

            comments = []
            for item in res.get("items", [])[:limit]:
                comments.append(
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                )
            return comments
        except:
            return []

    # ===============================
    # CHARTS
    # ===============================
    def show_sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()

        # make sure order is fixed
        order = ["Negative", "Neutral", "Positive"]
        s = s.reindex(order).fillna(0)

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(
                s,
                labels=s.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=COLORS
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(3.6, 3))
            s.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # CONFIDENCE DISPLAY
    # ===============================
    def show_confidence_summary(confidences):
        avg_conf = sum(confidences) / max(len(confidences), 1)
        st.info(f"✅ Average Confidence Score: **{avg_conf:.2f}**")

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3 = st.tabs([
        "📦 Product / Topic (YouTube)",
        "📺 Channel Insights",
        "📁 CSV Upload"
    ])

    # ===============================
    # TAB 1: PRODUCT / TOPIC
    # ===============================
    with tab1:
        analysis_type = st.radio(
            "What are you analyzing?",
            ["Product", "General Topic (Song / Movie / News)"]
        )

        topic = st.text_input("Enter product / topic")

        if st.button("Analyze Topic"):
            if not topic.strip():
                st.error("Please enter a product/topic")
            else:
                st.info(f"🔍 Analyzing public opinion on: {topic}")

                comments = []
                video_ids = search_videos(topic, limit=10)

                for vid in video_ids:
                    comments.extend(fetch_comments(vid, limit=100))

                st.success(f"✅ Videos fetched: {len(video_ids)} | ✅ Comments fetched: {len(comments)}")

                if len(comments) == 0:
                    st.warning("No comments found. Try another keyword.")
                else:
                    sentiments, confidences = predict_sentiments_batch(comments, batch_size=16)
                    show_confidence_summary(confidences)
                    show_sentiment_charts(sentiments)

                    st.subheader("📄 Sample Comments")
                    for i, c in enumerate(comments[:5], 1):
                        st.write(f"{i}. {c}")

                    if analysis_type == "Product":
                        st.subheader("🧠 Aspect-Based Sentiment (Bar)")
                        absa = aspect_based_sentiment(comments, sentiments)
                        if not absa.empty:
                            st.bar_chart(absa.value_counts().unstack().fillna(0))
                        else:
                            st.info("No aspect keywords matched in comments.")
                    else:
                        st.info("Aspect-based sentiment not applicable for general topics.")

    # ===============================
    # TAB 2: CHANNEL INSIGHTS
    # ===============================
    with tab2:
        channel_name = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            if not channel_name.strip():
                st.error("Please enter a channel name")
            else:
                search = youtube.search().list(
                    q=channel_name,
                    part="snippet",
                    type="channel",
                    maxResults=1
                ).execute()

                if not search.get("items"):
                    st.error("Channel not found")
                else:
                    cid = search["items"][0]["snippet"]["channelId"]

                    channel_data = youtube.channels().list(
                        part="snippet,statistics",
                        id=cid
                    ).execute()["items"][0]

                    channel_title = channel_data["snippet"]["title"]
                    subs = int(channel_data["statistics"].get("subscriberCount", 0))
                    total_views = int(channel_data["statistics"].get("viewCount", 0))
                    total_videos = int(channel_data["statistics"].get("videoCount", 0))

                    st.subheader(f"📺 Channel Name: {channel_title}")
                    st.write(f"**Subscribers:** {subs:,}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Views", f"{total_views:,}")
                    c2.metric("Total Videos", f"{total_videos:,}")
                    c3.metric("Recent Videos Analyzed", "25")

                    videos = youtube.search().list(
                        channelId=cid,
                        part="id",
                        type="video",
                        maxResults=25
                    ).execute().get("items", [])

                    video_rows = []
                    all_comments = []
                    total_likes = 0

                    for v in videos:
                        vid = v["id"]["videoId"]
                        data = youtube.videos().list(
                            part="snippet,statistics",
                            id=vid
                        ).execute()["items"][0]

                        title = data["snippet"]["title"]
                        views = int(data["statistics"].get("viewCount", 0))
                        likes = int(data["statistics"].get("likeCount", 0))
                        total_likes += likes

                        video_rows.append({"Title": title, "Views": views, "Likes": likes})

                        all_comments.extend(fetch_comments(vid, limit=40))

                    df_vid = pd.DataFrame(video_rows).sort_values("Views", ascending=False)

                    st.write(f"✅ **Videos:** {len(videos)}")
                    st.write(f"✅ **Total Comments:** {len(all_comments)}")
                    st.write(f"✅ **Total Views (Recent Videos):** {df_vid['Views'].sum():,}")
                    st.write(f"✅ **Total Likes (Recent Videos):** {total_likes:,}")

                    st.subheader("📊 Views per Video (Recent)")
                    st.bar_chart(df_vid.set_index("Title")["Views"])

                    st.subheader("🎬 Video Titles + Views (Scrollable)")
                    st.dataframe(df_vid, use_container_width=True, height=320)

                    if len(all_comments) > 0:
                        sentiments, confidences = predict_sentiments_batch(all_comments, batch_size=16)
                        show_confidence_summary(confidences)
                        show_sentiment_charts(sentiments)
                    else:
                        st.warning("No comments found for recent videos.")

    # ===============================
    # TAB 3: CSV UPLOAD
    # ===============================
    with tab3:
        file = st.file_uploader("Upload CSV", type="csv")

        if file:
            if st.button("Analyze Dataset"):
                df = pd.read_csv(file, encoding_errors="ignore")

                df.columns = (
                    df.columns
                    .str.lower()
                    .str.strip()
                    .str.replace("\ufeff", "")
                )

                st.success(f"CSV loaded: {len(df)} rows")
                st.write("Detected columns:", list(df.columns))

                TEXT_COLS = [
                    "text", "tweet", "comment",
                    "review", "content", "sentence"
                ]

                text_col = next((c for c in TEXT_COLS if c in df.columns), None)

                if not text_col:
                    st.error("❌ No text column detected (text/tweet/comment/review/content/sentence).")
                else:
                    texts = df[text_col].astype(str).head(1000).tolist()

                    sentiments, confidences = predict_sentiments_batch(texts, batch_size=16)
                    show_confidence_summary(confidences)
                    show_sentiment_charts(sentiments)

                    st.subheader("📄 Sample Rows")
                    st.dataframe(df[[text_col]].head(5), use_container_width=True)
