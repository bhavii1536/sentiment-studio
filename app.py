import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from googleapiclient.discovery import build

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Sentiment Analysis Studio", layout="wide")

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
    # COLORS (3-class)
    # ===============================
    NEG_COLOR = "#2563eb"   # Blue
    NEU_COLOR = "#a855f7"   # Purple
    POS_COLOR = "#f97316"   # Orange
    COLORS = [NEG_COLOR, NEU_COLOR, POS_COLOR]

    # ===============================
    # LOAD ROBERTA MODEL (3-class)
    # ===============================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=512
        )

    model = load_model()

    LABEL_MAP = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    # ===============================
    # YOUTUBE API
    # ===============================
    youtube = build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])

    # ===============================
    # SENTIMENT PREDICTION (FAST)
    # ===============================
    def predict_sentiments(texts, batch_size=16):
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            outputs = model(batch)
            for out in outputs:
                sentiments.append(LABEL_MAP.get(out["label"], "Neutral"))
        return sentiments

    # ===============================
    # PRODUCT ASPECTS
    # ===============================
    PRODUCT_ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "performance", "build"],
        "Camera": ["camera", "photo", "video"],
        "Battery": ["battery", "charge", "backup"]
    }

    def aspect_based_sentiment(texts):
        rows = []
        for t in texts:
            tl = t.lower()
            for asp, keys in PRODUCT_ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({"Aspect": asp, "Text": t})
        return pd.DataFrame(rows)

    # ===============================
    # YOUTUBE FUNCTIONS (LIMITED)
    # ===============================
    def search_videos(query, limit=10):
        res = youtube.search().list(
            q=query, part="id", type="video", maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=80):
        try:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=limit
            ).execute()
            return [
                i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for i in res["items"]
            ]
        except:
            return []

    # ===============================
    # CHARTS
    # ===============================
    def show_sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()
        s = s.reindex(["Negative", "Neutral", "Positive"]).fillna(0)

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(s, labels=s.index, autopct="%1.1f%%",
                   startangle=90, colors=COLORS)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(3.6, 3))
            ax.bar(s.index, s.values, color=COLORS)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3 = st.tabs([
        "📦 Product / Topic (YouTube)",
        "📺 Channel Insights",
        "📁 CSV Upload"
    ])

    # ===============================
    # TAB 1
    # ===============================
    with tab1:
        analysis_type = st.radio(
            "What are you analyzing?",
            ["Product", "General Topic (Song / Movie / News)"]
        )

        topic = st.text_input("Enter product / topic")

        if st.button("Analyze Topic"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")

            comments = []
            for vid in search_videos(topic, limit=10):
                comments.extend(fetch_comments(vid, limit=80))

            comments = comments[:800]
            st.success(f"Fetched {len(comments)} comments")

            sentiments = predict_sentiments(comments)
            show_sentiment_charts(sentiments)

            st.subheader("📄 Sample Comments")
            for i, c in enumerate(comments[:5], 1):
                st.write(f"{i}. {c}")

            if analysis_type == "Product":
                st.subheader("🧠 Aspect-Based Sentiment")
                absa = aspect_based_sentiment(comments)
                if not absa.empty:
                    absa_sent = predict_sentiments(absa["Text"].tolist())
                    absa["Sentiment"] = absa_sent
                    st.bar_chart(absa.groupby(["Aspect", "Sentiment"]).size().unstack().fillna(0))

    # ===============================
    # TAB 2
    # ===============================
    with tab2:
        channel_name = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            search = youtube.search().list(
                q=channel_name, part="snippet", type="channel", maxResults=1
            ).execute()

            if not search["items"]:
                st.error("Channel not found")
            else:
                cid = search["items"][0]["snippet"]["channelId"]

                channel_data = youtube.channels().list(
                    part="snippet,statistics", id=cid
                ).execute()["items"][0]

                st.subheader(f"📺 Channel: {channel_data['snippet']['title']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Subscribers", channel_data["statistics"]["subscriberCount"])
                c2.metric("Total Views", channel_data["statistics"]["viewCount"])
                c3.metric("Total Videos", channel_data["statistics"]["videoCount"])

                videos = youtube.search().list(
                    channelId=cid, part="id", type="video", maxResults=25
                ).execute()["items"]

                rows, comments = [], []

                for v in videos:
                    vid = v["id"]["videoId"]
                    data = youtube.videos().list(
                        part="snippet,statistics", id=vid
                    ).execute()["items"][0]

                    rows.append({
                        "Title": data["snippet"]["title"],
                        "Views": int(data["statistics"].get("viewCount", 0))
                    })

                    comments.extend(fetch_comments(vid, limit=40))

                df = pd.DataFrame(rows).sort_values("Views", ascending=False)
                st.bar_chart(df.set_index("Title")["Views"])
                st.dataframe(df, height=300)

                sentiments = predict_sentiments(comments)
                show_sentiment_charts(sentiments)

    # ===============================
    # TAB 3
    # ===============================
    with tab3:
        file = st.file_uploader("Upload CSV", type="csv")

        if file and st.button("Analyze Dataset"):
            df = pd.read_csv(file, encoding_errors="ignore")
            df.columns = df.columns.str.lower().str.strip()

            TEXT_COLS = ["text", "tweet", "comment", "review", "content", "sentence"]
            text_col = next((c for c in TEXT_COLS if c in df.columns), None)

            if not text_col:
                st.error("No text column found")
            else:
                texts = df[text_col].astype(str).head(1000).tolist()
                sentiments = predict_sentiments(texts)
                show_sentiment_charts(sentiments)
