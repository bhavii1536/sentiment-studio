import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

# ===============================
# BASIC STYLE
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

    # Colors
    POS_COLOR = "#ef4444"   # Red
    NEG_COLOR = "#2563eb"   # Blue
    COLORS = [POS_COLOR, NEG_COLOR]

    # ===============================
    # LOAD MODELS (FAST + CPU SAFE)
    # ===============================
    @st.cache_resource
    def load_models():
        en = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        multi = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        return en, multi

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
    # HELPERS
    # ===============================
    def detect_language(text):
        try:
            return detect(text)
        except:
            return "en"

    def predict_sentiment(text):
        model = sentiment_en if detect_language(text) == "en" else sentiment_multi
        label = model(text[:512])[0]["label"]
        return "Positive" if label in ["LABEL_1", "POSITIVE"] else "Negative"

    # ===============================
    # CHARTS
    # ===============================
    def show_sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()

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
            fig, ax = plt.subplots(figsize=(3.5, 3))
            s.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # MODE TABS
    # ===============================
    mode = st.selectbox(
        "Choose Analysis Type",
        [
            "Product / Topic (YouTube)",
            "Channel Insights (YouTube)",
            "CSV Upload Analysis"
        ]
    )

    # =====================================================
    # CHANNEL INSIGHTS
    # =====================================================
    if mode == "Channel Insights (YouTube)":

        channel_name = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):

            # ---- Find channel ----
            search = youtube.search().list(
                q=channel_name,
                part="snippet",
                type="channel",
                maxResults=1
            ).execute()

            if not search["items"]:
                st.error("Channel not found")
            else:
                channel_id = search["items"][0]["snippet"]["channelId"]

                # ---- Channel stats ----
                channel_data = youtube.channels().list(
                    part="statistics,snippet",
                    id=channel_id
                ).execute()["items"][0]

                subscribers = int(channel_data["statistics"].get("subscriberCount", 0))
                official_name = channel_data["snippet"]["title"]

                st.subheader(f"📺 Channel Name: **{official_name}**")
                st.caption(f"👥 Subscribers: **{subscribers:,}**")

                # ---- Fetch videos ----
                videos = youtube.search().list(
                    channelId=channel_id,
                    part="id",
                    type="video",
                    maxResults=25
                ).execute()["items"]

                video_ids = [v["id"]["videoId"] for v in videos]

                video_details = youtube.videos().list(
                    part="snippet,statistics",
                    id=",".join(video_ids)
                ).execute()["items"]

                total_views = 0
                total_likes = 0
                comments = []
                video_rows = []

                for v in video_details:
                    stats = v["statistics"]
                    snippet = v["snippet"]

                    views = int(stats.get("viewCount", 0))
                    likes = int(stats.get("likeCount", 0))

                    total_views += views
                    total_likes += likes

                    video_rows.append({
                        "Title": snippet["title"],
                        "Views": views
                    })

                    # Fetch limited comments per video
                    try:
                        c = youtube.commentThreads().list(
                            part="snippet",
                            videoId=v["id"],
                            maxResults=30
                        ).execute()

                        comments.extend([
                            i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                            for i in c["items"]
                        ])
                    except:
                        pass

                # ---- Metrics ----
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Videos", len(video_rows))
                m2.metric("Comments", len(comments))
                m3.metric("Total Views", f"{total_views:,}")
                m4.metric("Total Likes", f"{total_likes:,}")

                # ---- Views per video chart ----
                df_videos = pd.DataFrame(video_rows).sort_values(
                    by="Views", ascending=False
                )

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(df_videos)), df_videos["Views"], color=POS_COLOR)
                ax.set_title("Views per Video (Recent)")
                ax.set_ylabel("Views")
                ax.set_xlabel("Videos (ordered)")
                st.pyplot(fig)

                # ---- Sentiment ----
                sentiments = [predict_sentiment(c) for c in comments]
                show_sentiment_charts(sentiments)

                # ---- Video titles list ----
                st.subheader("🎬 Recent Videos (Top Views)")
                st.dataframe(
                    df_videos,
                    use_container_width=True,
                    height=300
                )

    # =====================================================
    # PRODUCT / TOPIC & CSV
    # (kept unchanged from your working version)
    # =====================================================
