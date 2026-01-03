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
html, body, [class*="css"] { font-size: 14px; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ===============================
# CENTER LAYOUT
# ===============================
_, main_col, _ = st.columns([1, 4, 1])

with main_col:

    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    # COLORS
    POS = "#f97316"
    NEG = "#2563eb"
    COLORS = [POS, NEG]

    # ===============================
    # LOAD MODEL
    # ===============================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    model = load_model()

    # ===============================
    # YOUTUBE API
    # ===============================
    youtube = build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])

    # ===============================
    # HELPERS
    # ===============================
    def predict_sentiment(text):
        try:
            label = model(text[:512])[0]["label"]
            return "Positive" if label == "POSITIVE" else "Negative"
        except:
            return "Negative"

    def fetch_comments(video_id, limit=40):
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=limit
            ).execute()
            return [
                i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for i in res["items"]
            ]
        except:
            return []

    # ===============================
    # SENTIMENT CHARTS
    # ===============================
    def sentiment_charts(sentiments):
        s = pd.Series(sentiments).value_counts()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(s, labels=s.index, autopct="%1.1f%%", colors=COLORS, startangle=90)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(3.6, 3))
            s.plot(kind="bar", color=COLORS, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3 = st.tabs(
        ["📦 Product / Topic (YouTube)", "📺 Channel Insights", "📁 CSV Upload"]
    )

    # ======================================================
    # CHANNEL INSIGHTS (UPDATED)
    # ======================================================
    with tab2:
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):

            # ---- Find channel ----
            search = youtube.search().list(
                q=channel, part="snippet", type="channel", maxResults=1
            ).execute()

            if not search["items"]:
                st.error("Channel not found")
            else:
                channel_id = search["items"][0]["snippet"]["channelId"]

                # ---- Channel details ----
                channel_data = youtube.channels().list(
                    part="snippet,statistics",
                    id=channel_id
                ).execute()["items"][0]

                channel_name = channel_data["snippet"]["title"]
                subscribers = int(channel_data["statistics"].get("subscriberCount", 0))

                st.subheader(f"📺 Channel Name: **{channel_name}**")
                st.caption(f"👥 Subscribers: **{subscribers:,}**")

                # ---- Fetch videos ----
                search_videos = youtube.search().list(
                    channelId=channel_id,
                    part="id",
                    type="video",
                    maxResults=25
                ).execute()["items"]

                video_ids = [v["id"]["videoId"] for v in search_videos]

                video_data = youtube.videos().list(
                    part="snippet,statistics",
                    id=",".join(video_ids)
                ).execute()["items"]

                rows = []
                comments = []
                total_likes = 0

                for item in video_data:
                    title = item["snippet"]["title"]
                    views = int(item["statistics"].get("viewCount", 0))
                    likes = int(item["statistics"].get("likeCount", 0))

                    total_likes += likes
                    rows.append({"Title": title, "Views": views})

                    comments.extend(fetch_comments(item["id"], 30))

                df_videos = pd.DataFrame(rows).sort_values(
                    by="Views", ascending=False
                )

                # ---- METRICS ----
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Videos", len(df_videos))
                m2.metric("Comments", len(comments))
                m3.metric("Total Views", f"{df_videos['Views'].sum():,}")
                m4.metric("Total Likes", f"{total_likes:,}")

                # ---- BAR CHART (VIDEO COMPARISON) ----
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(df_videos["Title"], df_videos["Views"], color=POS)
                ax.invert_yaxis()
                ax.set_xlabel("Views")
                ax.set_title("Views per Video (Recent)")
                st.pyplot(fig)

                # ---- SENTIMENT ----
                sentiments = [predict_sentiment(c) for c in comments]
                sentiment_charts(sentiments)

                # ---- VIDEO LIST ----
                st.subheader("🎬 Video Titles & Views")
                st.dataframe(
                    df_videos,
                    use_container_width=True,
                    height=300
                )
