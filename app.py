import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from transformers import pipeline
from googleapiclient.discovery import build
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide"
)

# ===============================
# BACKGROUND + EMOJI ATMOSPHERE
# ===============================
st.markdown("""
<style>
html, body {
    background: linear-gradient(135deg, #020617, #0f172a);
    font-size: 14px;
}
.block-container { padding-top: 2rem; }

.emoji-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-16px); }
    100% { transform: translateY(0px); }
}

.emoji {
    position: absolute;
    opacity: 0.18;
    animation: float 20s ease-in-out infinite;
}

.small { font-size: 64px; filter: blur(1.2px); }
.medium { font-size: 96px; filter: blur(0.7px); }
.large { font-size: 128px; filter: blur(0.4px); }
</style>

<div class="emoji-bg">
    <span class="emoji large"  style="top:8%; left:6%;">😊</span>
    <span class="emoji medium" style="top:18%; left:82%;">😍</span>
    <span class="emoji small"  style="top:42%; left:10%;">😐</span>
    <span class="emoji large"  style="top:55%; left:88%;">😮</span>
    <span class="emoji medium" style="top:72%; left:6%;">😡</span>
    <span class="emoji small"  style="top:85%; left:74%;">🤔</span>
</div>
""", unsafe_allow_html=True)

# ===============================
# CENTER LAYOUT
# ===============================
_, main, _ = st.columns([1, 4, 1])

with main:
    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    PIE_COLORS = ["#2563EB", "#F97316"]       # Blue / Orange
    ASPECT_COLORS = ["#22C55E", "#EF4444"]    # Green / Red

    # ===============================
    # LOAD MODELS
    # ===============================
    @st.cache_resource
    def load_models():
        en = pipeline("sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english")
        multi = pipeline("sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
        return en, multi

    sentiment_en, sentiment_multi = load_models()

    # ===============================
    # YOUTUBE API
    # ===============================
    youtube = build("youtube", "v3",
        developerKey=st.secrets["YOUTUBE_API_KEY"])

    # ===============================
    # HELPERS
    # ===============================
    def detect_language(text):
        try: return detect(text)
        except: return "en"

    def predict_sentiment(text):
        model = sentiment_en if detect_language(text) == "en" else sentiment_multi
        label = model(text[:512])[0]["label"]
        return "Negative" if label in ["LABEL_0", "NEGATIVE"] else "Positive"

    ASPECTS = {
        "Price": ["price", "cost", "expensive", "cheap"],
        "Quality": ["quality", "pure", "organic"],
        "Packaging": ["packaging", "bottle", "box"],
        "Delivery": ["delivery", "shipping"]
    }

    def aspect_based_sentiment(texts):
        rows = []
        for t in texts:
            tl = t.lower()
            for a, keys in ASPECTS.items():
                if any(k in tl for k in keys):
                    rows.append({"Aspect": a, "Sentiment": predict_sentiment(t)})
        return pd.DataFrame(rows)

    # ===============================
    # YOUTUBE DATA
    # ===============================
    def search_videos(query, limit=8):
        res = youtube.search().list(
            q=query, part="id", type="video", maxResults=limit).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=150):
        res = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100).execute()
        return [i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for i in res["items"][:limit]]

    def fetch_channel_data(channel_id, video_limit=8, comment_limit=150):
        search = youtube.search().list(
            channelId=channel_id, part="id", type="video",
            maxResults=video_limit).execute()

        comments, video_ids = [], []
        for v in search["items"]:
            vid = v["id"]["videoId"]
            video_ids.append(vid)
            comments.extend(fetch_comments(vid, comment_limit))

        stats = youtube.videos().list(
            part="statistics,snippet", id=",".join(video_ids)).execute()

        total_views, total_likes, monthly_views = 0, 0, {}
        for item in stats["items"]:
            views = int(item["statistics"].get("viewCount", 0))
            likes = int(item["statistics"].get("likeCount", 0))
            month = item["snippet"]["publishedAt"][:7]
            total_views += views
            total_likes += likes
            monthly_views[month] = monthly_views.get(month, 0) + views

        return {
            "videos": len(video_ids),
            "comments": comments,
            "comment_count": len(comments),
            "total_views": total_views,
            "total_likes": total_likes,
            "monthly_views": monthly_views
        }

    def last_6_months_views(monthly_views):
        today = datetime.today()
        months = []
        for i in range(5, -1, -1):
            dt = today - relativedelta(months=i)
            months.append((dt.strftime("%Y-%m"), dt.strftime("%b %Y")))

        return pd.DataFrame({
            "Month": [label for _, label in months],
            "Views": [monthly_views.get(key, 0) for key, _ in months]
        })

    # ===============================
    # CHART FUNCTIONS
    # ===============================
    def pie_and_bar(sentiments):
        s = pd.Series(sentiments).value_counts().reindex(
            ["Negative", "Positive"], fill_value=0)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pie(s, labels=s.index, autopct="%1.1f%%",
                   colors=PIE_COLORS, startangle=90,
                   textprops={"color":"white"})
            ax.axis("equal")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4,3))
            s.plot(kind="barh", color=PIE_COLORS, ax=ax)
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    def sentiment_pie(sentiments):
        s = pd.Series(sentiments).value_counts().reindex(
            ["Negative", "Positive"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie(s, labels=s.index, autopct="%1.1f%%",
               colors=PIE_COLORS, startangle=90,
               textprops={"color":"white"})
        ax.axis("equal")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    def aspect_pies(absa):
        cols = st.columns(len(ASPECTS))
        for col, aspect in zip(cols, ASPECTS):
            with col:
                data = absa[absa["Aspect"]==aspect]["Sentiment"] \
                    .value_counts().reindex(["Positive","Negative"], fill_value=0)
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(data, labels=data.index, autopct="%1.1f%%",
                       colors=ASPECT_COLORS, startangle=90,
                       textprops={"color":"white", "fontsize":9})
                ax.axis("equal")
                ax.set_title(aspect)
                st.pyplot(fig)

    # ===============================
    # UI MODE
    # ===============================
    mode = st.selectbox("Choose Analysis Type", [
        "Product / Topic Analysis (YouTube)",
        "YouTube Channel Insights",
        "CSV Upload Analysis"
    ])

    # ===============================
    # PRODUCT / TOPIC
    # ===============================
    if mode == "Product / Topic Analysis (YouTube)":
        topic = st.text_input("Enter product / topic")

        if st.button("Analyze"):
            st.info(f"🔍 Analyzing public opinion on: {topic}")
            comments = []
            vids = search_videos(topic)
            for v in vids:
                comments.extend(fetch_comments(v))

            st.write(f"**Videos analyzed:** {len(vids)}")
            st.write(f"**Comments analyzed:** {len(comments)}")

            if comments:
                sentiments = [predict_sentiment(c) for c in comments]
                pie_and_bar(sentiments)

                absa = aspect_based_sentiment(comments)
                if not absa.empty:
                    st.subheader("🧠 Aspect-Based Sentiment (Bar)")
                    aspect_df = absa.groupby(
                        ["Aspect","Sentiment"]).size().unstack().fillna(0)
                    st.bar_chart(aspect_df)

    # ===============================
    # CHANNEL INSIGHTS
    # ===============================
    elif mode == "YouTube Channel Insights":
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            res = youtube.search().list(
                q=channel, part="snippet", type="channel",
                maxResults=1).execute()

            if res["items"]:
                cid = res["items"][0]["snippet"]["channelId"]
                data = fetch_channel_data(cid)

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Videos", data["videos"])
                m2.metric("Comments", data["comment_count"])
                m3.metric("Total Views", f'{data["total_views"]:,}')
                m4.metric("Total Likes", f'{data["total_likes"]:,}')

                mv_df = last_6_months_views(data["monthly_views"])
                st.subheader("📈 Monthly Views (Last 6 Months)")
                st.bar_chart(mv_df.set_index("Month"))

                sentiments = [predict_sentiment(c) for c in data["comments"]]
                sentiment_pie(sentiments)

                absa = aspect_based_sentiment(data["comments"])
                if not absa.empty:
                    st.subheader("🧠 Aspect-Based Sentiment (Pie)")
                    aspect_pies(absa)

    # ===============================
    # CSV UPLOAD
    # ===============================
    else:
        file = st.file_uploader("Upload CSV (text column required)", type="csv")
        if file:
            df = pd.read_csv(file)
            if "text" in df.columns:
                sentiment_pie(df["text"].apply(predict_sentiment))
            else:
                st.error("CSV must contain a 'text' column")
