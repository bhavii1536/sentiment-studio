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

.block-container {
    padding-top: 2rem;
}

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
_, main_col, _ = st.columns([1, 4, 1])

with main_col:
    st.title("📊 Sentiment Analysis Studio")
    st.caption("Real-Time Media Opinion Analysis Using Machine Learning")

    PIE_COLORS = ["#2563EB", "#F97316"]
    ASPECT_COLORS = ["#22C55E", "#EF4444"]

    # ===============================
    # LOAD MODELS
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
        "youtube", "v3",
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
            q=query, part="id", type="video", maxResults=limit
        ).execute()
        return [i["id"]["videoId"] for i in res["items"]]

    def fetch_comments(video_id, limit=150):
        res = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100
        ).execute()
        return [
            i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for i in res["items"][:limit]
        ]

    def fetch_channel_data(channel_id, video_limit=8, comment_limit=150):
        search_res = youtube.search().list(
            channelId=channel_id, part="id", type="video", maxResults=video_limit
        ).execute()

        comments, video_ids = [], []

        for v in search_res["items"]:
            vid = v["id"]["videoId"]
            video_ids.append(vid)
            comments.extend(fetch_comments(vid, comment_limit))

        stats_res = youtube.videos().list(
            part="statistics,snippet", id=",".join(video_ids)
        ).execute()

        total_views, total_likes, monthly_views = 0, 0, {}

        for item in stats_res["items"]:
            views = int(item["statistics"].get("viewCount", 0))
            likes = int(item["statistics"].get("likeCount", 0))
            month = item["snippet"]["publishedAt"][:7]

            total_views += views
            total_likes += likes
            monthly_views[month] = monthly_views.get(month, 0) + views

        return {
            "video_count": len(video_ids),
            "comment_count": len(comments),
            "total_views": total_views,
            "total_likes": total_likes,
            "monthly_views": monthly_views,
            "comments": comments
        }

    def last_n_months_views(monthly_views, n=6):
        today = datetime.today()
        months = [(today - relativedelta(months=i)).strftime("%Y-%m") for i in range(n-1, -1, -1)]
        return pd.DataFrame({
            "Month": [datetime.strptime(m, "%Y-%m").strftime("%b %Y") for m in months],
            "Views": [monthly_views.get(m, 0) for m in months]
        })

    # ===============================
    # PIE + BAR (PRODUCT ANALYSIS)
    # ===============================
    def show_sentiment_pie_and_bar(sentiments):
        s = pd.Series(sentiments).value_counts().reindex(["Negative", "Positive"], fill_value=0)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                s,
                labels=s.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=PIE_COLORS,
                textprops={"color": "white", "fontsize": 10}
            )
            ax.axis("equal")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            s.plot(kind="barh", color=PIE_COLORS, ax=ax)
            ax.set_xlabel("Comments")
            ax.set_ylabel("")
            ax.set_title("Sentiment Comparison")
            st.pyplot(fig)

    # ===============================
    # PIE ONLY (CHANNEL / CSV)
    # ===============================
    def show_sentiment_pie(sentiments):
        s = pd.Series(sentiments).value_counts().reindex(["Negative", "Positive"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            s,
            labels=s.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=PIE_COLORS,
            textprops={"color": "white"}
        )
        ax.axis("equal")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    # ===============================
    # UI MODE
    # ===============================
    mode = st.selectbox(
        "Choose Analysis Type",
        [
            "Product / Topic Analysis (YouTube)",
            "YouTube Channel Insights",
            "CSV Upload Analysis"
        ]
    )

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
                show_sentiment_pie_and_bar(sentiments)

                absa = aspect_based_sentiment(comments)
                if not absa.empty:
                    st.subheader("🧠 Aspect-Based Sentiment")
                    aspect_df = (
                        absa.groupby(["Aspect", "Sentiment"])
                        .size()
                        .unstack()
                        .reindex(columns=["Positive", "Negative"], fill_value=0)
                    )
                    st.bar_chart(aspect_df)

                st.subheader("📄 Sample Comments")
                for i, c in enumerate(comments[:5], 1):
                    st.write(f"{i}. {c}")

    # ===============================
    # CHANNEL INSIGHTS
    # ===============================
    elif mode == "YouTube Channel Insights":
        channel = st.text_input("Enter Channel Name")

        if st.button("Analyze Channel"):
            res = youtube.search().list(
                q=channel, part="snippet", type="channel", maxResults=1
            ).execute()

            if res["items"]:
                cid = res["items"][0]["snippet"]["channelId"]
                data = fetch_channel_data(cid)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Videos", data["video_count"])
                m2.metric("Comments", data["comment_count"])
                m3.metric("Total Views", f'{data["total_views"]:,}')
                m4.metric("Total Likes", f'{data["total_likes"]:,}')

                st.subheader("📈 Monthly Views (Last 6 Months)")
                mv_df = last_n_months_views(data["monthly_views"])
                st.bar_chart(mv_df.set_index("Month"))

                show_sentiment_pie([predict_sentiment(c) for c in data["comments"]])

                st.subheader("📄 Sample Audience Comments")
                for i, c in enumerate(data["comments"][:5], 1):
                    st.write(f"{i}. {c}")

    # ===============================
    # CSV UPLOAD
    # ===============================
    else:
        file = st.file_uploader("Upload CSV (text column required)", type="csv")
        if file:
            df = pd.read_csv(file)
            if "text" in df.columns:
                show_sentiment_pie(df["text"].apply(predict_sentiment))
            else:
                st.error("CSV must contain a 'text' column")
