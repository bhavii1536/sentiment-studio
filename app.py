import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from googleapiclient.discovery import build
from langdetect import detect, LangDetectException


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
    # COLORS
    # ===============================

    NEG_COLOR = "#ef4444"
    NEU_COLOR = "#2563eb"
    POS_COLOR = "#f97316"

    COLORS = [NEG_COLOR, NEU_COLOR, POS_COLOR]


    # ===============================
    # LOAD BOTH MODELS
    # ===============================

    @st.cache_resource
    def load_models():

        english_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

        multilingual_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )

        return english_model, multilingual_model


    roberta_model, xlm_model = load_models()


    # ===============================
    # YOUTUBE API
    # ===============================

    youtube = build(
        "youtube",
        "v3",
        developerKey=st.secrets["YOUTUBE_API_KEY"]
    )


    # ===============================
    # SENTIMENT FUNCTION
    # ===============================

    def predict_sentiment(text):

        try:

            lang = detect(text)


            if lang == "en":

                out = roberta_model(text[:512])[0]


            elif lang in ["ta", "hi"]:

                out = xlm_model(text[:512])[0]


            else:

                out = xlm_model(text[:512])[0]


            label = out["label"]


            if label in ["LABEL_0", "NEGATIVE"]:

                return "Negative"


            elif label in ["LABEL_1", "NEUTRAL"]:

                return "Neutral"


            else:

                return "Positive"


        except LangDetectException:

            return "Neutral"


    # ===============================
    # ASPECTS (UNCHANGED)
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

            t_lower = t.lower()

            for aspect, keys in PRODUCT_ASPECTS.items():

                if any(k in t_lower for k in keys):

                    rows.append({

                        "Aspect": aspect,

                        "Sentiment": predict_sentiment(t)

                    })

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

        return [i["id"]["videoId"] for i in res["items"]]



    def fetch_comments(video_id, limit=100):

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


            for vid in search_videos(topic):

                comments.extend(fetch_comments(vid))


            st.success(f"Fetched {len(comments)} comments")


            sentiments = [

                predict_sentiment(c)

                for c in comments

            ]


            show_sentiment_charts(sentiments)


            st.subheader("📄 Sample Comments")


            for i, c in enumerate(comments[:5], 1):

                st.write(f"{i}. {c}")


            if analysis_type == "Product":

                st.subheader("🧠 Aspect-Based Sentiment")

                absa = aspect_based_sentiment(comments)


                if not absa.empty:

                    st.bar_chart(

                        absa.value_counts().unstack().fillna(0)

                    )


            else:

                st.info(

                    "Aspect-based sentiment not applicable for general topics."

                )


    # ===============================
    # TAB 2
    # ===============================

    with tab2:

        channel_name = st.text_input("Enter Channel Name")


        if st.button("Analyze Channel"):


            search = youtube.search().list(
                q=channel_name,
                part="snippet",
                type="channel",
                maxResults=1
            ).execute()


            if not search["items"]:

                st.error("❌ Channel not found")


            else:

                cid = search["items"][0]["snippet"]["channelId"]


                channel_data = youtube.channels().list(
                    part="snippet,statistics",
                    id=cid
                ).execute()["items"][0]


                channel_title = channel_data["snippet"]["title"]


                channel_desc = channel_data["snippet"].get("description", "No description")


                subs = int(channel_data["statistics"].get("subscriberCount", 0))


                total_views = int(channel_data["statistics"].get("viewCount", 0))


                total_videos = int(channel_data["statistics"].get("videoCount", 0))


                st.subheader(f"📺 Channel: {channel_title}")


                st.caption(channel_desc)


                m1, m2, m3 = st.columns(3)


                m1.metric("Subscribers", f"{subs:,}")


                m2.metric("Total Views", f"{total_views:,}")


                m3.metric("Total Videos", f"{total_videos:,}")


                videos = youtube.search().list(
                    channelId=cid,
                    part="id",
                    type="video",
                    maxResults=25
                ).execute()["items"]


                video_rows = []


                comments = []


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


                    video_rows.append({
                        "Title": title,
                        "Views": views
                    })


                    comments.extend(fetch_comments(vid, 40))


                df_vid = pd.DataFrame(video_rows).sort_values("Views", ascending=False)


                c1, c2 = st.columns(2)


                c1.metric("Total Comments Analyzed", len(comments))


                c2.metric("Total Likes (Recent Videos)", f"{total_likes:,}")


                st.subheader("📊 Views per Video (Recent)")


                st.bar_chart(df_vid.set_index("Title")["Views"])


                st.subheader("🎬 Video Titles & Views")


                st.dataframe(df_vid, use_container_width=True, height=300)


                sentiments = [

                    predict_sentiment(c)

                    for c in comments

                ]


                show_sentiment_charts(sentiments)


    # ===============================
    # TAB 3
    # ===============================

    with tab3:

        file = st.file_uploader("Upload CSV", type="csv")


        if file and st.button("Analyze Dataset"):


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

                "text",

                "tweet",

                "comment",

                "review",

                "content",

                "sentence"

            ]


            text_col = next(

                (c for c in TEXT_COLS if c in df.columns),

                None

            )


            if not text_col:

                st.error("❌ No text column detected.")


            else:


                texts = df[text_col].astype(str).head(1000)


                sentiments = texts.apply(predict_sentiment)


                show_sentiment_charts(sentiments)
