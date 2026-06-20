from googleapiclient.discovery import build  # type: ignore
import pandas as pd  # type: ignore
from langdetect import detect, LangDetectException  # type: ignore
import time

# --- Step 1: Setup your API key ---
api_key = "YOUTUBE_API_KEY"

# --- Step 2: Connect to YouTube API ---
youtube = build('youtube', 'v3', developerKey=api_key)

# --- Step 3: Load your manually collected video links ---
manual_video_links = [
    "https://youtu.be/qxFIGAChur8?si=qQY_qOmJX1b0qO0Y",
    "https://youtu.be/8tmO14S7k8E?si=kBMmAzZq0QMhGMgW",
    "https://youtu.be/fMRSCq1LvaY?si=oWaTL-WcLSgIJsbh",
    "https://youtu.be/fj9al7lrcIg?si=nMrssvDvqbICI85G",
    "https://youtu.be/8ZI3cudNtfE?si=YUxkZ_GuOtk2P4zq",
    "https://youtu.be/0Hx8D913AHE?si=g0AgTciY8z1qlyRz",
    "https://youtu.be/K5cG6S3I8tQ?si=S2PtWTA3AbSKUbV3",
    "https://youtu.be/XSDYznL5Vfo?si=LLVQYo5IKGuDVA4q",
    "https://youtu.be/iIsW9b3t2R4?si=GlqwU2ozKsMhucup",
    "https://youtu.be/yjG7WWX2gG0?si=sou-Ai2QOCOcpZvx",
    "https://youtu.be/-ejRS7TodKk?si=-S3u8-WZc4ZJY26Q",
    "https://youtu.be/bT0IkffwHS0?si=8u6S7dGL-p6FwK4d",
    "https://youtu.be/qUDPB-rCJhM?si=KJhhrkBTt5Xf6RTU",
    "https://youtu.be/3wfRIlsnl3g?si=E098XE8-MHxTsXeG",
    "https://youtu.be/bbGEC2qeuTM?si=L9yR39Tn6XYt_ep1",
    "https://youtu.be/G1hM-gGnVTk?si=V6PohXT_U-kMO27n",
    "https://youtu.be/USd8UniIWmc?si=h5Erfb4_811Cd95t",
    "https://youtu.be/1dwv34vTKwI?si=vpYRPZ_l0oK0CrAC",
    "https://youtu.be/4LCLDXaeJj4?si=y7X1cM8aAD9-f2Sj",
    "https://youtu.be/y_wphyKXg38?si=yb7s1ZMMStkuRYJP",
    "https://youtu.be/1XnLPbiITnY?si=gzSDJ1FFsDvsljsT",
    "https://youtu.be/ijwUM39oICQ?si=uB1cyuDU6erpriz9",
    "https://youtu.be/kpmVJPJFMEM?si=RSBp_SXcoK34So-2",
    "https://youtu.be/0xuwd9TBiOE?si=zriPfQdT-908QAwC",
    "https://youtu.be/c13a5h95Pm8?si=PVo6N9b0CycmScft",
    "https://youtu.be/nhTrUHcbm7I?si=5PANYxmqFb1bRchl",
    "https://youtu.be/rJKIo_eTlC0?si=YclJJsgM6NwX0MvB",
    "https://youtu.be/PdReQlDdZN8?si=f_boeBLY5AG6Os2G",
    "https://youtu.be/v0_0LZoAJSM?si=gS-uZ2WGmVPrRIAE",
    "https://youtu.be/vnYCpXmynYY?si=OPZLyGVc2hvBrF4j",
    "https://youtu.be/-DGZqxFMhQw?si=0DxjccSW2EprOGhu",
    "https://youtu.be/lJcS1KKgw_c?si=H4IGnAC2zibqt35G",
    "https://youtu.be/4CPCeFQ7jLg?si=nsEeASL4WCPcEtKV",
    "https://youtu.be/1y8vGPvz-_A?si=bolaTXOmJjSzFalf",
    "https://youtu.be/Rqxjn3ZgCh8?si=pcGYQt4dJXd5XNPy",
    "https://youtu.be/7MvEEygfdqA?si=6dbd-P30gN267lrM",
    "https://youtu.be/OzQmQdKZCoM?si=fxdM9-DIDI5UHb3s",
    "https://youtu.be/DtUdxvYR2X4?si=Q9wsLeVeWjoNm0Hs",
    "https://youtu.be/qUDPB-rCJhM?si=KqqJNELBsab6xbgO",
    "https://youtube.com/shorts/aOD7A_Yx-rE?si=q_0YnIhvAIzpI1dF",
    "https://youtu.be/3wfRIlsnl3g?si=tMtuDh0eFlz40UVp",
    "https://youtu.be/fY5k1xttV3s?si=SDMsCy-In1if1Sl_",
    "https://youtu.be/Ni7elmqpjMg?si=93yWAsMaOcEEkHCv",
    "https://youtu.be/g9hJR4G3YOY?si=TC6EbgqY19E4Qax0",
    "https://youtu.be/uNdoeTBCBUs?si=Kb72q-ZnqkExD8oD",
    "https://youtu.be/4qzrlAaA9MM?si=-kBgH-0M2y_05G0C",
    "https://youtu.be/g6rtxi7UY98?si=DcaSskEnoXC5XsVw",
    "https://youtube.com/shorts/uSRl_SnatZw?si=fQR_LsFrQlWZN4AZ",
    "https://youtu.be/7Qs2H29iqm8?si=bY5fyxRurdCE08Cz",
    "https://youtu.be/ahWq45sV4zQ?si=2ab1eGfO2hLzBIbt",
    "https://youtu.be/DQ1z4DAtkJA?si=0SgwjVrRPn6LPA0D",
    "https://youtu.be/sfmMNTe9COg?si=Li9g393yXHy_jXnh",
    "https://youtu.be/a09N5TXpw3g?si=uIcpA_T4FXmrUIXX",
    "https://youtu.be/VDxj2Xh01O0?si=lhIxoqxlZwrFyeHW",
    "https://youtu.be/AkiAffE2ibM?si=rM1vVLaV947K214z",
    "https://youtu.be/LijMfvd5GaA?si=bpyCcSfMyzGFYQQI",
    "https://youtu.be/hbkAxqhEdXc?si=QMUz2bNbdbgdjr_w",
    "https://youtu.be/B1vyt0HtcP4?si=IwdaHBc_iF6Vnk9W",
    "https://youtu.be/jyN2LH-sH80?si=E3JmwEkR58l1iTsg",
    "https://youtu.be/setb8jzxcYU?si=UIb8BJ1HH7AulCch",
    "https://youtu.be/Ev6_u-nx84s?si=GhOH3SOK5buED7MA",
    "https://youtu.be/ub2F_l2UXfE?si=eQWUHWx8DfDnMMfy",
    "https://youtube.com/shorts/7UTprRLTepE?si=Kn6GSaoQtYgvlL_7",
    "https://youtu.be/L-wZyr7KWck?si=7mxBb29XNNwDMSOG",
    "https://youtu.be/kXXYlNYClFg?si=JNJ4fMJKY7LZ66nr",
    "https://youtu.be/YBCP2hAA3LY?si=7kRovsmzM-8LqyT9",
    "https://youtu.be/ZRUf5uZl7Y4?si=24_4fzS4ratKhRSl",
    "https://youtu.be/zWuPQsqq4F0?si=TyP7RMhDciZqIT5i",
    "https://youtu.be/1-R0MZ09JE4?si=hS_cZvx0pSYQ4o3f",
    "https://youtu.be/XQ8UYNlbIMs?si=5eN362G6O-jjDNWm",
    "https://youtu.be/4nqZVRm3WDc?si=EwPWLROSBkLiFnpY",
    "https://youtu.be/1LIgRf4uQ_0?si=2tWEivBb2kX9YD_y",
    "https://youtu.be/si2k80xN25I?si=JSVtXzl0Swj0zMwB",
    "https://youtu.be/7x0WH1dDxl8?si=ShskIzNxrFV6PLwx",
    "https://youtu.be/aaUIR8IzMEg?si=9_pJMGN9ffli2vQt",
    "https://youtu.be/gT1Gwe7MxZI?si=L9V0xKCZnF3evnsT",
    "https://youtu.be/ABFKtC4JfhE?si=-uqUgtXJNYMfEjvl",
    "https://youtu.be/Kyx-H5x_fPY?si=VNp_h4fbUNhT9ETx",
    "https://youtu.be/nIHgD9pNycI?si=qR81t2rXCMOAQ9R_",
    "https://youtu.be/uRIoaj-6Uf4?si=v3DAJNC62GG_jJB3",
    "https://youtu.be/uRIoaj-6Uf4?si=mWtKzAHuqq8kzqO9",
    "https://youtu.be/shrE_2b02oY?si=H-028ojBANg1evE0",
    "https://youtu.be/A_S1U0i_Jrs?si=j2Ki6oh1PDNzKyqu",
    "https://youtu.be/bAMhfrp9jdY?si=frSV2a4XQA1uEgKs",
    "https://youtu.be/3ePVP1mhwZY?si=qH0MQhKeu_v4CW4x",
    "https://youtu.be/oMZHKs_IbrE?si=rvWlT7HWZIUlXjoa",
    "https://youtu.be/5gTU5pDiq9E?si=aToviQ9I1ZdJJhFG",
    "https://youtu.be/CYSwjOQDZsI?si=qw9M17QgMiHPf9AU",
    "https://youtu.be/4XVujbzP0b8?si=9yiDZ_wb-FDTFoWs",
    "https://youtu.be/uSaCMvQb3rA?si=xfqjvstZKCIhPfUL",
    "https://youtu.be/DtUdxvYR2X4?si=Tu-Ez6-qMQ92apOI"
    "https://youtu.be/fMRSCq1LvaY?si=JXVcXnbY0Pe_n3l3"
    "https://youtu.be/fY5k1xttV3s?si=KL89S1a-9uEgFflG"
    "https://youtu.be/fY5k1xttV3s?si=PEASXq12rqNrnu7k"
    "https://youtu.be/qUDPB-rCJhM?si=1jX5A0TN_sHXdOcz"
    "https://youtu.be/z762YeyBaoo?si=9qlRCBDyYcJ1V5Wu"
    "https://youtu.be/sYYkXszeOrc?si=-s9CE2xcsOaxIfof"
    "https://youtu.be/_uRLIlgH258?si=WNVYwkevWnlXBQgY"
    "https://youtu.be/uNdoeTBCBUs?si=NFRodtQ4aBXQWl6w"
    "https://youtu.be/PQDRTSOvR2I?si=FxKwalYGxOS-F9n8"
    "https://youtu.be/B1vyt0HtcP4?si=Y7GDB88iaQyoBWcU"
    "https://youtu.be/h0AIoAO52Xc?si=-eR7QRbylND7SJUR"
    "https://youtu.be/QBYnSB5wQkk?si=qz6iyeiw7p-WVdfS"
    "https://youtu.be/3wfRIlsnl3g?si=RCLK2S2eoBYmNoBZ"
    "https://youtu.be/vvl23flaRyU?si=YMcBaXhyo_gxbHOm"
    "https://youtu.be/tHUiz5thqa0?si=SnrKKldvEagZsBEv"
    "https://youtu.be/K5cG6S3I8tQ?si=3WXPPeMcuDMMnKFz"
    "https://youtu.be/5oPe2AHFYh0?si=B-zwE9Avr4oIrIpp"
    "https://youtu.be/oA5jk10Demc?si=VZ_YLSvdo2A_bqsB"
    "https://youtu.be/70fYJzlIagM?si=05lMInHW6Dkkve3H"
    "https://youtu.be/k0fha_ty7to?si=RZDZHwK9D4Y4psSD"
    "https://youtu.be/ki2n_Bp66lA?si=eL_MWpILIVZ0obzY"
]

# ✅ Remove duplicate links
manual_video_links = list(set(manual_video_links))
print(f"🎥 Total unique video links loaded: {len(manual_video_links)}")

# --- Extract proper video IDs from all links ---
def extract_video_id(link):
    if "youtu.be/" in link:
        return link.split("youtu.be/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in link:
        return link.split("v=")[-1].split("&")[0]
    elif "youtube.com/shorts/" in link:
        return link.split("shorts/")[-1].split("?")[0]
    return None

video_ids = list(set([extract_video_id(link) for link in manual_video_links if extract_video_id(link)]))
print(f"🎥 Found {len(video_ids)} unique videos for comment extraction")

# --- Step 4: Fetch comments ---
comments = []
authors_seen = set()  # optional

for index, vid in enumerate(video_ids, start=1):
    print(f"\n📥 ({index}/{len(video_ids)}) Fetching comments for video: {vid}")
    next_page_token = None
    video_comment_count = 0

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()
        except Exception as e:
            print(f"⚠️ Skipping video {vid} due to error: {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment = snippet["textDisplay"]
            author = snippet["authorDisplayName"]
            published = snippet["publishedAt"]

            # optional: ignore duplicate authors
            if author in authors_seen:
                continue
            authors_seen.add(author)

            # ✅ Detect English-only comments
            try:
                if detect(comment) == "en":
                    comments.append([vid, author, comment, published])
                    video_comment_count += 1
            except LangDetectException:
                continue

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.3)

    print(f"✅ {video_comment_count} English comments collected from video {vid}")

    if len(comments) >= 7000:
        print("🎯 Reached 7000 comments limit, stopping early.")
        break

# --- Step 5: Save results ---
df = pd.DataFrame(comments, columns=["VideoID", "Author", "Comment", "PublishedAt"])
df.to_csv("youtube_easter_english_comments.csv", index=False, encoding="utf-8")
print(f"\n✅ Done! Total English comments collected: {len(df)}")


