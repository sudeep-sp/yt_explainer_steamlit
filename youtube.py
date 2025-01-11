from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def fetch_youtube_transcript(video_url):
    try:
        # Normalize the video ID
        if "watch?v=" in video_url:
            video_id = video_url.split("v=")[-1]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format")

        # Remove any additional parameters in the URL
        video_id = video_id.split("&")[0]

        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[
                                                         'en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'nl', 'pt', 'ru', 'zh-Hans', 'zh-Hant', 'hi'])
        # Combine all text into a single string
        full_text = " ".join([item['text'] for item in transcript])
        return full_text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def explainer(transcript_text):
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    template = """
        You are a highly accurate YouTube AI assistant. Your task is to provide a detailed explanation of the YouTube video transcript.

        Use the transcript provided from the video: {transcript_text}

        Instructions:
        1. Go through the transcript step by step and explain every detail.
        2. Ensure that no important information is skipped or omitted.
        3. Maintain a logical flow in your explanation, matching the structure of the transcript.
        4. Use clear and concise language to make the explanation easy to understand.
        5. Avoid summarizing. Instead, focus on providing a full, detailed explanation.
        6. response needs to be well structured like heading, bullet points etc.. and easy to follow.
        7. dont use "The transcript" use "The video" instead

        Your response should provide an accurate and pin-to-pin explanation of the transcript content.
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser
    result = chain.invoke({"transcript_text": transcript_text})
    return result


def summarizer(transcript_text):
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    template = """
        You are a highly accurate YouTube AI assistant. Your task is to provide a **concise summary** of the YouTube video transcript.

        Use the transcript provided from the video: {transcript_text}

        Instructions:
        1. Summarize the video in **5-7 sentences**, capturing all the key points mentioned in the transcript.
        2. Exclude any irrelevant or repetitive information.
        3. The summary should focus on the main ideas and important details, providing a clear understanding of the video's content.
        4. Use simple and professional language to make the summary clear and engaging.

        Your response should accurately represent the overall context and purpose of the video.
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser
    result = chain.invoke({"transcript_text": transcript_text})
    return result


# Streamlit App
st.write("🛠️ Built by Sudeep S Patil")
st.title("YouTube Tool: Explain or Summarize Videos")
st.write("This tool allows you to create detailed explanations or summaries of YouTube videos simply by providing the video URL.")

# Input for YouTube video URL
video_url = st.text_input("Enter the YouTube video URL:")

# Reset session state if the video URL changes
if "previous_url" not in st.session_state:
    st.session_state.previous_url = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "summary" not in st.session_state:
    st.session_state.summary = None

if video_url:
    # Reset responses if a new URL is entered
    if video_url != st.session_state.previous_url:
        st.session_state.previous_url = video_url
        st.session_state.explanation = None
        st.session_state.summary = None

    # Embed the YouTube video
    st.video(video_url)

    # Fetch the transcript
    with st.spinner("Fetching transcript..."):
        transcript_text = fetch_youtube_transcript(video_url)

    if transcript_text:
        st.success("Transcript fetched successfully!")

        # Display options for explanation or summary
        task = st.radio("Choose what you want to do:",
                        ("Explain the Video", "Summarize the Video"))

        if task == "Explain the Video":
            if st.session_state.explanation is None:
                with st.spinner("Generating explanation..."):
                    st.session_state.explanation = explainer(transcript_text)
            st.subheader("Explanation:")
            st.write(st.session_state.explanation)

        elif task == "Summarize the Video":
            if st.session_state.summary is None:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarizer(transcript_text)
            st.subheader("Summary:")
            st.write(st.session_state.summary)
    else:
        st.error("Transcript not available or an error occurred.")
