import re
import torch
import gradio as gr
import asyncio
import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline

# Load summarization model with GPU optimization
device = 0 if torch.cuda.is_available() else -1
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16, device=device)

def summary(input_text):
    """Summarizes the input text, handling token limits efficiently."""
    max_chunk_size = 1024  # Model limit
    chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
    
    # Summarize all chunks in a single batch
    summaries = text_summary(chunks, batch_size=len(chunks))
    return " ".join([s['summary_text'] for s in summaries])

def extract_video_id(url):
    """Extracts video ID from a YouTube URL."""
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

async def fetch_transcript(video_id):
    """Asynchronously fetches the YouTube transcript."""
    try:
        transcript = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        return text_transcript
    except Exception as e:
        return f"Error fetching transcript: {e}"

async def async_summary(input_text):
    """Asynchronously summarizes the input text."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, summary, input_text)

async def get_youtube_transcript(video_url):
    """Fetches and summarizes a YouTube video's transcript asynchronously."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL."
    
    # Fetch transcript asynchronously
    text_transcript = await fetch_transcript(video_id)
    if text_transcript.startswith("Error"):
        return text_transcript
    
    # Summarize asynchronously
    summarized_text = await async_summary(text_transcript)
    return summarized_text

# Define Gradio interface
demo = gr.Interface(
    fn=get_youtube_transcript,
    inputs=[gr.Textbox(label="Input YouTube URL to Summarize", lines=1)],
    outputs=[gr.Textbox(label="Summarized Text", lines=4)],
    title="YouTube Transcript Summarizer",
    description="Summarize YouTube video transcripts instantly!"
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()