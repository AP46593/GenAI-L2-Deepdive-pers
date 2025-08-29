from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from app.tools.youtube_utils import extract_youtube_video_id
from app.graph.state import AppState

def transcript_agent_node(state: AppState) -> AppState:
    last_user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    video_id = extract_youtube_video_id(last_user_msg)
    if not video_id:
        state["messages"].append({
            "role": "assistant",
            "content": "❌ I couldn't extract a valid YouTube video ID from your message. Please try again with a full URL."
        })
        return state

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore
        transcript_text = " ".join([t["text"] for t in transcript_list])
        state["transcript"] = transcript_text
        state["messages"].append({
            "role": "assistant",
            "content": "📄 Transcript extracted successfully from the video."
        })
    except TranscriptsDisabled:
        state["messages"].append({
            "role": "assistant",
            "content": "⚠️ This video does not have transcripts available."
        })
    except Exception as e:
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Failed to fetch transcript: {str(e)}"
        })

    return state
