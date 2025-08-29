import re

def extract_youtube_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from a variety of URL formats.
    """
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",
        r"(?:https?://)?youtu\.be/([^?&]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^?&]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([^?&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
