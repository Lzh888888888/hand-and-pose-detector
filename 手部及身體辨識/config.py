import os

# 影片儲存路徑設定
VIDEO_SAVE_PATH = os.path.join(os.path.expanduser("~"), "Documents", "program", "OpenCV實作", "手部及身體辨識", "video")

# 確保影片儲存路徑存在
def ensure_video_path():
    if not os.path.exists(VIDEO_SAVE_PATH):
        os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
    return VIDEO_SAVE_PATH