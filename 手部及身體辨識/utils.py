import os
from datetime import datetime

def create_video_filename(prefix, base_path):
    """
    創建帶有時間戳記的影片檔案路徑
    
    Args:
        prefix (str): 檔案名稱前綴 (例如 'original' 或 'pose')
        base_path (str): 基礎儲存路徑
        
    Returns:
        str: 完整的影片檔案路徑
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.mp4'
    return os.path.join(base_path, filename)