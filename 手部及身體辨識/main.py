import cv2
import numpy as np
from hand_detector import HandDetector
from pose_detector import PoseDetector
from config import ensure_video_path
from utils import create_video_filename

def main():
    # 確保影片儲存路徑存在
    video_path = ensure_video_path()
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    
    # 獲取影片尺寸
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 創建影片寫入器
    original_video_path = create_video_filename('original', video_path)
    skeleton_video_path = create_video_filename('skeleton', video_path)
    
    original_video = cv2.VideoWriter(
        original_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    skeleton_video = cv2.VideoWriter(
        skeleton_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    
    # 創建檢測器實例
    hand_detector = HandDetector()
    pose_detector = PoseDetector()
    
    print(f"影片將儲存在: {video_path}")
    print("按 'q' 結束錄製")
    
    while True:
        # 讀取攝像頭畫面
        success, frame = cap.read()
        if not success:
            break
            
        # 處理原始影像
        original_frame = frame.copy()
        
        # 處理手部和姿態檢測
        hand_results = hand_detector.process_frame(frame)
        pose_results = pose_detector.process_frame(frame)
        
        # 在原始影像上繪製
        hand_detector.draw_landmarks(original_frame, hand_results)
        pose_detector.draw_landmarks(original_frame, pose_results)
        
        # 在黑色背景上繪製骨架
        skeleton_frame = np.zeros(frame.shape, dtype=np.uint8)
        hand_detector.draw_landmarks(skeleton_frame, hand_results, draw_on_black=True)
        pose_detector.draw_landmarks(skeleton_frame, pose_results, draw_on_black=True)
        
        # 寫入影片
        original_video.write(original_frame)
        skeleton_video.write(skeleton_frame)
        
        # 顯示結果
        cv2.imshow("Original Feed", original_frame)
        cv2.imshow("Skeleton View", skeleton_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 釋放資源
    cap.release()
    original_video.release()
    skeleton_video.release()
    cv2.destroyAllWindows()
    
    print(f"\n錄製完成！")
    print(f"原始影片: {original_video_path}")
    print(f"骨架影片: {skeleton_video_path}")

if __name__ == "__main__":
    main()