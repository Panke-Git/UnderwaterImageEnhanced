"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: resize.py
    @Time: 2025/9/7 01:19
    @Email: None
"""
import cv2

in_path  = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\1371_GT.jpg"
out_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\1371_GT256.jpg"

img = cv2.imread(in_path, cv2.IMREAD_COLOR)
resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # 直接缩放到 256×256
cv2.imwrite(out_path, resized)
print("Saved ->", out_path)