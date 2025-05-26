"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: tribute_banner.py
    @Time: 2025/5/24 20:31
    @Email: None
"""

def show_banner():
    with open('/root/cyx/CL/PRO/MyPro/UnderwaterImageEnhanced/tools/banner.txt', 'rb') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            print(line)
        f.close()
