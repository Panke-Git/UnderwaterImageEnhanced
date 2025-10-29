"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: output_img.py
    @Time: 2025/9/6 23:35
    @Email: None
"""
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
import src.models as models


def resize_img(inp, out_path):
    img = cv2.imread(inp, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # 直接缩放到 256×256
    cv2.imwrite(out_path, resized)
    print("Saved ->", out_path)


# ===================== 配置部分 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（与你训练时用的模型保持一致）
model = models.UNetHybridAttentionV23_2().to(device)
model.load_state_dict(torch.load(
    r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2\20250819_222858\best_result\TOP_PSNR.pth",
    map_location=device
))
model.eval()

# 输入图片路径 & 输出路径
input_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_input.jpg"
GT_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_GT.jpg"

out_256_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_256_out.jpg"
out_restore_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_out.jpg"

input_256_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_input256.jpg"
GT_256_path = r"G:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\2634\2634_GT256.jpg"

# 可选：先各自保存 256×256 版本（非必须，仅检查用）
resize_img(input_path, input_256_path)
resize_img(GT_path, GT_256_path)


# 推理尺寸
infer_size = (256, 256)

# 是否等比缩放（带黑边），默认 False 为直接拉伸
keep_aspect_ratio = False
# ==================================================

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def resize_with_letterbox(img_rgb, size=(256, 256), fill=(0, 0, 0)):
    """
    等比缩放到不超过 size 的最大尺寸，然后四周填充到 size。
    返回：letterbox图, 缩放后的宽高, 填充 (left, top)
    """
    h, w = img_rgb.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算填充
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    # 填充
    letterboxed = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT, value=fill)
    return letterboxed, (new_w, new_h), (left, top)

def remove_letterbox_and_restore(out_256_rgb, new_wh, pad_lt, original_wh):
    """
    去掉 letterbox 的 padding，再缩放回原图尺寸
    """
    new_w, new_h = new_wh
    left, top = pad_lt
    # 裁掉 padding 区域
    crop = out_256_rgb[top:top+new_h, left:left+new_w, :]
    # 还原回原始尺寸
    W0, H0 = original_wh
    restored = cv2.resize(crop, (W0, H0), interpolation=cv2.INTER_CUBIC)
    return restored

def main():
    # 读取原图(BGR) -> RGB
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取输入图片: {input_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    # 读取 GT (BGR) -> RGB
    gt_bgr = cv2.imread(GT_path)
    if gt_bgr is None:
        raise FileNotFoundError(f"无法读取GT图片: {GT_path}")
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

    # --------- 预处理到 256x256 ----------
    if keep_aspect_ratio:
        letterbox_rgb, new_wh, pad_lt = resize_with_letterbox(img_rgb, infer_size, fill=(0, 0, 0))
        net_in_rgb = letterbox_rgb
    else:
        net_in_rgb = cv2.resize(img_rgb, infer_size, interpolation=cv2.INTER_AREA)

    img_tensor = to_tensor(net_in_rgb).unsqueeze(0).to(device)  # [1,C,H,W], 0-1

    # --------- 推理 ----------
    with torch.no_grad():
        out = model(img_tensor).clamp(0, 1)   # [1,C,256,256]

    # --------- 保存 256x256 结果（可选） ----------
    out_256 = out.squeeze(0).cpu().numpy()  # [C,H,W]
    out_256 = np.transpose(out_256, (1, 2, 0))  # [H,W,C], 0-1
    out_256_u8 = (out_256 * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(out_256_path, cv2.cvtColor(out_256_u8, cv2.COLOR_RGB2BGR))

    # --------- 还原回原始分辨率 ----------
    if keep_aspect_ratio:
        restored_rgb = remove_letterbox_and_restore(out_256_u8, new_wh, pad_lt, (W0, H0))
    else:
        # 直接把 256x256 结果缩放回原尺寸
        restored_rgb = cv2.resize(out_256_u8, (W0, H0), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(out_restore_path, cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    # ================== 计算 PSNR / SSIM ==================
    # 1) 与 256×256 对齐的指标（训练验证口径）
    gt_256_rgb = cv2.resize(gt_rgb, infer_size, interpolation=cv2.INTER_AREA)
    gt_256_t = to_tensor(gt_256_rgb).unsqueeze(0).to(device)   # [1,C,256,256], 0-1
    pred_256_t = torch.from_numpy(np.transpose(out_256_u8.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)
    # 或者直接用网络输出 out（已经是 [0,1] tensor），两种等价
    pred_256_t = out   # 更直接

    psnr_256 = peak_signal_noise_ratio(pred_256_t, gt_256_t, data_range=1.0, dim=(1,2,3), reduction='none').item()
    ssim_256 = structural_similarity_index_measure(pred_256_t, gt_256_t, data_range=1.0).item()

    print(f"[METRIC@256]  PSNR={psnr_256:.4f}, SSIM={ssim_256:.4f}")

    # 2) 与原始分辨率对齐的指标（视觉还原口径）
    pred_orig_t = to_tensor(restored_rgb).unsqueeze(0).to(device)  # [1,C,H0,W0], 0-1
    gt_orig_t   = to_tensor(gt_rgb).unsqueeze(0).to(device)        # [1,C,H0,W0], 0-1

    psnr_orig = peak_signal_noise_ratio(pred_orig_t, gt_orig_t, data_range=1.0, dim=(1,2,3), reduction='none').item()
    ssim_orig = structural_similarity_index_measure(pred_orig_t, gt_orig_t, data_range=1.0).item()

    print(f"[METRIC@ORIG] PSNR={psnr_orig:.4f}, SSIM={ssim_orig:.4f}")
    # ======================================================

    print(f"✅ 推理完成：\n - 256x256结果: {out_256_path}\n - 还原到原尺寸: {out_restore_path}")

if __name__ == "__main__":
    main()
