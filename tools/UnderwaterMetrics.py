"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: UnderwaterMetrics.py
    @Time: 2025/6/24 23:48
    @Email: None
"""
import torch
import numpy as np
import cv2
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
import lpips

# --------- UIQM 和 UCIQE 本地实现 ---------
class UIQM:
    def __init__(self):
        self.block_size = 8

    def getUIQM(self, img):
        return (0.0282 * self._UICM(img) +
                0.2953 * self._UISM(img) +
                3.5753 * self._UIConM(img))

    def _UICM(self, img):
        img = img.astype(np.float32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        rg = r - g
        yb = 0.5 * (r + g) - b

        alpha = np.mean(rg)
        beta = np.mean(yb)
        rgsigma = np.std(rg)
        ybsigma = np.std(yb)

        power = np.sqrt(alpha ** 2 + beta ** 2) + 0.3 * np.sqrt(rgsigma ** 2 + ybsigma ** 2)
        return -power

    def _gradient_entropy(self, ch):
        gy, gx = np.gradient(ch)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = np.clip(grad_mag, 1e-6, None)
        hist, _ = np.histogram(grad_mag.flatten(), bins=256, range=(0, np.max(grad_mag)), density=True)
        from scipy.stats import entropy
        return entropy(hist + 1e-6)

    def _UISM(self, img):
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        e_R = self._gradient_entropy(R)
        e_G = self._gradient_entropy(G)
        e_B = self._gradient_entropy(B)
        return 0.299 * e_R + 0.587 * e_G + 0.114 * e_B

    def _UIConM(self, img):
        Y = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        m, n = Y.shape
        blocks = []
        for i in range(0, m - self.block_size + 1, self.block_size):
            for j in range(0, n - self.block_size + 1, self.block_size):
                block = Y[i:i + self.block_size, j:j + self.block_size]
                blocks.append(block)
        con = [np.max(b) - np.min(b) for b in blocks]
        return np.mean(con)


class UCIQE:
    def getUCIQE(self, img):
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = img_lab[:, :, 0]
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]

        chroma = np.sqrt(a ** 2 + b ** 2)
        sc = np.std(chroma)
        conl = np.percentile(L, 95) - np.percentile(L, 5)
        sat = chroma / (L + 1e-6)
        us = np.mean(sat)

        return 0.4680 * sc + 0.2745 * conl + 0.2576 * us


# --------- 自定义PCQI简易实现 ---------
def compute_pcqi(img_ref, img_dist, block_size=8):
    if img_ref.max() <= 1.0:
        img_ref = (img_ref * 255).astype(np.uint8)
    if img_dist.max() <= 1.0:
        img_dist = (img_dist * 255).astype(np.uint8)

    img_ref = img_ref.astype(np.float32)
    img_dist = img_dist.astype(np.float32)

    h, w = img_ref.shape
    M = h // block_size
    N = w // block_size

    total_score = 0
    count = 0
    for i in range(M):
        for j in range(N):
            ref_block = img_ref[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            dist_block = img_dist[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            mu_ref = np.mean(ref_block)
            mu_dist = np.mean(dist_block)
            sigma_ref = np.std(ref_block)
            sigma_dist = np.std(dist_block)

            c = (2 * sigma_ref * sigma_dist + 1e-12) / (sigma_ref ** 2 + sigma_dist ** 2 + 1e-12)
            l = (2 * mu_ref * mu_dist + 1e-12) / (mu_ref ** 2 + mu_dist ** 2 + 1e-12)
            total_score += c * l
            count += 1
    return total_score / count


# --------- CBPD 计算 ---------
def compute_cbpd(gray_pred, gray_target):
    contrast_pred = np.std(gray_pred)
    contrast_target = np.std(gray_target)
    contrast_diff = abs(contrast_pred - contrast_target)
    from skimage.metrics import structural_similarity as ssim
    ssim_val = ssim(gray_pred, gray_target)
    return contrast_diff * (1 - ssim_val)


# --------- FID 简化实现 ---------
class FIDCalculator:
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        self.resize = transforms.Resize((256, 256))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def _get_features(self, img_tensor):
        # img_tensor: [3,H,W], float [0,1]
        pil_img = self.to_pil(img_tensor.cpu())
        img = self.resize(pil_img)
        img = self.to_tensor(img).unsqueeze(0).to(self.device)  # [1,3,299,299]
        img = img * 2 - 1  # [-1,1]
        with torch.no_grad():
            features = self.inception_model(img)
        # average pool到1x1
        features = adaptive_avg_pool2d(features, output_size=(1, 1)).squeeze().cpu().numpy()
        return features

    def compute_fid(self, pred, target):
        feat_pred = self._get_features(pred)
        feat_target = self._get_features(target)

        mu1, mu2 = feat_pred, feat_target
        # 这里只计算欧氏距离代替Fisher距离，简单快速
        diff = mu1 - mu2
        fid = diff.dot(diff)
        return float(fid)


# --------- 综合指标计算类 ---------
class UnderwaterMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.uiqm_calc = UIQM()
        self.uciqe_calc = UCIQE()
        # self.fid_calc = FIDCalculator(device)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

    def to_numpy_img(self, tensor_img):
        # tensor: [3,H,W], float [0,1]
        img = tensor_img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def compute_metrics(self, pred, target=None):
        """
        pred,target: tensor [3,H,W], float [0,1]
        target可以为None，只计算无参考指标
        返回指标字典
        """
        pred_np = self.to_numpy_img(pred)
        uiqm = self.uiqm_calc.getUIQM(pred_np)
        uciqe = self.uciqe_calc.getUCIQE(pred_np)
        fdum = self.compute_fdum(pred_np)
        ccf = self.compute_ccf(pred_np)

        result = {
            'UIQM': uiqm,
            'UCIQE': uciqe,
            'FDUM': fdum,
            'CCF': ccf,
            'PCQI': None,
            'CBPD': None,
            'FID': None,
            'LPIPS': None,
        }

        if target is not None:
            target_np = self.to_numpy_img(target)
            gray_pred = cv2.cvtColor(pred_np, cv2.COLOR_RGB2GRAY)
            gray_target = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
            result['PCQI'] = compute_pcqi(gray_target, gray_pred)
            result['CBPD'] = compute_cbpd(gray_pred, gray_target)
            # result['FID'] = self.fid_calc.compute_fid(pred, target)
            result['LPIPS'] = self.compute_lpips(pred, target)

        return result

    def compute_fdum(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dark_channel = np.min(img, axis=2)
        dark_mean = np.mean(dark_channel)
        contrast = np.std(gray)
        return float(dark_mean / (contrast + 1e-6))

    def compute_ccf(self, img):
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))

    def compute_lpips(self, pred, target):
        pred = pred.unsqueeze(0).to(self.device) * 2 - 1
        target = target.unsqueeze(0).to(self.device) * 2 - 1
        with torch.no_grad():
            d = self.lpips_model(pred, target)
        return float(d.item())
