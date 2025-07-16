"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: uiqm_uciqe.py
    @Time: 2025/6/25 00:09
    @Email: None
"""

import numpy as np
import cv2
from scipy.stats import entropy

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
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag = np.clip(grad_mag, 1e-6, None)
        hist, _ = np.histogram(grad_mag.flatten(), bins=256, range=(0, np.max(grad_mag)), density=True)
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

