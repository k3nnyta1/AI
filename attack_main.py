# attack_main.py
# 步驟2：對加浮水印的圖片進行各種攻擊，產生多個攻擊後圖片
# 執行：python attack_main.py

import cv2
from attack_simulation import jpeg_compress, noise_attack
import numpy as np

# ====== 參數設定 ======
watermarked_img = 'watermarked.png'  # 已加浮水印的圖片

# ====== JPEG 壓縮攻擊 ======
# 將圖片以低品質存成 JPEG，模擬壓縮失真
jpeg_img = 'attacked_jpeg.jpg'
jpeg_compress(watermarked_img, jpeg_img, quality=20)  # quality 越低壓縮越嚴重
print(f'JPEG壓縮攻擊完成，輸出檔案：{jpeg_img}')

# ====== 高斯雜訊攻擊 ======
# 對圖片加入高斯雜訊，模擬感測器雜訊或傳輸干擾
noise_img = 'attacked_noise.png'
noise_attack(watermarked_img, noise_img, noise_std=60)  # noise_std 越大雜訊越明顯
print(f'高斯雜訊攻擊完成，輸出檔案：{noise_img}')

# ====== 翻轉攻擊（水平翻轉） ======
# 將圖片左右翻轉，模擬圖片被鏡像處理
flip_img = 'attacked_flip.png'
img = cv2.imread(watermarked_img, cv2.IMREAD_GRAYSCALE)
img_flip = cv2.flip(img, 1)
cv2.imwrite(flip_img, img_flip)
print(f'翻轉攻擊完成，輸出檔案：{flip_img}')

# ====== 90度旋轉攻擊 ======
# 將圖片逆時針旋轉90度，模擬圖片方向被改變
rot90_img = 'attacked_rot90.png'
img = cv2.imread(watermarked_img, cv2.IMREAD_GRAYSCALE)
img_rot90 = np.rot90(img, 1)
cv2.imwrite(rot90_img, img_rot90)
print(f'90度旋轉攻擊完成，輸出檔案：{rot90_img}') 