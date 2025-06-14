# decode_main.py
# 步驟3：統一解碼所有攻擊後圖片，顯示浮水印提取結果
# 執行：python decode_main.py

from watermark_extract import extract_watermark
import cv2
import numpy as np

# ====== 參數設定 ======
watermark_str = '4111029024'  # 學號
watermark_len = len(watermark_str)
redundancy = 10  # 嵌入/提取時的冗餘度，需與嵌入時一致
repeat = 3  # 重複碼，需與嵌入時一致
coef_base = (2, 3)  # DCT嵌入座標，需與嵌入時一致
delta = 60  # 嵌入時的最小差值，需與嵌入時一致

# ====== 要解碼的圖片列表 ======
# 每個tuple: (圖片檔名, 說明)
img_list = [
    ('watermarked.png', '原圖'),
    ('attacked_jpeg.jpg', 'JPEG壓縮'),
    ('attacked_noise.png', '高斯雜訊'),
    ('attacked_flip.png', '翻轉'),
    ('attacked_rot90.png', '90度旋轉'),
]

# ====== 統一解碼並顯示結果 ======
def decode_with_rotations(img_path, desc):
    # 讀取圖片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'無法讀取 {img_path}')
        return
    # 準備8種方向：原圖、水平翻轉、垂直翻轉、水平+垂直翻轉、90度、270度、90度+翻轉、270度+翻轉
    imgs = [
        img,
        cv2.flip(img, 1),  # 水平翻轉
        cv2.flip(img, 0),  # 垂直翻轉
        cv2.flip(img, -1), # 水平+垂直翻轉
        np.rot90(img, 1),  # 逆時針90度
        np.rot90(img, 3),  # 逆時針270度
        cv2.flip(np.rot90(img, 1), 1),  # 90度+水平翻轉
        cv2.flip(np.rot90(img, 3), 1),  # 270度+水平翻轉
    ]
    print(f'\n[{desc} 提取]')
    for img_dir in imgs:
        # 直接在記憶體中嘗試所有方向，不產生暫存檔案
        result = extract_watermark(None, watermark_len, redundancy=redundancy, coef_base=coef_base, repeat=repeat, delta=delta, img_array=img_dir)
        if result is not None:
            # 只顯示 Extracted watermark，不顯示方向
            break
    else:
        print('Extracted watermark: (未找到正確前導碼，無法還原)')

# 依序對每個攻擊圖片進行解碼
for img_path, desc in img_list:
    decode_with_rotations(img_path, desc) 