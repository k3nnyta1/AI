# embed_main.py
# 步驟1：將原圖加上浮水印，產生 watermarked.png
# 執行：python embed_main.py

import cv2
from watermark_embed import embed_watermark

# ====== 參數設定 ======
input_img = 'input.png'  # 原始圖片，請放灰階圖
watermarked_img = 'watermarked.png'  # 輸出：加浮水印後的圖片
watermark_str = '4111029024'  # 浮水印內容（學號）

redundancy = 10  # 冗餘度，每個bit重複嵌入的區塊數，越高越robust
repeat = 3       # 重複碼，每個bit重複幾次，提升錯誤更正能力
coef_base = (2, 3)  # DCT嵌入座標，建議用中頻
# delta 控制嵌入時的最小差值，越大越robust但失真也大
# 若圖片失真明顯可適度調小
delta = 60       # 嵌入時的最小差值

# ====== 執行嵌入 ======
# 會自動將浮水印（含前導碼）嵌入圖片，產生 watermarked.png
embed_watermark(
    input_img,           # 原圖路徑
    watermarked_img,     # 輸出路徑
    watermark_str,       # 浮水印內容
    redundancy=redundancy,  # 冗餘度
    coef_base=coef_base,    # DCT嵌入座標
    repeat=repeat,          # 重複碼
    delta=delta             # 嵌入強度
)
print(f'浮水印已嵌入，輸出檔案：{watermarked_img}') 