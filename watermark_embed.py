import cv2
import numpy as np
from utils import img_to_blocks, blocks_to_img, dct2, idct2, str_to_bits

# 將每個bit重複repeat次，提升錯誤更正能力
# 例如 bit=1, repeat=3 會變成 [1,1,1]
def repeat_bits(bits, repeat=3):
    return [b for bit in bits for b in [bit] * repeat]

# 固定32位元前導碼，用於同步與方向判斷，提升抗攻擊能力
# 前導碼設計為高低頻交錯，便於方向偵測
def preamble_bits():
    preamble = "10101010110011001110001011110000"
    return [int(x) for x in preamble]

# 浮水印嵌入主函式
# 將浮水印（含前導碼）嵌入圖片DCT中頻區塊
# img_path: 原圖路徑
# output_path: 輸出路徑
# watermark_str: 浮水印內容
# block_size: DCT區塊大小（預設8x8）
# coef_base: DCT嵌入座標（建議中頻）
# redundancy: 冗餘度，每bit重複嵌入幾個區塊
# repeat: 重複碼，每bit重複幾次
# delta: 嵌入時的最小差值，越大越robust
# debug_preamble_only: 僅嵌入前導碼（debug用）
def embed_watermark(
    img_path,
    output_path,
    watermark_str,
    block_size=8,
    coef_base=(2, 3),
    redundancy=10,
    repeat=3,
    delta=60,
    debug_preamble_only=False,
):
    # 讀取灰階圖片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape
    img = img.astype(np.float32)
    # 將圖片分割為8x8區塊
    blocks = img_to_blocks(img, block_size)
    N = len(blocks)

    # 準備嵌入bit序列（前導碼+浮水印）
    if debug_preamble_only:
        bits = preamble_bits()
        print(f"[DEBUG] 嵌入前導碼bit: {bits}")
    else:
        bits = preamble_bits() + str_to_bits(watermark_str)
        print(f"[DEBUG] 嵌入前導碼bit: {preamble_bits()}")
    # 重複碼處理
    bits = repeat_bits(bits, repeat)
    total_bits = len(bits)
    # 檢查圖片容量是否足夠
    if total_bits * redundancy > N // 2:
        raise ValueError("Image too小，無法容納這麼多冗餘度")

    # DCT嵌入座標（對稱座標用於抗翻轉）
    x, y = coef_base
    x_sym, y_sym = 7 - x, 7 - y
    used = set()  # 避免重複嵌入同一區塊
    watermarked_blocks = [block.copy() for _, block in blocks]
    # 依序將每個bit嵌入多個對稱區塊
    for bit_idx in range(total_bits):
        for r in range(redundancy):
            idx = bit_idx * redundancy + r
            if idx >= N // 2:
                continue
            idx_sym = N - 1 - idx
            for real_idx in [idx, idx_sym]:
                if real_idx in used:
                    continue
                used.add(real_idx)
                dct_block = dct2(blocks[real_idx][1])
                bit = bits[bit_idx]
                c1, c2 = dct_block[x, y], dct_block[x_sym, y_sym]
                diff = c1 - c2
                # 根據bit調整係數差值
                if bit == 1:
                    if diff <= delta:
                        adjust = (delta - diff) / 2 + 1
                        dct_block[x, y] += adjust
                        dct_block[x_sym, y_sym] -= adjust
                else:
                    if diff >= -delta:
                        adjust = (delta + diff) / 2 + 1
                        dct_block[x, y] -= adjust
                        dct_block[x_sym, y_sym] += adjust
                # 反轉DCT回空間域
                watermarked_blocks[real_idx] = idct2(dct_block)
    # 合併所有區塊，重建圖片
    watermarked_img = blocks_to_img(list(zip([b[0] for b in blocks], watermarked_blocks)), (h, w), block_size)
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, watermarked_img)
    print(f"Watermarked image saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python watermark_embed.py input.png output.png \"4111029024\"")
    else:
        embed_watermark(sys.argv[1], sys.argv[2], sys.argv[3]) 