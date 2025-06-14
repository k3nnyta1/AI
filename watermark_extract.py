import cv2
import numpy as np
from utils import img_to_blocks, dct2, bits_to_str
from collections import Counter

# 多數決，回傳出現最多次的bit
# 例如 [1,0,1,1] -> 1
# 若無資料則回傳0
def majority_vote(lst):
    if not lst:
        return 0
    return Counter(lst).most_common(1)[0][0]

# 固定32位元前導碼，用於同步與方向判斷
# 前導碼設計為高低頻交錯，便於方向偵測
# 提高抗攻擊能力
def preamble_bits():
    preamble = "10101010110011001110001011110000"
    return [int(x) for x in preamble]

# 嘗試從一張圖片提取完整bit序列，並比對前導碼
# 若前導碼正確，回傳浮水印bit，否則回傳None
# img: 輸入圖片（numpy array）
# total_bits: 預期bit總數（含前導碼）
# preamble_len: 前導碼長度
# block_size: DCT區塊大小
# coef_base: DCT嵌入座標
# redundancy: 冗餘度
# repeat: 重複碼
# delta: 嵌入時的最小差值
# debug: 是否顯示debug資訊
def try_extract(img, total_bits, preamble_len, block_size, coef_base, redundancy, repeat, delta, debug=False):
    blocks = img_to_blocks(img, block_size)
    N = len(blocks)
    x, y = coef_base
    x_sym, y_sym = 7 - x, 7 - y
    bits = []
    for bit_idx in range(total_bits):
        bit_votes = []
        for r in range(redundancy):
            idx = bit_idx * redundancy + r
            if idx >= N // 2:
                continue
            idx_sym = N - 1 - idx
            for real_idx in [idx, idx_sym]:
                _, block = blocks[real_idx]
                dct_block = dct2(block)
                c1, c2 = dct_block[x, y], dct_block[x_sym, y_sym]
                diff = c1 - c2
                # 根據係數差值判斷bit
                bit_votes.append(1 if diff > 0 else 0)
        # 冗餘區塊多數決
        bits.append(majority_vote(bit_votes))
    # 重複碼多數決
    final_bits = []
    for i in range(0, len(bits), repeat):
        chunk = bits[i : i + repeat]
        final_bits.append(majority_vote(chunk))
    # 前導碼比對
    preamble = preamble_bits()
    if debug:
        print(f"[DEBUG] 提取前導碼bit: {final_bits[:preamble_len]}")
    if final_bits[:preamble_len] == preamble:
        return final_bits[preamble_len:]
    else:
        return None

# 浮水印提取主函式
# img_path: 檔案路徑或 None
# watermark_length: 浮水印長度（字元數）
# block_size: DCT區塊大小
# coef_base: DCT嵌入座標
# redundancy: 冗餘度
# repeat: 重複碼
# delta: 嵌入時的最小差值
# debug: 是否顯示debug資訊
# img_array: numpy array 圖片（優先）
def extract_watermark(
    img_path=None,
    watermark_length=None,
    block_size=8,
    coef_base=(2, 3),
    redundancy=10,
    repeat=3,
    delta=0,
    debug=False,
    img_array=None,
):
    preamble = preamble_bits()
    preamble_len = len(preamble)
    total_bits = (preamble_len + watermark_length * 8) * repeat
    # 支援直接傳入 numpy array
    if img_array is not None:
        img = img_array
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
    # 依序嘗試8種方向，尋找正確前導碼
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
    direction_names = [
        '原圖', '水平翻轉', '垂直翻轉', '水平+垂直翻轉',
        '90度旋轉', '270度旋轉', '90度+水平翻轉', '270度+水平翻轉'
    ]
    for i, img_dir in enumerate(imgs):
        bits = try_extract(img_dir, total_bits, preamble_len, block_size, coef_base, redundancy, repeat, delta, debug=debug)
        if bits is not None:
            watermark = bits_to_str(bits)
            print(f"Extracted watermark ({direction_names[i]}): {watermark}")
            return watermark
    print("Extracted watermark: (未找到正確前導碼，無法還原)")
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python watermark_extract.py input.png watermark_length")
    else:
        extract_watermark(sys.argv[1], int(sys.argv[2]), debug=True) 