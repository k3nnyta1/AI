import numpy as np
import cv2

# 將圖片分割為8x8區塊，回傳[(位置, 區塊)]清單
# 例如 512x512 圖片會分成4096個區塊
def img_to_blocks(img, block_size=8):
    h, w = img.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                blocks.append(((i, j), block))
    return blocks

# 將8x8區塊合併還原成圖片
# blocks: [(位置, 區塊)]
# img_shape: 原圖大小
def blocks_to_img(blocks, img_shape, block_size=8):
    img = np.zeros(img_shape, dtype=np.float32)
    for (i, j), block in blocks:
        img[i:i+block_size, j:j+block_size] = block
    return img

# 對8x8區塊做2D DCT（離散餘弦轉換）
def dct2(block):
    return cv2.dct(np.float32(block))

# 對8x8區塊做2D IDCT（反離散餘弦轉換）
def idct2(block):
    return cv2.idct(np.float32(block))

# 字串轉bit list（每字元8位元）
# 例如 'A' -> [0,1,0,0,0,0,0,1]
def str_to_bits(s):
    return [int(b) for c in s for b in format(ord(c), '08b')]

# bit list轉回字串
# 例如 [0,1,0,0,0,0,0,1] -> 'A'
def bits_to_str(bits):
    chars = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(str(bit) for bit in byte), 2)))
    return ''.join(chars)

# 將圖片resize成寬高皆為8的倍數，避免DCT分割出現殘缺區塊
def resize_to_multiple_of_8(img):
    h, w = img.shape[:2]
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) 