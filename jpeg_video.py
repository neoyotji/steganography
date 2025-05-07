import cv2
import numpy as np

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(b, 2)) for b in chars)

def embed_message_in_frame(frame, message_bits):
    h, w, _ = frame.shape
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(frame_ycrcb)

    # 8x8 bloklara ayır ve DCT uygula
    bit_index = 0
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = np.float32(y[i:i+8, j:j+8])
            dct_block = cv2.dct(block)

            # DCT bloğunun (1,1) veya (0,1) gibi düşük frekans katsayısı alınır
            coeff = dct_block[4, 4]
            coeff_int = int(coeff)
            if bit_index < len(message_bits):
                coeff_int = (coeff_int & ~1) | int(message_bits[bit_index])
                bit_index += 1
            dct_block[4, 4] = coeff_int

            idct_block = cv2.idct(dct_block)
            y[i:i+8, j:j+8] = np.uint8(np.clip(idct_block, 0, 255))

            if bit_index >= len(message_bits):
                break
        if bit_index >= len(message_bits):
            break

    merged = cv2.merge([y, cr, cb])
    result_frame = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return result_frame

def extract_message_from_frame(frame, num_bits):
    h, w, _ = frame.shape
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(frame_ycrcb)

    bits = ""
    bit_index = 0
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = np.float32(y[i:i+8, j:j+8])
            dct_block = cv2.dct(block)
            coeff = int(dct_block[4, 4])
            bits += str(coeff & 1)
            bit_index += 1
            if bit_index >= num_bits:
                return bits
    return bits

def embed_message_in_video(input_video, output_video, message):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    message_bits = text_to_bits(message) + '1111111111111110'  # end mark
    embedded = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not embedded:
            frame = embed_message_in_frame(frame, message_bits)
            embedded = True
        out.write(frame)
    cap.release()
    out.release()
    print(f"Mesaj video içerisine gömüldü: {output_video}")

def extract_message_from_video(video_path, num_chars):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Video okunamadı.")
        return

    num_bits = num_chars * 8 + 16  # +16 end mark için
    bits = extract_message_from_frame(frame, num_bits)
    bits = bits.replace("1111111111111110", "")
    message = bits_to_text(bits)
    return message

# 160 karakterlik mesajı video içine göm
embed_message_in_video("input_video.mp4", "video_with_message.avi", "Bu bir test mesajıdır. JPEG algoritmasına benzer DCT yapısıyla bu mesaj video karesine gömülmüştür.")

# Videodan mesajı çıkar
msg = extract_message_from_video("video_with_message.avi", 160)
print("Çözülen Mesaj:", msg)

