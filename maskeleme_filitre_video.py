import cv2
import numpy as np

# Mesajı ikili formata çevir
def text_to_bits(text):
    return ''.join(format(ord(i), '08b') for i in text)

# İkili veriyi tekrar metne çevir
def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

# 8x8 blok için DCT uygula
def apply_dct(block):
    return cv2.dct(np.float32(block))

# Ters DCT uygula
def apply_idct(block):
    return cv2.idct(block)

# Mesajı videoya gömme
def embed_message_in_video(input_path, output_path, message):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))), False)

    message_bits = text_to_bits(message + "|END|")
    bit_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or bit_index >= len(message_bits):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if bit_index >= len(message_bits):
                    break
                block = gray[y:y+8, x:x+8]
                if block.shape[0] != 8 or block.shape[1] != 8:
                    continue

                dct_block = apply_dct(block)

                # Mesaj bitini DCT'nin yüksek frekans bileşenine göm
                bit = int(message_bits[bit_index])
                dct_block[7][6] = (dct_block[7][6] // 2) * 2 + bit  # LSB benzeri

                idct_block = apply_idct(dct_block)
                gray[y:y+8, x:x+8] = np.uint8(np.clip(idct_block, 0, 255))
                bit_index += 1

        out.write(gray)

    cap.release()
    out.release()
    print("Mesaj başarıyla videoya gömüldü.")

# Mesajı videodan çıkar
def extract_message_from_video(video_path, expected_length=160):
    cap = cv2.VideoCapture(video_path)
    bits = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        for y in range(0, h, 8):
            for x in range(0, w, 8):
                block = gray[y:y+8, x:x+8]
                if block.shape[0] != 8 or block.shape[1] != 8:
                    continue

                dct_block = apply_dct(block)
                bit = int(dct_block[7][6]) % 2
                bits += str(bit)

                if "|END|" in bits_to_text(bits):
                    cap.release()
                    return bits_to_text(bits).replace("|END|", "")

    cap.release()
    return bits_to_text(bits)


# Mesajı videoya göm
embed_message_in_video("input.avi", "output_stego.avi", "Maskeleme ve filtreleme ile video içine gizlenmiş mesaj örneğidir. " * 2)

# Videodan mesajı çöz
msg = extract_message_from_video("output_stego.avi")
print("Gizlenmiş mesaj:", msg)
