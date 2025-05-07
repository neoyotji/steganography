import cv2
import numpy as np

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(b, 2)) for b in chars)

def complexity(block):
    """Basit bir karmaşıklık ölçütü: bit değişimi sayısı / toplam bit"""
    transitions = np.sum(block[:, :-1] != block[:, 1:]) + np.sum(block[:-1, :] != block[1:, :])
    return transitions / (block.shape[0] * block.shape[1] * 2)

def embed_message_in_frame(frame, message_bits, threshold=0.3):
    h, w, _ = frame.shape
    bit_index = 0
    for channel in range(3):  # R, G, B kanalları
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = frame[i:i+8, j:j+8, channel]
                for bit_plane in range(8):  # 8 bit düzlemi
                    plane = ((block >> bit_plane) & 1).astype(np.uint8)
                    if complexity(plane) > threshold:
                        if bit_index + 64 > len(message_bits):
                            return frame  # Tüm mesaj gömüldü
                        message_block = np.array([int(b) for b in message_bits[bit_index:bit_index+64]]).reshape((8, 8))
                        block = block & ~(1 << bit_plane)  # O düzlemdeki bitleri sıfırla
                        block = block | (message_block << bit_plane)  # Yeni bitleri ekle
                        frame[i:i+8, j:j+8, channel] = block
                        bit_index += 64
    return frame

def extract_message_from_frame(frame, num_bits, threshold=0.3):
    h, w, _ = frame.shape
    bits = ""
    for channel in range(3):
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = frame[i:i+8, j:j+8, channel]
                for bit_plane in range(8):
                    plane = ((block >> bit_plane) & 1).astype(np.uint8)
                    if complexity(plane) > threshold:
                        bits += ''.join(str(b) for b in plane.flatten())
                        if len(bits) >= num_bits:
                            return bits[:num_bits]
    return bits

def embed_message_in_video(input_path, output_path, message):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    message_bits = text_to_bits(message) + '1111111111111110'  # End mark
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
    print(f"Mesaj başarıyla video içerisine gömüldü: {output_path}")

def extract_message_from_video(path, num_chars):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    if not ret:
        print("Video okunamadı.")
        return ""

    bits = extract_message_from_frame(frame, num_chars * 8 + 16)
    bits = bits.replace("1111111111111110", "")
    message = bits_to_text(bits)
    return message


# Mesajı göm
embed_message_in_video("input.mp4", "output_bpcs.avi", "BPCS algoritması ile mesaj videoya gömüldü. Toplamda 160 karakter gizlenebilir.")

# Mesajı çıkar
msg = extract_message_from_video("output_bpcs.avi", 160)
print("Gizlenmiş mesaj:", msg)
