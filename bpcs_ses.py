import numpy as np
from scipy.io import wavfile

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(b, 2)) for b in chars)

def complexity(bit_plane):
    bit_plane = np.array([int(b) for b in bit_plane])
    transitions = np.sum(bit_plane[:-1] != bit_plane[1:])
    return transitions / (len(bit_plane) - 1)

def embed_message_in_audio(input_wav, output_wav, message, threshold=0.3):
    rate, data = wavfile.read(input_wav)

    # Mono veya stereo'yu destekler
    if data.ndim == 2:
        data = data[:, 0]

    if data.dtype != np.int16:
        raise ValueError("Yalnızca 16-bit PCM WAV desteklenir.")

    flat_data = data.copy()
    message_bits = text_to_bits(message) + '1111111111111110'  # End mark
    bit_index = 0

    for i in range(len(flat_data)):
        sample = flat_data[i]
        for bit_plane in range(8):
            bits = format(sample, '016b')
            target_bit = bits[15 - bit_plane]
            plane_bits = [target_bit]

            if complexity(plane_bits) > threshold:
                if bit_index >= len(message_bits):
                    wavfile.write(output_wav, rate, flat_data)
                    print("Mesaj başarıyla gömüldü.")
                    return

                new_bit = message_bits[bit_index]
                sample_bits = list(format(sample, '016b'))
                sample_bits[15 - bit_plane] = new_bit
                flat_data[i] = int(''.join(sample_bits), 2)
                bit_index += 1

    wavfile.write(output_wav, rate, flat_data)
    print("Mesaj başarıyla gömüldü.")

def extract_message_from_audio(wav_path, max_chars=160):
    rate, data = wavfile.read(wav_path)

    if data.ndim == 2:
        data = data[:, 0]

    flat_data = data.copy()
    bits = ""

    for i in range(len(flat_data)):
        sample = flat_data[i]
        for bit_plane in range(8):
            bits += format(sample, '016b')[15 - bit_plane]
            if "1111111111111110" in bits:
                bits = bits.replace("1111111111111110", "")
                return bits_to_text(bits[:max_chars * 8])
    return bits_to_text(bits[:max_chars * 8])


# Gömme işlemi
embed_message_in_audio("input.wav", "output_bpcs.wav", "BPCS ile ses dosyasına gizlenen 160 karakterlik gizli mesaj örneği.")

# Çıkarma işlemi
msg = extract_message_from_audio("output_bpcs.wav")
print("Gizlenmiş mesaj:", msg)
