import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def embed_message(input_wav, output_wav, message, carrier_freq_start=15000, sample_rate_threshold=44100):
    rate, data = wavfile.read(input_wav)

    if rate < sample_rate_threshold:
        raise ValueError("Ses örnekleme hızı en az 44.1kHz olmalıdır (yüksek frekanslar için).")

    if data.ndim > 1:
        data = data[:, 0]  # mono yap

    data = data.astype(np.float32)
    N = len(data)
    freq_data = fft(data)

    message_bits = text_to_bits(message) + '1111111111111110'  # End marker
    bit_index = 0

    freq_resolution = rate / N
    start_index = int(carrier_freq_start / freq_resolution)

    for i in range(start_index, N // 2):
        if bit_index >= len(message_bits):
            break
        bit = int(message_bits[bit_index])
        mag = np.abs(freq_data[i])
        phase = np.angle(freq_data[i])

        if bit == 1:
            mag += 50  # 1 için frekansta hafif artış
        else:
            mag -= 50  # 0 için frekansta hafif azalış

        freq_data[i] = mag * np.exp(1j * phase)
        freq_data[-i] = np.conj(freq_data[i])  # Simetri

        bit_index += 1

    modified_signal = np.real(ifft(freq_data)).astype(np.int16)
    wavfile.write(output_wav, rate, modified_signal)
    print("Mesaj başarıyla gömüldü.")

def extract_message(wav_path, message_length=160, carrier_freq_start=15000):
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]

    data = data.astype(np.float32)
    N = len(data)
    freq_data = fft(data)
    bits = ""

    freq_resolution = rate / N
    start_index = int(carrier_freq_start / freq_resolution)

    for i in range(start_index, start_index + (message_length * 8) + 16):  # +16 end marker
        mag = np.abs(freq_data[i])
        bits += '1' if mag > 100 else '0'

        if "1111111111111110" in bits:
            bits = bits.replace("1111111111111110", "")
            return bits_to_text(bits[:message_length * 8])

    return bits_to_text(bits[:message_length * 8])


# 160 karakterlik mesajı ses dosyasına gizle
embed_message("input.wav", "masked_output.wav", "Maskeleme ve filtreleme yöntemiyle bu mesaj sesi bozmayacak şekilde gizlenmiştir.")

# Mesajı çıkar
message = extract_message("masked_output.wav", message_length=160)
print("Gizlenmiş mesaj:", message)
