import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct, idct

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join([chr(int(b, 2)) for b in chars])

def embed_message(audio_path, output_path, message):
    rate, data = wav.read(audio_path)
    
    if data.ndim > 1:
        data = data[:, 0]  # mono kanala indir
    
    data = data.astype(np.float32)
    audio_dct = dct(data, norm='ortho')

    bits = text_to_bits(message)
    bits += '1111111111111110'  # Son işaret (mesaj sonu belirteci)

    for i, bit in enumerate(bits):
        coeff = audio_dct[i]
        coeff = int(coeff)
        coeff = (coeff & ~1) | int(bit)
        audio_dct[i] = coeff

    modified_audio = idct(audio_dct, norm='ortho')
    modified_audio = np.clip(modified_audio, -32768, 32767)
    modified_audio = modified_audio.astype(np.int16)

    wav.write(output_path, rate, modified_audio)
    print(f"Mesaj başarıyla gizlendi. Kaydedilen dosya: {output_path}")

def extract_message(audio_path):
    rate, data = wav.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    data = data.astype(np.float32)
    audio_dct = dct(data, norm='ortho')

    bits = ""
    for coeff in audio_dct[:8000]:  # Maks 1000 karakter tarar
        coeff = int(coeff)
        bits += str(coeff & 1)
        if bits.endswith("1111111111111110"):
            break

    bits = bits.replace("1111111111111110", "")
    message = bits_to_text(bits)
    return message


# Gömme
embed_message("ornek.wav", "gizli_mesaj.wav", "Bu gizli bir mesajdır. 160 karaktere kadar desteklenir...")

# Çıkarma
mesaj = extract_message("gizli_mesaj.wav")
print("Çözülen Mesaj:", mesaj)
