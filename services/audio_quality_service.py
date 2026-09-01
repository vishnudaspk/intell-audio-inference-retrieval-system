"""
Audio Quality Analysis Service.
Stateless analyzer for RMS energy, clipping detection, dynamic range, and SNR estimation.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import soundfile as sf

from schemas.analysis import AudioQuality
from utils.logger import logger


class AudioQualityService:
    """Service providing diagnostics on input audio quality."""

    def analyze(self, wav_path: Path, non_speech_intervals: Optional[List[tuple]] = None) -> AudioQuality:
        """
        Analyze audio quality metrics for a WAV file.
        """
        try:
            data, sr = sf.read(str(wav_path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # 1. RMS Energy & Peak Amplitude
            rms = float(np.sqrt(np.mean(data**2)))
            peak_amp = float(np.max(np.abs(data))) if len(data) > 0 else 0.0

            # 2. Clipping detection (> 0.999 amplitude)
            clipping_samples = int(np.sum(np.abs(data) >= 0.999))
            clipping_detected = clipping_samples > 0

            # 3. Dynamic range in dB & Noise floor
            nonzero = np.abs(data[np.abs(data) > 1e-6])
            if len(nonzero) > 0:
                p_max = np.percentile(nonzero, 99.5)
                p_min = np.percentile(nonzero, 5)
                dynamic_range_db = float(20 * np.log10(p_max / max(1e-6, p_min)))
                noise_floor_db = float(20 * np.log10(max(1e-6, p_min)))
            else:
                dynamic_range_db = 0.0
                noise_floor_db = -60.0

            # 4. Zero crossing rate & Spectral centroid approximation
            zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
            zcr = float(zero_crossings / max(1, len(data)))
            
            # Simple FFT spectral centroid
            fft_mag = np.abs(np.fft.rfft(data[:min(len(data), sr * 5)]))
            freqs = np.fft.rfftfreq(min(len(data), sr * 5), 1.0 / sr)
            spectral_centroid_hz = float(np.sum(freqs * fft_mag) / max(1e-6, np.sum(fft_mag)))

            # 5. SNR estimation if non-speech noise intervals provided
            snr_db = None
            if non_speech_intervals and len(non_speech_intervals) > 0:
                noise_samples = []
                for st, et in non_speech_intervals:
                    s_idx = int(st * sr)
                    e_idx = int(et * sr)
                    if e_idx > s_idx:
                        noise_samples.append(data[s_idx:e_idx])
                if noise_samples:
                    noise_concat = np.concatenate(noise_samples)
                    noise_rms = float(np.sqrt(np.mean(noise_concat**2))) if len(noise_concat) > 0 else 1e-6
                    if noise_rms > 0 and rms > 0:
                        snr_db = float(20 * np.log10(rms / noise_rms))

            # 6. Audio Quality Score (0 - 100)
            score = 100.0
            if clipping_detected:
                score -= 15.0
            if rms < 0.02:
                score -= 10.0
            if snr_db is not None and snr_db < 15.0:
                score -= (15.0 - snr_db) * 1.5
            audio_score = max(10.0, min(100.0, score))

            # 7. Warnings list
            warnings: List[str] = []
            if clipping_detected:
                warnings.append(f"Audio clipping detected ({clipping_samples} saturated samples).")
            if rms < 0.01:
                warnings.append("Low overall signal energy (audio may be quiet or muffled).")
            if snr_db is not None and snr_db < 10.0:
                warnings.append("Low SNR estimated (< 10 dB) — background noise present.")

            return AudioQuality(
                rms_energy=round(rms, 4),
                clipping_detected=clipping_detected,
                dynamic_range_db=round(dynamic_range_db, 2),
                snr_estimate_db=round(snr_db, 2) if snr_db is not None else None,
                peak_amplitude=round(peak_amp, 4),
                noise_floor_db=round(noise_floor_db, 1),
                spectral_centroid_hz=round(spectral_centroid_hz, 1),
                zero_crossing_rate=round(zcr, 4),
                audio_quality_score=round(audio_score, 1),
                warnings=warnings,
            )
        except Exception as exc:
            logger.warning(f"AudioQualityService analysis failed: {exc}")
            return AudioQuality(warnings=[f"Analysis warning: {exc}"])
