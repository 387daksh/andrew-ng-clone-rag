import os
import tempfile
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent / "database" / "voices"
REFERENCE_PATH = REFERENCE_DIR / "my_voice.wav"


def save_reference_voice(audio_bytes):
    if not audio_bytes:
        raise ValueError("The reference recording is empty.")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_bytes(audio_bytes)
    return REFERENCE_PATH


def has_reference_voice():
    return REFERENCE_PATH.exists() and REFERENCE_PATH.stat().st_size > 0


def delete_reference_voice():
    if REFERENCE_PATH.exists():
        REFERENCE_PATH.unlink()


def load_chatterbox_model():
    try:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as exc:
        raise RuntimeError(
            "Voice cloning is not installed. Install requirements-voice.txt first."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ChatterboxTurboTTS.from_pretrained(device=device)


def clone_voice_to_wav_bytes(model, text, reference_path=REFERENCE_PATH):
    if not str(text).strip():
        raise ValueError("No text to speak.")
    if not Path(reference_path).exists():
        raise FileNotFoundError("Record and save your reference voice first.")

    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for cloned voice output.") from exc

    try:
        ref_wav, ref_sr = torchaudio.load(str(reference_path))
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)
        if ref_sr != model.sr:
            import torchaudio.transforms as T
            resampler = T.Resample(ref_sr, model.sr)
            ref_wav = resampler(ref_wav)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
            temp_ref_path = tmp_ref.name
        torchaudio.save(temp_ref_path, ref_wav, model.sr)
    except Exception as exc:
        raise RuntimeError(f"Failed to process reference audio: {exc}")

    try:
        wav = model.generate(
            str(text).strip(),
            audio_prompt_path=temp_ref_path,
        )
    finally:
        try:
            os.remove(temp_ref_path)
        except OSError:
            pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    try:
        torchaudio.save(output_path, wav.detach().cpu(), model.sr)
        return Path(output_path).read_bytes()
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
