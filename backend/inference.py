import torch
from models import MultimodelSentimentModel
import os
import cv2
import numpy as np
import subprocess
import soundfile as sf
import torchaudio
import whisper
from transformers import AutoTokenizer

# Mapping for predictions
EMOTION_MAP = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'joy',
    4: 'neutral',
    5: 'sadness',
    6: 'surprise'
}
SENTIMENT_MAP = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

# ------------------------- Video Processing -------------------------
class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            raise ValueError(f"Video file {video_path} could not be opened.")

        while len(frames) < 30 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video {video_path} has no frames.")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

# ------------------------- Audio Processing -------------------------
class AudioProcessor:
    def extract_features(self, video_path, max_len=300):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(waveform).unsqueeze(0)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            if mel_spec.size(2) < max_len:
                padding = max_len - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_len]

            return mel_spec

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

# ------------------------- Segment Extraction -------------------------
class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir='./temp'):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed: " + segment_path)

        return segment_path

# ------------------------- Model Load -------------------------
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = MultimodelSentimentModel().to(device)

    # ✅ Fix model file name to match your checkpoint
    model_path = os.path.join(model_dir, 'model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in path: {model_path}")

    print("Loading model from path:", model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model('base', device=device),
        'device': device,
    }

# ------------------------- Prediction -------------------------
def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    result = model_dict['transcriber'].transcribe(video_path, word_timestamps=True)
    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result['segments']:
        try:
            segment_path = utterance_processor.extract_segment(video_path, segment['start'], segment['end'])

            video_frames = utterance_processor.video_processor.process_video(segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)

            text_inputs = tokenizer(
                segment['text'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotions_probs = torch.softmax(outputs['emotions'], dim=1)[0]
                sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

                emotions_values, emotions_indices = torch.topk(emotions_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            predictions.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': segment['text'],
                'emotions': [
                    {'label': EMOTION_MAP[idx.item()], 'confidence': conf.item()}
                    for idx, conf in zip(emotions_indices, emotions_values)
                ],
                'sentiments': [
                    {'label': SENTIMENT_MAP[idx.item()], 'confidence': conf.item()}
                    for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })

        except Exception as e:
            print("Segment failed inference:", str(e))

        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)

    return {"utterances": predictions}

# ------------------------- Local Runner -------------------------
def process_local_video(video_path, model_dir="./"):
    model_dict = model_fn(model_dir)
    input_data = {'video_path': video_path}
    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions['utterances']:
        print(f"\nUtterance: {utterance['text']}")
        print(f"Start: {utterance['start_time']}s, End: {utterance['end_time']}s")
        print("\nTop Emotions:")
        for emotion in utterance['emotions']:
            print(f"{emotion['label']}: {emotion['confidence']:.2f}")
        print("\nTop Sentiments:")
        for sentiment in utterance['sentiments']:
            print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")
        print("-" * 50)


if __name__ == '__main__':
    process_local_video("./dia0_utt0.mp4", model_dir="./model")
