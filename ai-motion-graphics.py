
import cv2
import numpy as np
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip
import librosa
from concurrent.futures import ThreadPoolExecutor

#image Processing
def load_images(image_folder):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (1280, 720))  # Resize to 720p for faster processing
            images.append(img)
    return images

def generate_transition_frames(img1, img2, steps=10):
    frames = []
    for alpha in np.linspace(0, 1, steps):
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

def generate_transitions(images, steps=10):
    all_frames = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_transition_frames, images[i], images[i+1], steps) 
                   for i in range(len(images) - 1)]
        for future in futures:
            all_frames.extend(future.result())
    return all_frames

#audio Integration
def detect_beats(audio_path):
    y, sr = librosa.load(audio_path)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times

def synchronize_with_music(transition_frames, beat_times, fps=24):
    sync_frames = []
    num_frames = len(transition_frames)
    beat_indices = [int(t * fps) for t in beat_times]
    beat_indices = [min(bf, num_frames - 1) for bf in beat_indices]
    
    for i in range(len(beat_indices) - 1):
        sync_frames.extend(transition_frames[beat_indices[i]:beat_indices[i + 1]])
    
    return sync_frames

#main video creation function
def create_video(images, audio_path, output_video_path, fps=24):
    transition_frames = generate_transitions(images, steps=5)
    
    beat_times = detect_beats(audio_path)
    
    sync_frames = synchronize_with_music(transition_frames, beat_times, fps)
    
    #Save video with audio
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sync_frames], fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final_clip = clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264")

if __name__ == "__main__":
    images = load_images("images")
    audio_path = "audio\music.mp3"
    output_video_path = "output_video.mp4"
    
    create_video(images, audio_path, output_video_path, fps=24)
