import librosa
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip

class AudioProcessor:
    def __init__(self):
        self.sr = 22050  # Sample rate
        self.hop_length = 512  # Number of samples between successive frames
        
    def detect_beats(self, audio_path):
        """
        Detect beats in an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            list: Beat timestamps in seconds
        """
        try:
            # Validate file exists
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found: {audio_path}")
            
            # Load the audio file with fallback to moviepy for problematic formats
            try:
                y, sr = librosa.load(audio_path, sr=self.sr)
            except Exception as load_error:
                # Fallback: use moviepy to convert audio first
                from moviepy.editor import AudioFileClip
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                try:
                    audio_clip = AudioFileClip(audio_path)
                    audio_clip.write_audiofile(temp_wav_path, verbose=False, logger=None)
                    y, sr = librosa.load(temp_wav_path, sr=self.sr)
                    audio_clip.close()
                finally:
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
            
            # Get tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
            
            return {
                "tempo": tempo,
                "beat_times": beat_times.tolist()
            }
            
        except Exception as e:
            raise Exception(f"Error processing audio '{os.path.basename(audio_path)}': {str(e)}")
    
    def extract_and_analyze_audio(self, video_path):
        """
        Extract audio from video and analyze it for beats.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            dict: Beat and tempo information
        """
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
            # Extract audio from video
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Analyze the extracted audio
            beat_data = self.detect_beats(temp_audio_path)
            
            # Clean up
            os.unlink(temp_audio_path)
            
            return beat_data
            
        except Exception as e:
            raise Exception(f"Error extracting audio from video: {str(e)}")
            
    def get_audio_duration(self, audio_path):
        """Get the duration of an audio file in seconds."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            return librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            raise Exception(f"Error getting audio duration: {str(e)}")
            
    def trim_audio(self, audio_path, target_duration):
        """
        Trim audio to target duration.
        
        Args:
            audio_path (str): Path to the audio file
            target_duration (float): Target duration in seconds
            
        Returns:
            numpy.ndarray: Trimmed audio data
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Calculate number of samples for target duration
            target_samples = int(target_duration * sr)
            
            if len(y) > target_samples:
                # Trim audio
                y = y[:target_samples]
            elif len(y) < target_samples:
                # Pad with silence if audio is too short
                padding = target_samples - len(y)
                y = np.pad(y, (0, padding), mode='constant')
                
            return y, sr
            
        except Exception as e:
            raise Exception(f"Error trimming audio: {str(e)}")