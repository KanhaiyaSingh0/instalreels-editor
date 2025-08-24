# 🎬 AI-Powered Instalreels Editor

The most advanced AI-powered video editing tool for creating Instagram Reels! Just describe how you want your video edited, and let artificial intelligence do the magic.

## ✨ Features

### 🧠 AI-Powered Editing
- **Natural Language Instructions**: Tell the AI how you want your video edited in plain English
- **Intelligent Content Analysis**: AI understands your photos and videos to make smart editing decisions
- **Visual Content Recognition**: Detects faces, objects, mood, and scene types
- **Smart Clip Ordering**: Optimally arranges your media based on content and style

### 🎵 Advanced Audio Processing
- **Beat Synchronization**: Automatically syncs cuts to music beats
- **Intelligent Audio Mixing**: Smart volume balancing and audio enhancement
- **Music-driven Editing**: Creates videos that flow with your soundtrack

### 🎨 Professional Video Creation
- **Multiple Editing Styles**: Energetic, Aesthetic, Cinematic, Promotional, and more
- **Smart Transitions**: AI selects the best transitions based on content
- **Dynamic Effects**: Ken Burns effect, speed ramping, color grading
- **Text Overlays**: Intelligent text placement and styling

### 🔄 Fallback System
- **Traditional Mode**: Works without AI for basic editing
- **Hybrid Intelligence**: Combines AI with computer vision for robust results
- **Error Recovery**: Graceful handling of failed operations

## 🚀 Quick Start

### Option 1: AI-Powered Setup (Recommended)

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd instalreels-editor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup AI (One-time)**:
```bash
python setup_ai.py
```
This will automatically install and configure Ollama with required AI models.

3. **Start the Editor**:
```bash
# Start Ollama service (in one terminal)
ollama serve

# Start the editor (in another terminal)
python app.py
```

### Option 2: Traditional Mode (No AI)

If you prefer to use without AI features:
```bash
pip install -r requirements.txt
python app.py
```

## 🎯 How to Use

### AI Mode (Recommended)
1. **Describe Your Video**: Use natural language like:
   - "Create an energetic gym workout video with fast cuts that sync to music"
   - "Make a calm aesthetic slideshow with smooth fades"
   - "Create a cinematic storytelling video with dramatic transitions"

2. **Upload Your Media**: Add your photos/videos and optional music

3. **Let AI Work**: The system will analyze your content and create a professional video

### Example AI Instructions
```
🏃‍♂️ Energetic: "Create an energetic video with fast cuts and beat sync"
🌸 Aesthetic: "Make a calm aesthetic video with smooth fades and longer transitions"
🎬 Cinematic: "Create a storytelling video with dramatic transitions and text overlay"
📱 Promotional: "Make a promotional video with professional cuts and branding"
🎵 Beat Sync: "Sync all cuts perfectly to the music beats for maximum impact"
```

## 🏗️ Architecture

```
🧠 User Instruction → AI Parser → Content Analyzer → Intelligent Editor → Final Video
                                      ↓
📱 Media Files → Visual AI → Scene Understanding → Smart Composition
                                      ↓
🎵 Audio File → Beat Detection → Rhythm Analysis → Synchronized Cuts
```

## 🛠️ Tech Stack

### AI & Machine Learning
- **Ollama**: Local LLM for instruction parsing
- **LLaVA**: Vision-Language Model for content understanding
- **OpenCV**: Computer vision for media analysis
- **Librosa**: Audio processing and beat detection

### Video Processing
- **MoviePy**: Professional video editing
- **FFmpeg**: Video codec and format handling
- **PIL/Pillow**: Advanced image processing
- **NumPy**: Numerical computations

### Web Interface
- **Gradio**: Modern, intuitive web UI
- **Real-time Progress**: Live updates during processing

## 📁 File Structure

```
instalreels-editor/
├── 🎬 app.py                          # Main application
├── 🧠 setup_ai.py                     # AI setup script
├── 📋 requirements.txt                # Dependencies
├── src/
│   ├── 🤖 ai_instruction_parser.py    # Natural language processing
│   ├── 👁️ visual_content_analyzer.py  # Visual AI analysis
│   ├── 🎬 intelligent_video_engine.py # AI video creation
│   ├── 📹 video_analyzer.py           # Traditional video analysis
│   ├── 🎵 audio_processor.py          # Audio processing
│   └── 🎞️ video_generator.py          # Video generation
├── input/                             # Input files
│   ├── media/                        # Your photos/videos
│   ├── music/                        # Music files
│   └── reference/                    # Reference videos
└── output/                           # Generated videos
```

## 🎨 Editing Styles

| Style | Description | Best For |
|-------|-------------|----------|
| **Energetic** | Fast cuts, dynamic transitions, beat sync | Fitness, sports, dance videos |
| **Aesthetic** | Smooth fades, longer clips, artistic flow | Lifestyle, beauty, travel content |
| **Cinematic** | Dramatic transitions, storytelling focus | Narrative content, documentaries |
| **Promotional** | Professional cuts, text overlays | Business, product showcases |
| **Beat Sync** | Perfect music synchronization | Music videos, dance content |

## 🔧 Advanced Configuration

### AI Models
- **Text Model**: `llama3.1` (for instruction parsing)
- **Vision Model**: `llava` (for visual analysis)
- **Fallback**: Traditional CV methods when AI unavailable

### Performance Tuning
```python
# In src/intelligent_video_engine.py
self.target_size = (1080, 1920)  # Instagram Reels format
self.min_clip_duration = 1.0     # Minimum clip length
self.max_clip_duration = 6.0     # Maximum clip length
```

## 📈 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- FFmpeg installed

### Recommended
- Python 3.9+
- 16GB RAM
- GPU with 4GB VRAM (for AI acceleration)
- SSD storage
- Fast internet (for AI model downloads)

## 🐛 Troubleshooting

### AI Issues
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve

# Test AI connectivity
curl http://localhost:11434/api/tags
```

### Common Solutions
- **Slow processing**: Reduce media count or resolution
- **AI not working**: Run `python setup_ai.py` again
- **Memory errors**: Close other applications, use fewer files
- **Quality issues**: Use higher quality input media

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- 🧠 **AI Improvements**: Better instruction parsing, new editing styles
- 🎨 **Effects**: New transition types, visual effects
- 🎵 **Audio**: Advanced beat detection, audio effects
- 🌐 **UI/UX**: Better interface, mobile support
- 📚 **Documentation**: Tutorials, examples

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama**: For making local AI accessible
- **MoviePy**: For powerful video editing capabilities
- **Gradio**: For the beautiful web interface
- **LLaVA**: For vision-language understanding

---

🎬 **Ready to create amazing videos with AI? Get started now!** 🚀