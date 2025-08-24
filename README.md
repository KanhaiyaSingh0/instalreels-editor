# 🎬 Instalreels Editor - AI-Powered Video Editor

A professional Instagram Reels video editor with AI-powered features, reference video style transfer, and NVIDIA Cloud AI integration.

## ✨ Features

### 🎯 **Core Video Editing**
- **Reference Video Style Transfer** - Generate videos that match your reference video's exact lighting, colors, and style
- **Multiple Output Versions** - Create energetic, cinematic, and natural versions automatically
- **Intelligent Scene Detection** - Automatic scene analysis and timing
- **Beat Synchronization** - Sync video cuts with music beats
- **Instagram Reels Format** - Optimized for 9:16 aspect ratio (1080x1920)

### 🤖 **AI-Powered Features**
- **NVIDIA Cloud AI Integration** - Advanced AI processing without local GPU requirements
- **Automatic Style Analysis** - Analyzes reference videos for visual characteristics
- **Smart Lighting Correction** - Automatic brightness, contrast, and saturation adjustments
- **Intelligent Content Understanding** - AI-powered media analysis and recommendations

### 🎨 **Advanced Processing**
- **Scene Replacement** - Replace parts of reference videos with your content
- **Multiple Style Presets** - Energetic, Cinematic, Natural, Vibrant, Moody
- **Custom Lighting Presets** - Create presets that exactly match reference videos
- **Professional Transitions** - Smooth fades, zooms, and effects

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg installed on your system
- NVIDIA API key (optional, for cloud AI features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/instalreels-editor.git
cd instalreels-editor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up NVIDIA API (optional)**
```bash
# Copy the example config
cp nvidia_config.env.example nvidia_config.env
# Edit nvidia_config.env and add your NVIDIA API key
```

### Basic Usage

1. **Run the enhanced app**
```bash
python enhanced_app.py
```

2. **Upload your content**
   - Reference video (for style matching)
   - Photos/videos to edit
   - Music file (optional)

3. **Generate videos**
   - Choose from multiple style versions
   - Let AI analyze your reference video
   - Get perfectly styled output videos

## 🔧 Advanced Usage

### Reference Video Style Transfer

The editor automatically analyzes your reference video and applies the exact same:
- **Lighting characteristics** (brightness, contrast)
- **Color palette** (saturation, warmth)
- **Visual mood** (energetic, cinematic, etc.)

```python
from src.advanced_video_processor import AdvancedVideoProcessor

processor = AdvancedVideoProcessor()

# Analyze reference video
ref_analysis = processor.analyze_reference_video("reference.mp4")

# Generate video with reference style
versions = processor.generate_multiple_versions(
    base_clips, 
    audio_path, 
    instruction, 
    reference_video_path="reference.mp4"
)
```

### Custom Style Presets

Create lighting presets that match your reference videos:

```python
# The system automatically creates custom presets
custom_preset = processor._create_reference_style_preset(ref_analysis)
# Returns: brightness, contrast, saturation, warmth multipliers
```

## 📁 Project Structure

```
instalreels-editor/
├── src/                          # Core source code
│   ├── advanced_video_processor.py    # Main video processing engine
│   ├── video_analyzer.py              # Video analysis and scene detection
│   ├── audio_processor.py             # Audio processing and beat detection
│   ├── video_generator.py             # Video generation and rendering
│   ├── nvidia_cloud_ai_engine.py      # NVIDIA AI integration
│   └── intelligent_video_engine.py    # AI-powered video editing
├── input/                        # Input media files
│   ├── media/                    # Photos and videos to edit
│   ├── music/                    # Music files
│   └── reference/                # Reference videos for style
├── output/                       # Generated video outputs
├── enhanced_app.py               # Main Gradio web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎨 Style Presets

### **Energetic Fitness** 🏋️‍♂️
- High brightness and saturation
- Vibrant, warm colors
- Perfect for gym and fitness content

### **Cinematic** 🎬
- Dramatic lighting and high contrast
- Rich, deep colors
- Professional film-like appearance

### **Natural** 🌿
- Balanced, realistic lighting
- Muted, natural colors
- Clean, professional look

### **Vibrant** ✨
- Ultra-bright and saturated
- High-energy aesthetics
- Perfect for social media

## 🔑 NVIDIA Cloud AI Features

### **What You Get**
- **No Local GPU Required** - Pure cloud processing
- **Advanced AI Models** - Llama 3.1, Vision models
- **Intelligent Content Analysis** - Automatic style detection
- **Professional Quality** - Enterprise-grade AI processing

### **Setup**
1. Get NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)
2. Add to `nvidia_config.env`
3. Restart the app

## 🐛 Troubleshooting

### Common Issues

**"OpenCV error: Unsupported depth"**
- Fixed in latest version
- Ensure you have the latest code

**"Beat detection failed"**
- Non-critical warning
- Videos will still generate successfully

**"NVIDIA API key invalid"**
- Check your API key in `nvidia_config.env`
- Verify NVIDIA service is available

### Performance Tips

- **Use SSD storage** for faster video processing
- **Close other applications** during video generation
- **Optimize input video size** (1080x1920 recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NVIDIA Build** for cloud AI capabilities
- **MoviePy** for video processing
- **OpenCV** for computer vision
- **Gradio** for the web interface

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/instalreels-editor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/instalreels-editor/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for content creators who want professional-quality videos without the complexity.**