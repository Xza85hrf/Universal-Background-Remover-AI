# Universal-Background-Remover-AI
Universal Background Remover is a Python application that allows users to remove backgrounds from images using advanced AI models for object detection and image segmentation. It provides a graphical user interface for easy interaction and supports various selection methods for defining regions of interest (ROI).

# Features

- GUI for easy interaction and image processing
- Support for multiple image loading and batch processing
- Various ROI selection methods: Full image, Rectangle, Oval, Freehand, and AI-assisted
- Customizable background color
- Real-time image preview
- Progress tracking for batch processing
- Logging of processing steps and errors
- Themeable interface (light/dark mode)
- Customizable AI models for object detection and segmentation

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Xza85hrf/universal-background-remover.git
   cd universal-background-remover

2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

# Usage

1. Run the application:
   ```bash
   python gui.py
   
2. Use the GUI to:
- Load images
- Select ROI method
- Choose background color
- Process images
- Save processed images

# File Structure

- gui.py: Main application file with GUI implementation
- universal_background_remover.py: Core processing functions for background removal
- cache/: Directory for storing temporary files and model caches
- Input/: Directory for input images
- Output/: Directory for processed images and older versions
- processing.log: Log file for processing details

# Dependencies

- tkinter
- PIL (Pillow)
- OpenCV (cv2)
- NumPy
- torch
- torchvision
- transformers
- sv_ttk

# Customization

- You can customize the object detection and segmentation models by updating the model names in the GUI.
- The application supports various ROI selection methods, including AI-assisted detection.

# Contributing
Contributions to the Universal Background Remover project are welcome. Please feel free to submit pull requests, report bugs, or suggest features.

# License
MIT License

# Acknowledgments
This project uses several open-source libraries and pre-trained models. Please see the requirements.txt file for a full list of dependencies.
