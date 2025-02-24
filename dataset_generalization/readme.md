# Dataset Transformation Script (`transform.py`)

Given the limited availability of offroad tree datasets, this Python script, `transform.py`, empowers users to create their own diverse dataset of tree images starting from a small set of original images. By transforming these images into four distinct artistic styles—**photo** (original images), **sketch**, **art_painting** (oil painting effect), and **cartoons**—this script provides a powerful tool for expanding datasets. Whether you're working on domain adaptation, style transfer, or other computer vision projects, this script helps you generate a rich and varied dataset tailored to your needs.

The script uses popular libraries such as OpenCV, NumPy, scikit-image, SciPy, and TensorFlow, with optional GPU support to accelerate certain transformations, making it efficient and adaptable.

---

## Purpose of the Script

- **Expand Limited Data**: If you only have a handful of tree images, this script multiplies their utility by creating artistic variations.
- **Artistic Diversity**: It transforms images into realistic sketches, oil paintings, and cartoons, opening up possibilities for style-related machine learning tasks.
- **Speed and Efficiency**: With GPU support for the sketch transformation, processing is fast and scalable.
- **Flexibility**: Users can tweak class distributions or transformation settings to match their specific project requirements.

---

## Prerequisites

Before running the script, ensure you have the following:

- **Python 3.x**: The script is written in Python and requires a compatible version.
- **Required Libraries**:
  - `opencv-python`: For image processing.
  - `numpy`: For numerical operations.
  - `scikit-image`: For advanced image manipulation.
  - `matplotlib`: For visualization (optional, depending on usage).
  - `scipy`: For scientific computations.
  - `tensorflow`: For GPU-accelerated transformations.
- **Optional**: A GPU with CUDA support for faster sketch processing.

Install the dependencies with pip:

```bash
pip install opencv-python numpy scikit-image matplotlib scipy tensorflow
