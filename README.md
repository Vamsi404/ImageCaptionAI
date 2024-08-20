Here's a README file that describes the two separate code components: the image captioning model and the Flask application that uses this model.

---

# ImageCaptionAI

**ImageCaptionAI** is a comprehensive project for generating image captions using deep learning. It consists of two main components:

1. **Image Captioning Model**: A deep learning model trained to generate captions for images.
2. **Flask Application**: A web application that serves the model for real-time image captioning.

## 📁 Project Overview

### 1. Image Captioning Model

This component includes the code to preprocess images, train the model, and generate captions. It utilizes:

- **ResNet50** for extracting high-level image features.
- **LSTM** for generating captions from these features.
- **GloVe Embeddings** to enhance caption accuracy.

**Key Files:**

- `image_captioning_model.py`: Contains code for preprocessing images, encoding them, and generating captions.
- `train_model.py`: Script to train the captioning model.
- `data_preprocessing.py`: Handles data cleaning and tokenization for training.

**Main Functions:**

- `preprocess_img(img)`: Preprocesses the image for ResNet50.
- `encode_image(img)`: Extracts features from the image using ResNet50.
- `predict_caption(photo)`: Generates a caption for the image features using the trained LSTM model.

### 2. Flask Application

The Flask application serves as an interface to interact with the image captioning model. Users can upload images through a web interface, and the server will respond with generated captions.

**Key Files:**

- `app.py`: The main Flask application file that handles image uploads and returns generated captions.
- `requirements.txt`: Lists all necessary Python dependencies for running the Flask app.

**Main Functions:**

- `predict()`: Endpoint that accepts image uploads, processes the image, and returns the generated caption.

## 🛠️ Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- Flask
- Other dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vamsi404/ImageCaptionAI.git
   cd ImageCaptionAI
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset and embeddings:**
   - Flickr8k dataset
   - GloVe embeddings (50-dimensional)

   Place these files in the appropriate directories (`/data` and `/glove`).

### Usage

#### 1. Image Captioning Model

To use the model directly, run the following script to generate a caption for an image:

```bash
python image_captioning_model.py --image_path /path/to/image.jpg
```

#### 2. Flask Application

1. **Start the Flask server:**

   ```bash
   python app.py
   ```

   By default, the server will be accessible at `http://127.0.0.1:5000/`.

2. **Upload an image through the web interface:**

   Navigate to `http://127.0.0.1:5000/` and use the provided form to upload an image. The server will return the generated caption.

## 🔗 Link Between Components

The **Image Captioning Model** and the **Flask Application** are linked as follows:

- **Model**: The trained image captioning model (`model_9.h5`) is loaded in the Flask application to generate captions for uploaded images.
- **Flask App**: Handles HTTP requests, processes the uploaded image, and uses the model's functions to generate and return captions.

**`app.py`** uses functions from **`image_captioning_model.py`** to process images and generate captions. The Flask server interacts with the model to provide real-time captioning through a web interface.

## 📁 Directory Structure

- `data/`: Contains datasets and processed files.
- `glove/`: Contains GloVe embedding files.
- `src/`: Source code for preprocessing, model training, and inference.
- `flask_app/`: Flask application code.
- `requirements.txt`: Python dependencies for the Flask app.

## 🤝 Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` for guidelines on how to contribute to the project.


## 📬 Contact

For questions or feedback, please open an issue or contact [mail](mandavamsi302001@gmail.com).
