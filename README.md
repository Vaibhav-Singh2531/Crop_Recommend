# Crop Disease Detection API

This repository contains a FastAPI application that serves a deep learning model for classifying crop diseases from images. The model can identify 17 different conditions across various crops including Corn, Potato, Rice, Sugarcane, and Wheat.

## Model Details

The classification model is a `rexnet_150` architecture from the `timm` library, pre-trained and fine-tuned for this specific task. The model is loaded from the `crop_best_model.pth` file.

The model can predict the following 17 classes:
- `Corn___Common_Rust`
- `Corn___Gray_Leaf_Spot`
- `Corn___Healthy`
- `Corn___Northern_Leaf_Blight`
- `Potato___Early_Blight`
- `Potato___Healthy`
- `Potato___Late_Blight`
- `Rice___Brown_Spot`
- `Rice___Healthy`
- `Rice___Leaf_Blast`
- `Rice___Neck_Blast`
- `Sugarcane_Bacterial Blight`
- `Sugarcane_Healthy`
- `Sugarcane_Red Rot`
- `Wheat___Brown_Rust`
- `Wheat___Healthy`
- `Wheat___Yellow_Rust`

## API Endpoints

The application exposes the following endpoints:

- **`GET /`**: A home endpoint to confirm the API is running.
- **`GET /health`**: A health check endpoint that returns `{"status": "ok"}`.
- **`POST /predict`**: The main prediction endpoint. It accepts an image file and returns the predicted disease and a confidence score.

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Vaibhav-Singh2531/Crop_Recommend.git
    cd Crop_Recommend
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To start the FastAPI server, run the following command in your terminal:

```sh
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## How to Use

You can send a `POST` request with an image file to the `/predict` endpoint to get a prediction.

### Example using `curl`

```sh
curl -X POST -F "file=@/path/to/your/crop_image.jpg" http://127.0.0.1:8000/predict
```

### Example Response

A successful request will return a JSON object with the predicted class and the model's confidence:

```json
{
  "prediction": "Potato___Late_Blight",
  "confidence": 0.9985123872756958
}
```

In case of an error, the API will return a JSON object with an error message:

```json
{
  "error": "Details about the error."
}