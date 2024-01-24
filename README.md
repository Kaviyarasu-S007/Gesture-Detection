# Hand Gesture Recognition Model

## Overview

This project develops a hand gesture recognition model that accurately identifies and classifies different hand gestures from image or video data. The goal is to enable intuitive human-computer interaction and gesture-based control systems.

## Dataset

The dataset used for this project is the LeapGestRecog dataset, which can be found [here](https://www.kaggle.com/gti-upm/leapgestrecog).

The dataset consists of images of hand gestures recorded with the Leap Motion Controller. It includes different classes representing various hand gestures.

## Files

- `keras_model.py`: Trained model
- `data/`: Directory containing the dataset (please download and place the dataset here).
- `app.py`: Streamlit interface to test the model

## Usage

1. Download the LeapGestRecog dataset and place it in the `data/` directory.

2. Install the required packages using the command:

    ```bash
    pip install scikit-learn
    ```

3. Run the `app.py` script:

    ```bash
    streamlit run app.py
    ```

## Results

![image](https://github.com/Kaviyarasu-S007/Gesture-Detection/assets/151661034/1585ac63-c31a-40ee-bac4-fa076a4d8f11)


## Issues and Future Improvements

List any known issues or potential improvements for the hand gesture recognition model.

## Contributing

If you would like to contribute to the project, please open an issue or submit a pull request.
