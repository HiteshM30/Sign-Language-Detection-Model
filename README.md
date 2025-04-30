# ASL Sign Language Detection with Real-Time Sentence Building

This project uses **MediaPipe** for hand tracking and a trained machine learning model to detect American Sign Language (ASL) gestures in real-time using a webcam. The detected gestures are mapped to letters, and users can build sentences by changing gestures and using space gestures (wide open hand).

## Features

- Real-time detection of ASL gestures.
- Builds a sentence by tracking the ASL letters over time.
- **Space gesture**: A wide open hand is used to represent a space between words.
- **Backspace functionality** to remove the last character.
- **Save functionality**: Save the built sentence to a file.
- Supports dynamic and live sentence building while adjusting for hand position changes.

## Requirements

### Dependencies

- Python 3.x
- Install the required libraries:
    ```bash
    pip install opencv-python mediapipe numpy pickle
    ```

- **Model**: The machine learning model (`asl_model.pkl`) must be pre-trained on ASL gestures and stored in the same directory as the script.

## Usage

1. **Clone the repository** (or download the script):
    ```bash
    git clone https://github.com/your-repository/ASL-Sign-Language-Detection.git
    cd ASL-Sign-Language-Detection
    ```

2. **Place the model file** (`asl_model.pkl`) in the same directory.

3. **Run the script**:
    ```bash
    python asl_realtime_predict.py
    ```

4. **Instructions**:
    - **Make hand gestures**: The model detects the gestures and converts them into ASL letters.
    - **Space gesture**: Use a wide open hand (fingers spread out) to add a space between words.
    - **Backspace**: Press **Backspace** to delete the last letter in the sentence.
    - **Save sentence**: Press **Enter** to save the built sentence to a text file (`asl_sentence_output.txt`).
    - **Exit**: Press **ESC** to exit the program.

5. **Sentence Example**:
    - If you sign "HELLO" using the correct hand gestures, the sentence will be built as: `HELLO`.

## How it Works

1. **MediaPipe Hand Tracking**:
    - The program uses **MediaPipe** to track hand landmarks in real-time.
    - The coordinates of hand landmarks are passed to the trained ASL model to predict the corresponding letter.

2. **Machine Learning Model**:
    - A machine learning model is used to map the hand gestures to their corresponding letters.
    - You can train the model on a custom dataset of ASL gestures or use an existing one.

3. **Backspace and Save**:
    - The user can delete the last letter typed by pressing **Backspace**.
    - The sentence is saved to a text file when **Enter** is pressed.

## Files

- `asl_sentence_builder.py`: Main script for real-time ASL detection, gesture processing, and sentence building.
- `asl_model.pkl`: Pre-trained machine learning model (should be placed in the same directory).
- `asl_sentence_output.txt`: Text file where completed sentences are saved.

## Example Output

### In the terminal:

```bash 
Final Sentence: HELLO
 ```
### In the webcam window:
- **Predicted Letter**: Displays the current letter prediction.
- **Sentence**: Displays the built sentence as you make new gestures.

---

## Contributions

Contributions are welcome! Feel free to fork this repository, submit issues, and create pull requests.

---

### Acknowledgements

- **MediaPipe**: For hand tracking and pose estimation.
- **OpenCV**: For image processing and webcam handling.
- **Scikit-learn**: For machine learning model building (if used for training).
