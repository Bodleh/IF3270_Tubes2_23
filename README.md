# IF3270_Tubes2_23

This repository contains implementations of various neural network architectures, including Convolutional Neural Networks (CNN), Feedforward Neural Networks (FFNN), Long Short-Term Memory (LSTM), and Recurrent Neural Networks (RNN). Keras (Tensorflow) are used for the full implementations and NumPy for the from-scratch forward pass implementations. This project is part of the IF3270 course.

## Setup and Run

To set up and run the programs in this repository, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/IF3270_Tubes2_23.git
    cd IF3270_Tubes2_23
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Each sub-directory (`src/cnn`, `src/ffnn`, `src/lstm`, `src/rnn`) may have its own specific dependencies. Navigate to the relevant directory and install the required packages.

    For example, to install dependencies for CNN:

    ```bash
    cd src/cnn
    pip install -r requirements.txt
    ```

    And for RNN:

    ```bash
    cd src/rnn
    pip install -r requirements.txt
    ``` 

    _Note: If a `requirements.txt` file is not present in a specific subdirectory, you might need to install dependencies manually based on the code._

4.  **Run the programs:**
    Navigate to the specific subdirectory of the model you wish to run and execute the main script or Jupyter Notebook.

    To run the CNN module:

    ```bash
    # example
    cd src/cnn
    jupyter notebook main.ipynb
    ```

    To run the RNN module:

    ```bash
    # example
    cd src/rnn
    python main.py
    ```

    To run the LSTM module:

    ```bash
    # Navigate to LSTM module
    cd src/lstm

    # Run the experiments
    python experiment.py

    # Plot the experiments results
    python plot_result.py

    # Run forward pass comparation of from-scratch LSTM and Keras LSTM
    python compare_forward_pass.py

    # You can run and plot all experiments and compare forward pass with jupyter notebook
    jupyter notebook pengujian.ipynb
    ```

## Team Member Task Distribution

| Team Member         | NIM      | Task |
| :------------------ | :------- | :--- |
| Juan Alfred Widjaya | 13522073 | LSTM |
| Albert              | 13522081 | RNN  |
| Ivan Hendrawan Tan  | 13522111 | CNN  |
