# MatID: A Deep Learning Approach to WiFi-Based Material Identification

## Description:

MatID identifies materials from WiFi signals with deep learning. WiFi routers carry internet traffic. This project reads amplitude and phase changes in the signal itself to sense the environment, which gives a non-invasive, privacy-preserving alternative to camera-based sensing.

## Key Features:

- **WiFi Sensing**: Reads amplitude and phase changes in WiFi signals to interpret the environment.
- **Deep Learning Integration**: Combines classical WiFi signal processing with 1D CNNs, 2D CNNs and ResNet18 to identify materials.
- **Signal Preprocessing**: Reduces noise and sharpens the signal before it reaches the models.
- **Low Cost**: Runs on existing WiFi infrastructure, so you need no camera hardware.
- **Applications**: Object identification and material identification.

## Research Contributions:

- **Fused architecture**: Merges CSI-derived features and a spectrogram in one classifier.
- **New WiFi sensing application**: Extends WiFi sensing to material identification.

## Method Overview:

The model classifies **13 classes** (12 materials plus the empty setup): acrylic, aluminium, brass, copper, empty, lignum vitae, nylon, oak, pine, PP, PVC, rose wood and steel. We collect measurements in a shielded environment (*box*) and an unshielded one (*not_box*).

1. **Preprocessing**: The pipeline extracts amplitude and phase from the raw CSI, sanitizes the phase with a linear calibration and removes amplitude outliers. A Short-Time Fourier Transform turns the signal into a spectrogram, which the pipeline resizes for ResNet18.
2. **Feature branches**:
   - a **1D CNN** processes the sanitized phase;
   - a **2D CNN** processes amplitude, phase and RSSI together;
   - a **ResNet18** processes the spectrogram.
3. **Fusion**: Fully connected layers merge the two CNN branches. The pipeline concatenates that output with the ResNet18 output and passes it to a final classifier.

The best configuration from the hyperparameter search uses Adam with a learning rate of 1e-4 for both the network and the ResNet, a weight decay of 1e-4, a dropout of 0.5 and a batch size of 64.

## Repository Contents:

```
.
├── code/            Modular Python version of the project (see the note below)
│   ├── config.py
│   ├── main.py
│   ├── data_processing/   dataset and signal preprocessing
│   ├── models/            network architecture
│   ├── training/          training and evaluation loop
│   └── utils/             plots, metrics and helpers
├── ipynb/           Google Colab notebook with the full pipeline, experiments and results
├── pdf/             Thesis
└── presentation/    Thesis presentation
```

- **Source Code**: The MatID model and the signal preprocessing.
- **Documentation**: Thesis and presentation.

## Getting Started:

The notebook in `ipynb/` is the reference implementation. It holds the complete pipeline (data loading, preprocessing, model, training, hyperparameter search) and the experiment results.

1. Open the notebook in Google Colab and use a GPU runtime.
2. Upload the dataset to Google Drive and set `ROOT` in the constants cell to its path. The notebook expects the `Dataset/{train,dev,test}/<material>/{with,without}/` layout.
3. Run the cells in order. The notebook saves checkpoints and loss curves in the `ckp/` folder under `ROOT`.

**Note on the `code/` folder**: It is a modular rewrite of the notebook. Run it from the command line with `python main.py` (`--help` lists the options). It is a work in progress: the spectrogram functions and sample processing are placeholders, and the dataset loading and model differ from the notebook. Use the notebook to reproduce the results.

## Future Directions:

- Improve signal preprocessing.
- Explore additional applications and integrate with other IoT devices.
- Run experiments in different rooms to test robustness.
- Improve the dataset.
- Build a domain adaptation system for new environments.
- Align the modular `code/` version with the notebook.