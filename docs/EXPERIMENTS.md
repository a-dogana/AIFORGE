# AIFORGE Experiments Protocol

Every experiment run by AIFORGE is expected to follow strict reproducibility standards.

## Experiment v0.1: Digits Classification

**Objective:** Validate the end-to-end training pipeline of AIFORGE using a lightweight dataset.

**Dataset:** Scikit-learn `load_digits()`
- 1797 samples of 8x8 images of handwritten digits.
- Split: 70% Train, 15% Validation, 15% Test.
- Preprocessing: `StandardScaler`.

**Architecture:** Multilayer Perceptron (MLP)
- Input: 64 features
- Hidden Layers: 128, 64 (ReLU activations)
- Output: 10 classes

**Hyperparameters:**
- Epochs: 10
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.005

**Results (Run 1):**
- Hardware: Apple Silicon (MPS) / 8GB RAM
- Training Time: < 2 seconds
- Test Accuracy: ~96.68%
- Persisted in: `registry/exp_<id>.json`

## Reproducing
Run the experiment with:
```bash
source venv/bin/activate
python run_experiment.py
```
Check the `registry/` directory for the resulting JSON file.
