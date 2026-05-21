# Graph Neural Networks for MS Brain Network Classification

## Project Overview

This project implements a comprehensive comparison of three different machine learning approaches for classifying brain networks from Multiple Sclerosis (MS) patients vs. healthy volunteers (HV) using MRI-derived connectivity data.

### Dataset
- **Total Subjects:** 165 (120 MS patients, 45 healthy volunteers)
- **Brain Regions:** 76 nodes (AAL atlas)
- **Connectivity Matrices:** 3 types
  - FA (Diffusion Tensor Imaging - Fractional Anisotropy)
  - GM (Grey Matter)
  - RS (Resting-State fMRI)
- **Task:** Binary classification (HV vs. pwMS)

## Project Structure

```
.
├── optional_GNN_GerardAsbert_MarinaRosell.ipynb    # Main notebook with all implementations
├── requirements.txt                                # Python dependencies
├── README.md                                       # This file
├── data/
│   ├── demographics.csv                            # Subject metadata
│   ├── nodes.csv                                   # Brain region names
│   ├── FA/                                         # DTI-FA matrices (165 files)
│   ├── GM/                                         # GM matrices (165 files)
│   └── RS/                                         # RS matrices (165 files)
└── output/                                         # Results and visualizations (generated)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Repository
```bash
cd /path/to/GNN-example
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv gnn_env
source gnn_env/bin/activate  # On Windows: gnn_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Alternative: Install PyTorch Separately (for GPU Support)
For GPU acceleration with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install other packages:
```bash
pip install torch-geometric scikit-learn pandas numpy matplotlib seaborn networkx imbalanced-learn jupyter
```

## Running the Project

### Method 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook optional_GNN_GerardAsbert.ipynb
```

Then select "Run All Cells" or run cells sequentially using Shift+Enter.

### Method 2: Command Line
```bash
jupyter nbconvert --to notebook --execute optional_GNN_GerardAsbert.ipynb
```

### Method 3: IPython
```bash
ipython optional_GNN_GerardAsbert.ipynb
```

## Project Components

### Part 1: Traditional Graph Learning Approach

**Objective:** Extract hand-crafted graph features and train classical ML models

**Key Steps:**
1. Extract 33 graph-level features from each subject's connectivity matrices:
   - Network density, average strength, clustering coefficient
   - Global clustering, transitivity, path length
   - Assortativity, degree statistics, weight statistics
2. Apply data augmentation:
   - Brain region flipping (doubles dataset to 330 subjects)
   - SMOTE (balances class distribution)
3. Train and compare three ML models:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
4. Evaluate best model with confusion matrix, AUC-ROC, F1-score

**Results:** Traditional approach provides interpretable baseline performance

### Part 2: Graph Convolutional Networks (GCN)

**Objective:** Learn node embeddings through graph convolutions

**Model Architecture:**
- Input: Node features (1D per node) + Graph topology
- 3 GCN layers (64 hidden units)
- Global max pooling for graph-level representation
- 2 FC layers for classification

**Key Features:**
- Directly learns from graph structure
- Message passing between neighboring nodes
- 50 training epochs with Adam optimizer
- Dropout for regularization

**Advantages:**
- End-to-end learning
- Captures local graph patterns
- Scalable to large graphs

### Part 3: Graph Attention Networks (GAT)

**Objective:** Use attention mechanisms for adaptive neighbor weighting

**Model Architecture:**
- Input: Node features (1D per node) + Graph topology
- GAT layer with 8 attention heads
- Second GAT layer with 1 attention head
- Global max pooling
- 2 FC layers for classification

**Key Features:**
- Dynamic attention weights for each neighbor
- Multi-head attention for diverse representations
- 50 training epochs with Adam optimizer
- ELU activation for better gradient flow

**Advantages:**
- Learnable neighbor importance weighting
- Can focus on most relevant connections
- Interpretable attention patterns

## Data Augmentation Strategy

### Brain Region Flipping
Leverages the well-established hemispheric symmetry of the brain:
- Maps left-hemisphere regions to right-hemisphere equivalents
- Creates realistic synthetic samples by flipping adjacency matrices
- Doubles dataset from 165 to 330 samples

### SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic samples for minority class (HV)
- Balances class distribution from 90-10 to 50-50
- Applied after brain flipping to address remaining imbalance

## Performance Metrics

All approaches are evaluated using:
- **Accuracy:** Overall correctness of predictions
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** True positives, true negatives, false positives, false negatives
- **Precision-Recall Curve:** Trade-off between precision and recall

## Computational Requirements

| Approach | GPU Required | Training Time | CPU RAM | GPU RAM |
|----------|-------------|---------------|---------|---------|
| Part 1   | No          | < 1 minute    | 4 GB    | N/A     |
| Part 2   | Recommended | 5-10 minutes  | 4 GB    | 2 GB    |
| Part 3   | Recommended | 5-10 minutes  | 4 GB    | 2 GB    |

## Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'torch_geometric'"**
```bash
pip install torch-geometric
```

**Issue: "CUDA out of memory" error**
- Reduce batch size in DataLoader
- Or set device to CPU: `device = torch.device('cpu')`

**Issue: "FileNotFoundError: data directory not found"**
- Ensure you're running from the correct directory
- Verify data files exist in `./data/` subdirectory

**Issue: Slow performance on traditional approach (Part 1)**
- NetworkX graph operations are CPU-bound
- Consider running on a machine with multiple cores

## Expected Output

The notebook generates:
1. **Console Output:**
   - Data loading progress
   - Training loss curves
   - Performance metrics for each model

2. **Visualizations:**
   - Confusion matrices (heatmaps)
   - ROC curves
   - Precision-Recall curves
   - Model comparison bar plots
   - Training loss curves

3. **Results Summary:**
   - Comparison table of all three approaches
   - Detailed classification reports

## Interpretation of Results

### Binary Classification (HV vs. pwMS)
- **Accuracy > 0.80:** Model performs well at distinguishing HV from MS
- **AUC-ROC > 0.85:** Excellent discriminative ability
- **F1-Score > 0.80:** Good balance between precision and recall

### Model Selection
- Choose based on your application requirements:
  - **Part 1:** Fast, interpretable, CPU-friendly
  - **Part 2:** Balance of speed and performance
  - **Part 3:** Potentially best performance with attention mechanisms

## References

### Key Papers
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Veličković et al. (2018): "Graph Attention Networks"
- Chawla et al. (2002): "SMOTE: Synthetic Minority Over-sampling Technique"

### Data Source
- MS Unit at Hospital Clínic de Barcelona
- Ethics Committee approved study with informed consent

## License

This project is provided for educational purposes as part of the Master's program in Computer Vision.

## Author

Gerard Asbert
Master's Degree in Computer Vision
University Computer Vision Center (CVC)

## Contact & Support

For issues or questions about the project:
1. Check the Troubleshooting section above
2. Review the notebook's markdown cells for detailed explanations
3. Verify data files are correctly placed in the `./data/` directory

## Disclaimer

This project is for research and educational purposes. The models should not be used for clinical diagnosis without proper validation and approval by medical professionals and institutional review boards.
