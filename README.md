# Book Recommender System

A hybrid recommendation system combining Collaborative Filtering (SVD) and Content-Based Filtering to provide personalized book recommendations. Built for the Master in Artificial Intelligence program at Cadi Ayyad University.

## Authors

- Mustapha Mensouri
- Abderahman El-Hamidy

**Supervised by:** Pr. Yassine Afoudi

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Performance Benchmark](#performance-benchmark)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Future Work](#future-work)
11. [References](#references)

---

## Project Overview

This project implements a book recommender system using the Goodbooks-10k dataset. The system addresses key challenges in recommendation systems including:

- Data sparsity (99.8% sparse interaction matrix)
- Cold start problem for new users
- Scalability with large datasets

The final solution is a weighted hybrid model that combines:

- **SVD (Singular Value Decomposition)** for collaborative filtering
- **TF-IDF + Cosine Similarity** for content-based filtering

---

## Features

- **Hybrid Recommendation Engine**: Combines collaborative and content-based filtering
- **Cold Start Handling**: New users can receive recommendations based on content similarity
- **Interactive Web Interface**: Built with Streamlit for seamless user experience
- **Real-time Recommendations**: Instant personalized suggestions
- **Book Details View**: Comprehensive book information with similar recommendations

---

## System Architecture

### Hybrid Formula

The final recommendation score is calculated using a weighted combination:

```
Score_final = (α × SVD_score) + ((1 - α) × Content_score)
```

Where:
- `α = 0.7` (70% weight to collaborative filtering)
- `1 - α = 0.3` (30% weight to content-based filtering)

### Workflow

1. **User Detection**: System identifies known vs new users
2. **Known Users**: Hybrid engine combines SVD + content similarity
3. **New Users**: Cold start engine uses content-based filtering
4. **Display**: Results rendered in interactive Streamlit interface

---

## Dataset

**Goodbooks-10k Dataset**

- **Users**: 53,424
- **Books**: 10,000
- **Ratings**: ~6 million interactions
- **Sparsity**: 99.8%
- **Metadata**: Book titles, authors, publication years, average ratings

### Key Observations

- **Rating Distribution**: Strong positivity bias (most ratings are 4-5 stars)
- **Popularity Distribution**: Long-tail distribution (few books have many ratings)

---

## Methodology

### Models Evaluated

We benchmarked 7 different recommendation algorithms:

#### Matrix Factorization

1. **SVD** - Singular Value Decomposition
2. **SVD++** - SVD with implicit feedback
3. **NMF** - Non-Negative Matrix Factorization

#### Deep Learning

4. **NCF** - Neural Collaborative Filtering
5. **AutoEncoder** - Reconstruction-based model

#### Graph Neural Networks

6. **LightGCN** - Simplified graph convolution
7. **GraphSAGE** - Inductive graph learning

### Evaluation Metrics

- **RMSE** (Root Mean Square Error): Measures rating prediction accuracy
- **Precision@10**: Percentage of relevant items in top-10 recommendations
- **Recall@10**: Percentage of relevant items successfully retrieved

---

## Performance Benchmark

### Final Results

| Model | Type | RMSE | Precision@10 | Recall@10 |
|-------|------|------|--------------|-----------| 
| **SVD++** | Matrix Factorization | **0.8401** | **0.6748** | **0.9552** |
| **SVD** | Matrix Factorization | 0.8412 | 0.6747 | 0.9550 |
| NMF | Matrix Factorization | 0.8786 | 0.6717 | 0.9548 |
| NCF | Deep Learning | 0.9024 | 0.6735 | 0.9535 |
| AutoEncoder | Deep Learning | 0.9625 | 0.6730 | 0.9566 |
| GraphSAGE | Graph Neural Network | 1.0728 | 0.6687 | 0.9514 |
| LightGCN | Graph Neural Network | 1.2792 | 0.6668 | 0.9491 |

### Key Insights

- **Matrix Factorization Dominance**: SVD/SVD++ outperformed complex deep learning models
- **Dataset Characteristics**: Relatively dense dataset favors analytical solutions over neural networks
- **GNN Performance**: Higher RMSE but competitive recall (optimized for ranking, not rating prediction)

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Book-Recommender-System.git
cd Book-Recommender-System
```

2. **Set up the virtual environment**

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify Data and Models**

The repository comes pre-loaded with the necessary data and trained models. Ensure the following files exist in your directory:

- `data/books.csv` and `data/ratings.csv`
- `models/svd_model.pkl` and `models/cbf_model.pkl`

*(Note: If the data files are missing, you can download them from the [Goodbooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) and place them in the `data/` folder.)*

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Features

1. **User Profile**: Enter a user ID to get personalized recommendations.
2. **Cold Start Mode**: For new users, select a favorite book to initialize preferences.
3. **Search**: Find books by title.
4. **Book Details**: Click on any book to view full details and similar recommendations.

---

## Project Structure

```text
Book-Recommender-System/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # License file
│
├── assets/
│   └── style.css          # Custom CSS styling
│
├── data/                  # Dataset files
│   ├── books.csv
│   ├── ratings.csv
│   ├── book_tags.csv
│   ├── tags.csv
│   ├── to_read.csv
│   └── sample_book.xml
│
├── models/                # Trained models
│   ├── svd_model.pkl      # Collaborative filtering model
│   └── cbf_model.pkl      # Content-based filtering data
│
└── venv/                  # Virtual environment
```

---

## Technical Implementation

### Backend

- **Scikit-Learn**: TF-IDF vectorization, cosine similarity
- **Surprise Library**: SVD model training and inference
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Frontend

- **Streamlit**: Interactive web interface
- **Custom CSS**: Professional styling with flexbox layout

### Model Serialization

Models are serialized using Python's `pickle` module for instant inference without retraining.

---

## Future Work

### Planned Enhancements

1. **Dynamic User-Profile Tracking**
   - Implement temporal dynamics to track evolving user preferences
   - Weight recent interactions more heavily

2. **Online Learning Pipeline**
   - Real-time model updates after user interactions
   - Incremental learning without full retraining

3. **Deep Content Understanding**
   - Integrate Large Language Models (LLMs) for plot summary analysis
   - Use CNNs to extract visual features from book covers

4. **Advanced Features**
   - Multi-criteria recommendations (genre, mood, reading level)
   - Social features (friend recommendations, reading groups)
   - A/B testing framework for model comparison

---

## References

1. He, X., et al. (2020). LightGCN: Simplifying and powering graph convolution network for recommendation. SIGIR 2020.
2. Lops, P., et al. (2011). Content-based recommender systems: State of the art and trends. Recommender Systems Handbook.
3. [Surprise Library Documentation](https://surprise.readthedocs.io/en/stable/)
4. [Goodbooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

