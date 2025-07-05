# Applied Machine Learning for Options Pricing  
*Internship Project at [Quantinsider](https://quantinsider.io)*  
*Duration: Sep'24 – Jan'25*  

---

## 🎯 Objective

This project aimed to explore advanced machine learning methods for **pricing options more accurately** than traditional models such as Black-Scholes.  
The focus was on:
- Predicting **bid and ask prices** simultaneously,
- Improving **model robustness** under distribution shifts,
- Exploring architectures capable of **zero-shot generalization**.

---

## 📈 Data Summary

- **Size**: Over **12 million** option contracts  
- **Fields Used**:  
  - Underlying asset price  
  - Strike price  
  - Expiry time  
  - Implied volatility  
  - Bid/Ask spreads  
  - Option Greeks (Delta, Gamma, etc.)  

---

## 🔧 Modeling Approach

The following techniques and models were explored:

### 1. **Model Architectures**
- **MLP1, MLP2**: Feedforward networks with different layer depths and neuron widths  
- **LSTM**: For capturing sequential dependencies in time-series structured option data  

### 2. **Techniques Used**
- **Multi-task learning**: To jointly predict bid and ask prices  
- **Leaky ReLU**, **Batch Normalization**, and large **batch sizes (4096)** for faster and more stable convergence  
- **Distribution-Aware Training**: Ensuring that the model remains robust under changes in market regime  
- **Zero-shot generalization**: Testing the model’s ability to price previously unseen contracts

---

## ⚙️ Key Implementation Highlights

- Data preprocessing and normalization using market microstructure features  
- Designed separate heads for bid and ask outputs in MTL setting  
- Loss functions customized to penalize mispricing in edge scenarios  
- Used **AdamW optimizer** with **learning rate scheduling**

---

## 🏆 Results

| Metric                        | Traditional (Black-Scholes) | Proposed Model (MLP2) |
|------------------------------|------------------------------|------------------------|
| **MSE (Put Options)**        | 533                          | **8.8**               |
| **PE20 Accuracy**            | 24%                          | **55%**               |
| **Model Bias (MLP2)**        | –                            | **0.09**              |

- **MLP2** consistently showed the **lowest Mean Squared Error** and bias across test samples  
- **LSTM** demonstrated better performance under **relaxed accuracy thresholds**, suggesting strong generalization  
- Achieved **95–98% reduction in error** compared to Black-Scholes baseline

---

