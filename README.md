# DeFiAgent: On-chain AI for Predicting DEX Liquidity Pool APY

DeFiAgent presents an on-chain AI model that predicts the Annual Percentage Yield (APY) for decentralized exchange (DEX) liquidity pools, enabling users to make informed decisions. Leveraging the efficient DLinear model, this solution is optimized for minimal on-chain execution overhead while ensuring robust predictive performance.

## Key Features

- **On-chain APY Prediction:** Forecast DEX liquidity pool APYs directly on-chain, enhancing transparency and trust.
- **Efficient Model Design:** Utilizes a simplified, yet highly accurate DLinear architecture optimized for blockchain deployment.
- **Optimized Blockchain Execution:** Minimal model layers and computations for cost-effective and efficient on-chain operation.

## ⚙️ Setup & Installation

### Prerequisites

- Node.js (16.x or above)
- Python 3.x
- TensorFlow.js

### Installation

Clone the repository and install dependencies:

```bash
git clone [your_repository_url]
cd defiagent
npm install
```

Set up environment variables:

```bash
cp .env.template .env
```

Edit the `.env` file and fill in your details.

## 🚧 Usage

### Model Conversion & Publishing

Convert and publish your Web2 DL model to the Sui blockchain:

```bash
node model_publish.js
```

### Running Inference

Execute optimized on-chain inference:

```bash
node inference_v2.js
```

- `init`: Initialize model state on-chain.
- `load input`: Load input data.
- `run`: Execute inference with optimized blockchain transactions.

## 📄 Detailed File Descriptions

### Python Scripts

- **`dlinear_2.py`**
  - Trains the original DLinear model to predict the 3-day APY for the SUI-USDT liquidity pool and saves the model in `.h5` format.

- **`dlinear_3.py`**
  - Optimized version of `dlinear_2.py` with external computation of trend and residual components. Reduces model layers to three for efficient blockchain execution. Saves the model in `.h5` format.

- **`sui_key.py`**
  - Converts private keys beginning with `suiprivkey` into hexadecimal format suitable for blockchain interactions.

### JavaScript Scripts

- **`js_dlinear_2.js`**
  - JavaScript (TensorFlow.js) equivalent of `dlinear_2.py`. Trains the model and stores it in TFJS format.

- **`js_dlinear_3.js`**
  - JavaScript (TensorFlow.js) equivalent of `dlinear_3.py`. Includes precomputed trends and residuals for optimized blockchain execution and stores the model in TFJS format.

