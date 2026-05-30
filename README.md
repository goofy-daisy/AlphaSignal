# AlphaSignal

I built AlphaSignal as a local, open-source investment research platform that synthesises price momentum, news sentiment, regulatory filings, and social signals into a single, unified analytical interface. I designed the entire platform to operate locally on your machine, leveraging free APIs and open-weight models to ensure complete data privacy and zero ongoing platform costs.

---

## The Problem

Institutional-grade investment research depends on the continuous synthesis of multi-dimensional signals: technical price structures, shifting sentiment in media coverage, subtle language updates in regulatory filings, and localised social momentum. For independent researchers and portfolio managers, this infrastructure is typically fragmented or entirely locked behind enterprise terminal subscriptions costing upwards of $3,000 per month. 

I engineered AlphaSignal to provide a working production reference architecture for this problem. I aggregated these distinct pipelines locally, allowing you to run advanced deep learning models, maintain a semantic research index, and orchestrate an LLM analytical agent entirely on consumer-grade hardware without exposing proprietary watchlists to external cloud providers.

---

## Who This Is For

* **Quantitative Researchers:** Analysts looking for a functional, multi-signal engineering baseline that can be easily forked and customised with proprietary data feeds or custom weights.
* **Machine Learning Engineers:** Developers seeking an end-to-end reference implementation combining time-series forecasting (TFT), domain-specific NLP (FinBERT), vector search (FAISS), and structured tool execution via local LLM agents.
* **Portfolio Managers & Analysts:** Professionals managing a focused watchlist who require an automated, data-driven second opinion to highlight anomalies and aggregate context.
* **Financial Technology Engineers:** Software builders looking for clean patterns on handling data pipelines, walk-forward validation strategies, and local model deployment.

*(Note: AlphaSignal is a research and decision-support tool. It does not possess an execution layer, does not manage live portfolios, and does not provide formal financial or investment advice.)*

---

## Reference Architecture

    ========================================================================
                                  DATA INGESTION
       yfinance  |  SEC EDGAR  |  ASX Announcements  |  Alpha Vantage  |  NewsAPI
    ========================================================================
                                        |
                                        v
                           +-------------------------+
                           |   PostgreSQL Database   |
                           | (Prices, News, Social)  |
                           +-------------------------+
                                        |
         +------------------------------+------------------------------+
         |                              |                              |
         v                              v                              v
    +------------------+       +------------------+       +------------------+
    |  Feature Engine  |       | FinBERT Pipeline |       |    FAISS RAG     |
    |  40 Technical    |       |  Sentiment Loss  |       | Vector Search &  |
    |  Indicators (ta) |       | (News & Filings) |       | Semantic Context |
    +------------------+       +------------------+       +------------------+
         |                              |                              |
         v                              |                              |
    +------------------+                |                              |
    |    TFT Model     |                |                              |
    |  Multi-Horizon   |                v                              |
    |   Price Signal   |       +------------------+                    |
    +------------------+       |   Signal Layer   |                    |
         |                     | Score Aggregator |                    |
         +-------------------->+   [-1.0 to 1.0]  |                    |
                               +------------------+                    |
                                        |                              |
                                        v                              |
                           +-------------------------+                 |
                           |  LightGBM Meta-Learner  |                 |
                           | Composite Signal Output |                 |
                           +-------------------------+                 |
                                        |                              |
                                        v                              |
                           +-------------------------+                 |
                           |     LangChain Agent     |                 |
                           |   (Ollama: llama3.1)    |<----------------+
                           +-------------------------+
                                        |
                        +---------------+---------------+
                        |                               |
                        v                               v
           +-------------------------+     +-------------------------+
           |       FastAPI REST      |     |   Streamlit Interface   |
           |  Backend Services (:80) |     |    Dashboard (:8501)    |
           +-------------------------+     +-------------------------+

---

## Component Allocation

| Library / System | Functional Responsibility |
| :--- | :--- |
| **yfinance** | I used this to ingest baseline historical OHLCV pricing and match structured ticker headlines. |
| **ta** | I integrated this to compute 40 separate technical indicators across trend, momentum, volume, and volatility. |
| **pytorch-forecasting** | I implemented the Temporal Fusion Transformer (TFT) using this library for specialised time-series forecasting. |
| **transformers (FinBERT)** | I ran local sequence classification here to extract financial sentiment polarity from text. |
| **sentence-transformers** | I utilised this to map unstructured corporate reports and text into dense 384-dimensional vector spaces. |
| **faiss-cpu** | I deployed this to execute high-performance local vector similarity lookups for the RAG architecture. |
| **lightgbm** | I configured this as the final tabular meta-learner to blend individual analytical scores. |
| **langchain** | I built the analyst agent using this to manage tool registration, input routing, and memory states. |
| **Ollama (llama3.1)** | I powered the local language inference model with this to synthesise data metrics into prose. |
| **PostgreSQL** | I set this up as the relational storage layer for unified asset metrics, signals, and cache. |
| **FastAPI** | I exposed high-throughput REST endpoints tracking generated scores, assets, and histories. |
| **Streamlit** | I built the responsive interactive frontend application interface using this framework. |

---

## Known Limitations

System transparency is essential for reliable quantitative analysis. I designed the platform with the following technical boundaries in mind:

* **Forecasting Volatility Fallback:** I trained the Temporal Fusion Transformer (TFT) on approximately 4 years of daily intervals. For assets displaying structural fragmentation or low volume regimes, walk-forward Information Coefficients can break down. In these scenarios, I configured the model to gracefully return a fallback score of `0.0` rather than interpolating low-confidence trends.
* **ASX Filing Horizons:** I built the ASX ingestion engine to track real-time corporate action boards and return the most recent announcements (typically 5 to 20 items per asset). I did not construct deep multi-year structural archives for regional data.
* **SEC Filing Edge Cases:** While US corporate filings cover extensive historical windows, anomalies in data providers can occasionally occur. For instance, specific tickers (such as MSFT, JPM, HD, and CVX) may return empty metrics during baseline initialisation runs, forcing a neutral filing signal component.
* **Social Feed Density:** I explicitly utilised Alpha Vantage and NewsAPI to ensure social sentiment indicators are highly robust, though they are optimised primarily for major US large-cap equities. For less conversational or thinly traded symbols, I programmed the social signal vector to automatically fall back to general news text trends.
* **News Corpus Gaps:** Regional or asset-specific tickers (such as AZJ.AX and QAN.AX) can experience sparse coverage across generalised public NewsAPI feeds. For these tickers, I set the sentiment scoring to default strictly to raw historical headline strings from alternative data passes.
* **Local Agent Generation:** Running complex language tasks on consumer hardware using `llama3.1` can occasionally lead to non-deterministic string returns or raw JSON block escaping. While I built internal parsing exception catchers to filter out most anomalies, generation behaviour may vary across environments.
* **Hardware Compute Bounds:** Due to rigid numerical dependencies within historical versions of the underlying time-series frameworks, I set the inference and model passes to evaluate via CPU threads. This step incurs a brief computational layout delay of a few seconds per evaluated asset ticker during calculation runs.

---

## Local Environment Execution

### Prerequisites

* **Python 3.11.15 exactly:** Required for compatibility with the time-series forecasting dependencies (`pytorch-forecasting` explicitly errors out on 3.12+ environments).
* **PostgreSQL 15**
* **Ollama & llama3.1:** * **Mac:** Run `brew install ollama`, start the app, then run `ollama pull llama3.1`
  * **Windows:** Download from [ollama.com/download/windows](https://ollama.com/download/windows), install, then run `ollama pull llama3.1`
* **Free API Keys Required:**
  * **Alpha Vantage:** [Register for a free key here](https://www.alphavantage.co/support/#api-key)
  * **NewsAPI:** [Register for a free key here](https://newsapi.org/register)

### 0. Clone the Repository

```bash
git clone [https://github.com/your-username/alphasignal.git](https://github.com/your-username/alphasignal.git)
cd alphasignal
```

### 1. Database Provisioning

Ensure your local PostgreSQL instance is running, then execute the database creation query:

**Mac / Linux Platforms:**
```bash
createdb alphasignal
```

**Windows Platforms:**
```cmd
createdb alphasignal
```

### 2. Environment Configuration

Copy the template configuration file:

```bash
cp .env.example .env
```

Open `.env` in your editor and insert your respective database connection string and developer API keys.

### 3. Dependency Installation

**Mac / Linux Platforms:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows Platforms:**
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note for Windows Environments with Dedicated CUDA Acceleration: Run the following command directly before running the requirements install to ensure the explicit hardware-accelerated version of PyTorch overrides standard CPU versions: `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`)*

### 4. Database Ingestion & Initialisation

Populate your database schemas and build localised matrix indices by running the processing stack sequentially:

**Mac / Linux Platforms:**
```bash
python scripts/backfill_prices.py
python scripts/backfill_filings.py
python scripts/build_faiss_index.py
```

**Windows Platforms:**
```cmd
python scripts\backfill_prices.py
python scripts\backfill_filings.py
python scripts\build_faiss_index.py
```

### 5. Model Execution & Optimisation

Execute background tracking updates and optimise model training layers. *Note: Running the initial time-series training loop requires multiple processing hours; I recommend letting this run continuously in a decoupled shell session.*

**Mac / Linux Platforms:**
```bash
python scripts/train_tft.py
python scripts/assemble_signals.py
```

**Windows Platforms:**
```cmd
python scripts\train_tft.py
python scripts\assemble_signals.py
```

### 6. Starting Service Nodes

Launch both execution runtimes in separate terminal windows with your virtual environment active in both.

**Terminal Suite 1 (FastAPI Core Engine):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal Suite 2 (Interactive Streamlit Interface):**
```bash
streamlit run dashboard/app.py
```

### 7. Verification & Health Check

Open a third terminal and run these commands to verify the system is returning data:

```bash
# Check if the API is alive
curl http://localhost:8000/health

# Pull a signal report for an ASX ticker
curl http://localhost:8000/signals/BHP.AX
```

* **Dashboard Endpoint:** `http://localhost:8501`
* **API Open Documentation:** `http://localhost:8000/docs`

---

## Project Structure

```text
alphasignal/
├── api/                    # FastAPI application and route handlers
├── config/                 # Stock universe (50 tickers) and application settings
├── dashboard/              # Streamlit pages and app entry point
├── scripts/                # Data backfill, model training, and signal assembly
├── src/
│   ├── agent/              # LangChain agent tools and executor
│   ├── backtest/           # Backtester and report writing utilities
│   ├── data/               # Data fetchers — price, news, filings, social
│   ├── embeddings/         # FAISS index building and semantic retrieval
│   ├── features/           # 40 technical indicator computation pipeline
│   ├── models/             # TFT, FinBERT pipeline, and LightGBM meta-learner
│   └── signals/            # Price, sentiment, filing, social, and composite signals
├── .env.example            # Environment variable template
├── METHODOLOGY.md          # Signal methodology and design decisions
└── requirements.txt        # Pinned Python dependencies
```

---

## License

I have open-sourced this project under the MIT License — see the repository's `LICENSE` file for details.
