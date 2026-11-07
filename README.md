# Bachelor Thesis: When Websites Learn to Speak
**Implementation of a Retrieval-Augmented Generation System Based on Web-Scraped Data**

This repository contains the code for my bachelor thesis. I built a complete Retrieval-Augmented Generation (RAG) pipeline from scratch, covering everything from initial web scraping and data extraction to querying the system and running automated evaluations.

> **Note on testing:** To make evaluating the project easier, the pre-collected data and locally stored models are included in this repository. If you want to test the actual data collection and model download processes yourself, simply delete the `data` and `models` directories before running the code.

---

## Requirements & Installation

The project requires **Python 3.11.9 or higher**. I recommend setting up a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables (`.env`)

You need to provide API keys to run the generation and evaluation steps. Create a `.env` file in the root directory and configure the following:

* **OpenAI (Required for evaluation):** You need an account with available credits (a full test run costs less than 1€).
`OPENAI_API_KEY=your_key_here`

* **OpenRouter (Optional):** As an alternative to OpenAI, you can use a free model from OpenRouter for generation (note: automated evaluation still requires OpenAI).
`OPENROUTER_API_KEY=your_key_here`

* **Hugging Face (Optional but recommended):** Add an access token for faster model downloads.
`HF_TOKEN=your_token_here`

---

## Usage

You can start the entire pipeline by running:

```bash
python main.py
```

This script orchestrates the workflow. It creates the required data structures, downloads missing models, and then asks if you want to perform a full system evaluation or proceed directly to chatting with the RAG system in your terminal.

---

## Configuration

You can tweak the system's behavior and network settings directly in `config.py`. 

Available configurations include:
* **Network:** `PROXY`, `CERTIFICATE`
* **LLM Setup:** `GENERATION_API_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL`

---

## Academic Context

* **Author:** Oliver Reuß
* **University:** Hof University of Applied Sciences (Faculty of Computer Science, Media Informatics B.Sc.)
* **Semester:** Winter Semester 2025/26
* **Academic Supervisor:** Prof. Dr. Michael Spangenberg
* **Company Supervisor:** Thomas Pinsker
