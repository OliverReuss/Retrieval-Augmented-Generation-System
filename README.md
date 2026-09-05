# Bachelor Thesis: When Websites Learn to Speak: Implementation of a Retrieval-Augmented Generation System Based on Web-Scraped Data

## Description

A Retrieval-Augmented Generation (RAG) system based on web-scraped data was implemented. The project contains the complete pipeline, from data extraction to system usage and automated evaluation.

## Note

To facilitate the evaluation of the project, the collected data and locally stored models have been included. To test the data collection and model download processes, the `data` and `models` directories can be deleted. An OpenAI API key with available credits has been stored in the `.env` file for running the system and performing the evaluation.

## Requirements and Installation

* Python >= 3.11.9

* Recommended: Create a virtual environment using `python -m venv .venv` and `.venv\Scripts\activate`

* Install the dependencies from `requirements.txt` using `pip install -r requirements.txt`

* An OpenAI account with available credits is required for evaluation and generation, and an API key must be stored in the `.env` file (`OPENAI_API_KEY=VALUE`)
  *(Note: A complete test run costs less than €1.)*

* As an alternative to OpenAI, a free model from OpenRouter can also be used for generation (but not evaluation) (`OPENROUTER_API_KEY=VALUE`)

* Optional but recommended: Create a Hugging Face account and provide an API key for faster model downloads (`HF_TOKEN=VALUE`)

## Usage

Running `main.py` starts the entire process and creates all required data and/or downloads the necessary models. The user can choose whether to perform an evaluation or proceed directly to using the system. Interaction with the system takes place through input and output in the terminal.

## Configuration

* The system can be configured through `config.py`

* Proxy server (optional): `PROXY`

* SSL certificate (optional): `CERTIFICATE`

* LLM provider: `GENERATION_API_URL`

* LLM provider API key: `GENERATION_API_KEY`

* LLM model: `GENERATION_MODEL`

## Information

* Author: Oliver Reuß

* Submitted to: Prof. Dr. Michael Spangenberg

* Company Supervisor: Thomas Pinsker

* Hof University of Applied Sciences | Faculty of Computer Science | Media Informatics Program | Winter Semester 2025/26
