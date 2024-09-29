# Setup
## Install python env
```
conda create -n <env_name> python=3.9
conda activate <env_name>
pip install -r requirement.txt
```

## Install Ollama
[Install Ollama on linux](https://github.com/ollama/ollama/blob/main/docs/linux.md)

# Run chatbot
Remember to start ollama and activate environment before starting chatbot.
```
sudo systemctl start ollama
conda activate <env_name>
```
## Run with CLI
```
python app.py -m <model_name> -e <embedding_model> -p <path_to_documents>
```
 If no model is specified, it defaults to [llama3.1](https://ollama.com/library/llama3.1). If no path is specified, it defaults to `Document` located in the repository for example purposes.
## Run with UI (streamlit)
```
streamlit run ui.py
```