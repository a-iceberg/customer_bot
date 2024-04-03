# sudo docker compose up --build --force-recreate

sudo docker compose up --build -d --remove-orphans --force-recreate

export LANGCHAIN_API_KEY=""
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_PROJECT="customer_bot"
