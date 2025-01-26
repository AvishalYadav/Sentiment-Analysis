mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"yadavvishalyadav718@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

python -m nltk.downloader punkt
python -m nltk.downloader stopwords 