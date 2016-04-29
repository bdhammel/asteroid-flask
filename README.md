http://brew.sh

brew install freetype
brew install gcc
brew install python
brew install redis
brew install node

easy_install pip

pip install -r requirements.txt

redis-server
python worker.py
python game
