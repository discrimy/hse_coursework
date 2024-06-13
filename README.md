```bash
sudo apt install ffmpeg sox -y

poetry install --sync
poetry shell

python cli.py video.mp4 output.json
```