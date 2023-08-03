#!/bin/bash
# straight running python script
# python -m http.server
# npm run serve
# python3 gui_app.py
# sudo systemctl stop gdm
# sudo loginctl terminate-seat seat0
# xinit

streamlit run /home/jetson/streamlit_app/streamlit_app.py --server.headless true
firefox http://localhost:8501

 