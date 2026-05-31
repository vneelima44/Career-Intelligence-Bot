import os
from resume_xray import build_ui

port = int(os.environ.get('PORT', 7860))
demo = build_ui()
demo.launch(server_name='0.0.0.0', server_port=port)
