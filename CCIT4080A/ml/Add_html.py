import base64

class Add_html(object):
    def __init__(self):
        a=1
    # autoplay_audio(self, file_path: str):
    # Reads an audio file from the specified file_path, encodes it in base64, and generates an HTML <audio> tag with autoplay enabled.
    def autoplay_audio(self, file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            return md
