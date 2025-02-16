import gradio as gr

with gr.Blocks() as demo:
    gr.Chatbot([
        ("Show me an image and an audio file", "Here is an image"), 
        (None, ("lion.png",)), 
        (None, "And here is an audio file:"), 
    ])

demo.launch()
