import sys
import os
import gradio as gr

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator import generate_scene


def generate(user_prompt):

    if not user_prompt.strip():
        return "Please enter a scene idea."

    result = generate_scene(user_prompt)

    return result


interface = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Example: Two strangers meet in a bar and start a fight..."
    ),
    outputs=gr.Textbox(lines=20),
    title="Entertainment Content Generator",
    description="Generate screenplay scenes using RAG."
)


if __name__ == "__main__":
    interface.launch()