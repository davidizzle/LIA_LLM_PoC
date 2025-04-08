import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import base64

st.set_page_config(page_title="Gemma Demo", layout="wide")
# Model selection (STUBBED behavior)
model_option = st.selectbox(
    "Choose a Gemma to reveal hidden truths:",
    ["gemma-2b-it (Instruct)", "gemma-2b", "gemma-7b", "gemma-7b-it"],
    index=0,
    help="Stubbed selection – only gemma-2b-it will load for now."
)
st.markdown("<h1 style='text-align: center;'>Portal to Gemma</h1>", unsafe_allow_html=True)

# Load both GIFs in base64 format
def load_gif_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

still_gem_b64 = load_gif_base64("assets/stillGem.gif")
rotating_gem_b64 = load_gif_base64("assets/rotatingGem.gif")

# Placeholder for GIF HTML
gif_html = st.empty()
caption = st.empty()

# Initially show still gem
# gif_html.markdown(
#     f"<div style='text-align:center;'><img src='data:image/gif;base64,{still_gem_b64}' width='300'></div>",
#     unsafe_allow_html=True,
# )
gif_html.markdown(
    f"<div style='text-align:center;'><img src='https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMG00dmlwbjZsemZ5Mnh2eTIwOGNyYncwbGNqd3U3aHhiNGYxYjgwbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmJtDY3xgYVgXgQZYc/giphy.gif' width='300'></div>",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,
        torch_dtype=torch.float32
    )
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()
prompt = st.text_area("Enter your prompt:", "What is Gemma?")
# # Example prompt selector
# examples = {
#     "🧠 Summary": "Summarize the history of AI in 5 bullet points.",
#     "💻 Code": "Write a Python function to sort a list using bubble sort.",
#     "📜 Poem": "Write a haiku about large language models.",
#     "🤖 Explain": "Explain what a transformer is in simple terms.",
#     "🔍 Fact": "Who won the FIFA World Cup in 2022?"
# }

# selected_example = st.selectbox("Choose a Gemma to consult:", list(examples.keys()) + ["✍️ Custom input"])
# Add before generation
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0)

with col2:
    max_tokens = st.slider("Max tokens", 50, 500, 100)

with col3:
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95)
# if selected_example != "✍️ Custom input":
#     prompt = examples[selected_example]
# else:
#     prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    # Swap to rotating GIF
    # gif_html.markdown(
    #     f"<div style='text-align:center;'><img src='data:image/gif;base64,{rotating_gem_b64}' width='300'></div>",
    #     unsafe_allow_html=True,
    # )
    gif_html.markdown(
        f"<div style='text-align:center;'><img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXB0ZTEycW1yYWhvZWExdHFyNzBnemdtdm80NzY0MGg1ZnkyNTRqbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/FIMRlbkylLyniVP7WG/giphy.gif' width='300'></div>",
        unsafe_allow_html=True,
    )
    caption.markdown("<p style='text-align: center;'>Gemma is thinking... 🌀</p>", unsafe_allow_html=True)


    # Generate text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)

    # Back to still
    # gif_html.markdown(
    #     f"<div style='text-align:center;'><img src='data:image/gif;base64,{still_gem_b64}' width='300'></div>",
    #     unsafe_allow_html=True,
    # )
    gif_html.markdown(
        f"<div style='text-align:center;'><img src='https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMG00dmlwbjZsemZ5Mnh2eTIwOGNyYncwbGNqd3U3aHhiNGYxYjgwbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WmJtDY3xgYVgXgQZYc/giphy.gif' width='300'></div>",
        unsafe_allow_html=True,
    )
    caption.empty()


    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown("### ✨ Output:")
    st.write(result)