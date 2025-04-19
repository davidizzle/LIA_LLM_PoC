import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
import torch
import time
import base64

st.set_page_config(page_title="LIA Demo", layout="wide")
st.markdown("<h1 style='text-align: center;'>Ask LeoNardo!</h1>", unsafe_allow_html=True)

# Load both GIFs in base64 format
def load_gif_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
# Placeholder for GIF HTML
gif_html = st.empty()
caption = st.empty()

gif_html.markdown(
    f"<div style='text-align:center;'><img src='https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTRxYzI2bXJmY3N2bXBtMHJtOGV3NW9vZ3l3M3czbGYybGpkeWQ1YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/3uPWb5EYVvxdfoREQm/giphy.gif' width='300'></div>",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_id = "deepseek-ai/deepseek-llm-7b-chat"
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map=None,
        # torch_dtype=torch.float32
        device_map="auto",
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
        # attn_implementation="flash_attention_2",
        trust_remote_code = True
    )
    # model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()
prompt = st.text_area("Enter your prompt:", "What company is Leonardo S.p.A.?")
# Example prompt selector
# examples = {
#     "🧠 Summary": "Summarize the history of AI in 5 bullet points.",
#     "💻 Code": "Write a Python function to sort a list using bubble sort.",
#     "📜 Poem": "Write a haiku about large language models.",
#     "🤖 Explain": "Explain what a transformer is in simple terms.",
#     "🔍 Fact": "Who won the FIFA World Cup in 2022?"
# }

# selected_example = st.selectbox("Choose a Gemma to consult:", list(examples.keys()) + ["✍️ Custom input"])
# Add before generation
# col1, col2, col3 = st.columns(3)

# with col1:
#     temperature = st.slider("Temperature", 0.1, 1.5, 1.0)

# with col2:
#     max_tokens = st.slider("Max tokens", 50, 500, 100)

# with col3:
#     top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95)
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
        f"<div style='text-align:center;'><img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXViMm02MnR6bGJ4c2h3ajYzdWNtNXNtYnNic3lnN2xyZzlzbm9seSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/k32ddF9WVs44OUaZAm/giphy.gif' width='300'></div>",
        unsafe_allow_html=True,
    )
    caption.markdown("<p style='text-align: center; margin-top: 20px;'>LeoNardo is thinking... 🌀</p>", unsafe_allow_html=True)


    # Generate text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(   **inputs,
                                    # max_new_tokens=100, 
                                    max_new_tokens=256, 
                                    do_sample=True,
                                    temperature=1.0,
                                    top_p=0.95,
                                    top_k=50, 
                                    num_return_sequences=1, 
                                    eos_token_id=tokenizer.eos_token_id
                                    )

    # Back to still
    # gif_html.markdown(
    #     f"<div style='text-align:center;'><img src='data:image/gif;base64,{still_gem_b64}' width='300'></div>",
    #     unsafe_allow_html=True,
    # )
    gif_html.markdown(
        f"<div style='text-align:center;'><img src='https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTRxYzI2bXJmY3N2bXBtMHJtOGV3NW9vZ3l3M3czbGYybGpkeWQ1YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/3uPWb5EYVvxdfoREQm/giphy.gif' width='300'></div>",
        unsafe_allow_html=True,
    )
    caption.empty()


    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
     # Set up placeholder for streaming effect
    output_placeholder = st.empty()
    streamed_text = ""

    for word in decoded_output.split(" "):
        streamed_text += word + " "
        output_placeholder.markdown("### ✨ Output:\n\n" + streamed_text + "▌")
        # slight delay
        time.sleep(0.03)

    # Final cleanup (remove blinking cursor)
    output_placeholder.markdown("### ✨ Output:\n\n" + streamed_text)