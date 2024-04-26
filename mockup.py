import streamlit as st
import numpy as np
import pandas as pd
import time
#from bert_score import score

# @st.cache
# def calculate_bertscore(reference, generated):
#     _, _, bertscore = score([reference], [generated], lang='en', model_type='bert-base-uncased')
#     return bertscore.item()

# Set the page configuration
st.set_page_config(
    page_title="My Summeval App",
    page_icon=":memo:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# columns layout
col1, spacer , col2 = st.columns([2, 0.01, 0.5])

with st.sidebar:
    # admin/user view
    option = st.sidebar.radio("Choose a page", ("Admin view", "User view"))

    if option == "Admin view":
        st.write("Admin view selected.")
    else:
        st.write("User view selected.")

    # Selectbox
    evaluation_metric = st.selectbox("Choose an evaluation metric", ("ROUGE", "BLEU", "METEOR", "BERTScore", "MoverScore", "BERT-based metrics", "QuestEval"))
    if evaluation_metric == "ROUGE":
        st.markdown("""
            - **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
            - Widely used for summarization.
            - Measures the overlap of n-grams between the generated summary and reference summaries.
            - Focuses on recall by assessing how much of the reference content appears in the generated text.
            - For more information, you can refer to the [ROUGE website](https://www.aclweb.org/anthology/W04-1013.pdf).
        """)
    elif evaluation_metric == "BLEU":
        st.markdown("""
            - **BLEU (Bilingual Evaluation Understudy)**:
                - Originally developed for machine translation, but also used for summarization.
                - Measures the precision of n-grams in the generated text against the reference texts.
                - Generally less favored for summarization due to its focus on exact matches.
            - For more information, you can refer to the [BLEU paper](https://www.aclweb.org/anthology/P02-1040.pdf).
        """)
    elif evaluation_metric == "METEOR":
        st.markdown("""
            - **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**:
                - Initially for translation, applicable to summarization.
                - Incorporates synonymy and stemming, offering a more nuanced understanding of language.
            - For more information, you can refer to the [METEOR paper](https://www.aclweb.org/anthology/W05-0909.pdf).
        """)
    elif evaluation_metric == "BERTScore":
        st.markdown("""
            - **BERTScore**:
                - Leverages the contextual embeddings from models like BERT.
                - Evaluates semantic similarity between the generated text and the reference.
                - Considers the context of words rather than relying solely on exact word matches.
            - For more information, you can refer to the [BERTScore paper](https://arxiv.org/abs/1904.09675).
        """)
    elif evaluation_metric == "MoverScore":
        st.markdown("""
            - **MoverScore**:
                - Uses contextual embeddings (from BERT-like models).
                - Aligns embeddings between the source and generated summaries using Earth Mover's Distance.
                - Measures the minimal distance that embeddings of the generated summary need to "move" to match the embeddings of the reference summary.
            - For more information, you can refer to the [MoverScore paper](https://arxiv.org/abs/1909.02622).
        """)
    elif evaluation_metric == "BERT-based metrics":
        st.markdown("""
            - **BERT-based metrics (e.g., BLEURT, BARTScore)**:
                - Involve training evaluation models on a mix of human judgments and automated metrics.
                - BLEURT is trained on human judgments and other metrics.
                - BARTScore utilizes a pretrained model (BART) to predict the likelihood of a reference given a summary, considering both fluency and factual consistency.
            - For more information, you can refer to the [BLEURT paper](https://arxiv.org/abs/2004.04696) and the [BARTScore paper](https://arxiv.org/abs/1910.13461).
        """)
    elif evaluation_metric == "QuestEval":
        st.markdown("""
            - **QuestEval**:
                - Assesses the quality of summaries by generating questions and answers based on both the source text and the generated summary.
                - The degree to which answers from the generated summary align with answers from the source text indicates the summary’s faithfulness and relevance.
            - For more information, you can refer to the [QuestEval paper](https://arxiv.org/abs/2101.00143).
        """)

def initialize_scores(df):
    for score in ['BERTScore', 'ROUGE', 'METEOR', 'MoverScore', 'BLEU', 'BARTScore', 'QuestEval']:
        if score not in df.columns:
            df[score] = '0'
    return df

with col1:

    # Title
    st.title(":memo: My Summeval App")

    # Header
    #st.header("Welcome to the summary evaluation app!")

    # Text
    st.subheader("This website allows you to evaluate summaries using different metrics and also provides the option to evaluate your own summary.")

    # Markdown
    # st.markdown("## This is a markdown heading")

    # Upload and display DataFrame
    with st.expander("Upload your summary", expanded=True):
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.success("File uploaded successfully!")
            
            st.table(df)  # Display the DataFrame

            for column in ['BERTScore', 'ROUGE', 'METEOR', 'MoverScore', 'BLEU', 'BARTScore', 'QuestEval']:
                df[column] = 0

    # Create seven columns for the buttons
    mcol1, mcol2 = st.columns([1, 7])
    
    with mcol1:
        if uploaded_file is not None:
            if st.button("Compute BERTScore"):
                df['BERTScore'] = 0.9574
            if st.button("Compute ROUGE"):
                df['ROUGE'] = 0.8342
            if st.button("Compute METEOR"):
                df['METEOR'] = 0.7893
            if st.button("Compute MoverScore"):
                df['MoverScore'] = 0.7223
            if st.button("Compute BLEU"):
                df['BLEU'] = 0.7744
            if st.button("Compute BARTScore"):
                df['BARTScore'] = 0.6951
            if st.button("Compute QuestEval"):
                df['QuestEval'] = 0.6512

    
    with mcol2:
        # Display the updated DataFrame
        if uploaded_file is not None:
            st.table(df[['BERTScore', 'ROUGE', 'METEOR', 'MoverScore', 'BLEU', 'BARTScore', 'QuestEval']])

            
            
    

with col2:
    # library with stat of art LLM models like huggingface, weights & biases, Llamas, OpenAI, 
    st.markdown("""
        #### Libraries with state-of-the-art LLM models:
        
        - **OpenAI**: [GPT-4](https://openai.com/research/gpt-4)
        - **Llamas**: [Llama3](https://ai.meta.com/blog/meta-llama-3/)
        - **Google PalM**: [PalM2](https://ai.google/discover/palm2)
        - **Anthropic**: [Anthropic Claude](https://www.anthropic.com/claude)
        - **Hugging Face**: [StarCoder](https://huggingface.co/blog/starcoder)
                
        #### Additional libraries with useful tools:
        
        - **Weights & Biases**: [WandB](https://wandb.ai/)
        - **Hugging Face**: [Hugging Face](https://huggingface.co/)
        - **Langchain**: [Langchain](https://langchain.com/)
        - **TensorFlow**: [TensorFlow](https://www.tensorflow.org/)
        - **PyTorch**: [PyTorch](https://pytorch.org/)
    """)






# Button
# if st.button("Click me"):
#     st.write("Button clicked!")

# # Checkbox
# checkbox_state = st.checkbox("Check me")
# if checkbox_state:
#     st.write("Checkbox checked!")

# # Radio buttons
# radio_button = st.radio("Choose an option", ("Option 1", "Option 2", "Option 3"))
# st.write("Selected option:", radio_button)

# # Selectbox
# selectbox_option = st.selectbox("Choose an option", ("Option 1", "Option 2", "Option 3"))
# st.write("Selected option:", selectbox_option)

# # Slider
# slider_value = st.slider("Choose a value", 0, 10)
# st.write("Selected value:", slider_value)


# # Plotting
# import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y)

# # st.pyplot(fig, dpi=80)

# # Dataframe

# data = {
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [25, 30, 35],
#     "City": ["New York", "London", "Paris"]
# }

# df = pd.DataFrame(data)
# st.dataframe(df)

# # Table
# st.table(df)

# # Expander
# with st.expander("Click to expand"):
#     st.write("This is some hidden content.")

# # Show code
# with st.echo():
#     st.write("This is the code.")

# # Show JSON
# st.json({"name": "John", "age": 30})

# # Show progress

# progress_bar = st.progress(0)
# for i in range(100):
#     progress_bar.progress(i + 1)
#     time.sleep(0.1)

# # Show success message
# st.success("Task completed successfully!")

# # Show error message
# st.error("An error occurred.")

# # Show warning message
# st.warning("This is a warning.")

# # Show info message
# st.info("This is an info message.")