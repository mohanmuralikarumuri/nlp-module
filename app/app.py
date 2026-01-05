"""
Dialogue Summarization Web Application

This Streamlit app provides an interactive interface for dialogue summarization
using trained Hugging Face models.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import DialogueSummarizer


# Page configuration
st.set_page_config(
    page_title="Dialogue Summarization",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


def load_model(model_path: str):
    """Load the summarization model."""
    try:
        with st.spinner("Loading model..."):
            summarizer = DialogueSummarizer(model_path)
            st.session_state.summarizer = summarizer
            st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False


def main():
    """Main application function."""
    
    # Title and description
    st.title("💬 Dialogue Summarization System")
    st.markdown("""
    This application uses advanced NLP models to generate concise summaries of dialogues.
    Enter a conversation below and get an AI-generated summary!
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="facebook/bart-large-cnn",
        help="Enter the path to your trained model or a Hugging Face model name"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        load_model(model_path)
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    num_beams = st.sidebar.slider(
        "Number of Beams",
        min_value=1,
        max_value=10,
        value=4,
        help="Higher values produce better quality but slower generation"
    )
    
    length_penalty = st.sidebar.slider(
        "Length Penalty",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls summary length preference"
    )
    
    min_length = st.sidebar.slider(
        "Minimum Length",
        min_value=5,
        max_value=100,
        value=10,
        help="Minimum number of tokens in summary"
    )
    
    max_length = st.sidebar.slider(
        "Maximum Length",
        min_value=20,
        max_value=500,
        value=128,
        help="Maximum number of tokens in summary"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Dialogue")
        
        # Sample dialogues
        sample_dialogue = """Person A: Hey, how was your weekend?
Person B: It was great! I went hiking with some friends in the mountains.
Person A: That sounds amazing. Where exactly did you go?
Person B: We went to the Blue Ridge Mountains. The weather was perfect.
Person A: Did you see any wildlife?
Person B: Yes! We saw a few deer and lots of birds. The views were breathtaking.
Person A: I'd love to go sometime. Any recommendations?
Person B: Definitely try the Sunset Trail. It's moderately difficult but worth it."""
        
        use_sample = st.checkbox("Use sample dialogue", value=False)
        
        if use_sample:
            dialogue_input = st.text_area(
                "Dialogue",
                value=sample_dialogue,
                height=300,
                help="Enter the dialogue you want to summarize"
            )
        else:
            dialogue_input = st.text_area(
                "Dialogue",
                height=300,
                placeholder="Enter dialogue here...\n\nPerson A: Hello!\nPerson B: Hi there!",
                help="Enter the dialogue you want to summarize"
            )
    
    with col2:
        st.subheader("📄 Generated Summary")
        
        # Summary output placeholder
        summary_placeholder = st.empty()
        
        # Generate button
        if st.button("🚀 Generate Summary", type="primary"):
            if not st.session_state.model_loaded:
                st.warning("Please load a model first using the sidebar.")
            elif not dialogue_input.strip():
                st.warning("Please enter a dialogue to summarize.")
            else:
                try:
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.summarizer.summarize(
                            dialogue_input,
                            num_beams=num_beams,
                            length_penalty=length_penalty,
                            min_length=min_length,
                            max_length=max_length
                        )
                    
                    # Display summary
                    summary_placeholder.text_area(
                        "Summary",
                        value=summary,
                        height=300,
                        help="Generated summary"
                    )
                    
                    # Statistics
                    st.success("✅ Summary generated successfully!")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Input Length", f"{len(dialogue_input.split())} words")
                    with col_stat2:
                        st.metric("Summary Length", f"{len(summary.split())} words")
                    with col_stat3:
                        compression_ratio = len(summary.split()) / len(dialogue_input.split()) * 100
                        st.metric("Compression", f"{compression_ratio:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions in expander
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Instructions
        
        1. **Load a Model**: Enter a model path or Hugging Face model name in the sidebar and click "Load Model"
        2. **Enter Dialogue**: Type or paste a dialogue in the input area
        3. **Adjust Parameters**: (Optional) Modify generation parameters in the sidebar
        4. **Generate**: Click the "Generate Summary" button
        5. **Review**: The generated summary will appear in the right panel
        
        ### Tips
        
        - Use higher beam numbers for better quality (but slower generation)
        - Adjust length penalty to control summary length
        - Try different models for different summarization styles
        - The sample dialogue provides a good starting point
        """)


if __name__ == "__main__":
    main()
