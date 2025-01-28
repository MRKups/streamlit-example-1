import streamlit as st
from ollama import Client
import PyPDF2 # for PDF conversion
import os # Make sure this is at the top of your file

# Initialize session state for Ollama configuration and client
if 'ollama_host' not in st.session_state:
    st.session_state.ollama_host = "http://localhost:11434"
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = "llama3.2"
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = None
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "Not Connected"
if 'last_successful_config' not in st.session_state:
    st.session_state.last_successful_config = None


def initialize_ollama_client(host, model):
    # Try connection to ollama
    try:
        client = Client(host=host) # Create a connection
        client.show(model) # Get a list of available models

        return True, None # If we get here, both connection and model verification succeeded

    # This will raise an exception if either:
    # 1. The connection fails
    # 2. The model doesn't exist
    # And will return the error string.
    except Exception as e:
            return False, f"Unexpected error: {str(e)}"


def update_connection():
    """
    Update the Ollama client connection based on current configuration.
    Updates session state with connection status and client instance.
    """
    success, error = initialize_ollama_client(
        st.session_state.ollama_host,
        st.session_state.ollama_model
    )

    if success:
        st.session_state.ollama_client = Client(host=st.session_state.ollama_host)
        st.session_state.connection_status = "Connected"
        st.session_state.last_successful_config = {
            'host': st.session_state.ollama_host,
            'model': st.session_state.ollama_model
        }
    else:
        st.session_state.ollama_client = None
        st.session_state.connection_status = f"Connection Failed: {error}"


def generate_ollama_response(user_prompt, system_prompt):
    """
    Generates a response from Ollama based on user and system prompts.
    Utilizes the Ollama client and model from the session state.
    """
    if not user_prompt:
        st.warning("Please enter a prompt before generating.")
        return

    try:
        # Initialize response container
        response_container = st.empty()
        full_response = ""

        # Generate streaming response using existing client
        for chunk in st.session_state.ollama_client.generate(
                model=st.session_state.ollama_model,
                prompt=user_prompt,
                system=system_prompt if system_prompt else None,
                stream=True
        ):
            if 'response' in chunk:
                full_response += chunk['response']
                # Update the response container with accumulated text
                response_container.markdown(full_response)

        # Display debug information
        with st.expander("Debug Information"):
            st.code(
                f"Model: {st.session_state.ollama_model}\n"
                f"Host: {st.session_state.ollama_host}\n"
                f"System Prompt Length: {len(system_prompt)}\n"
                f"User Prompt Length: {len(user_prompt)}"
            )
    except Exception as e:
        st.error(f"An error occurred while generating: {str(e)}")
        # If we get an error, try to reconnect
        update_connection()


def read_document_content(uploaded_file):
    """
    Reads the content of the uploaded document based on its file extension.
    Supported extensions: TXT, PDF, MD, and various code files.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower() # Get file extension

    text_extensions = [".txt", ".md", ".js", ".py", ".cs", ".go", ".html", ".css", ".xml", ".json"] # Add file extensions here, stick to text-only files.

    if file_extension in text_extensions:
        try:
            return uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading text-based content: {e}")
            return None
    elif file_extension == ".pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF content: {e}")
            return None
    else:
        st.warning(f"Unsupported file type. Please upload TXT, PDF, MD, or code files ({', '.join(text_extensions[2:])}).") # Updated warning message
        return None


# Web page config
st.set_page_config(layout="wide")

# Sidebar area
with st.sidebar:
    st.title("LLM Toolbox")
    st.divider()

    # Ollama configuration with connection management
    new_host = st.text_input(
        "Ollama Host",
        value=st.session_state.ollama_host,
        help="Example: http://localhost:11434"
    )

    new_model = st.text_input(
        "Model Name",
        value=st.session_state.ollama_model,
        help="Example: llama3.2, phi4, deepseek-r1"
    )
    if st.button("Connect"):
        st.session_state.ollama_host = new_host
        st.session_state.ollama_model = new_model
        update_connection()

    # Display connection status
    status_color = "green" if st.session_state.connection_status == "Connected" else "red"
    st.markdown(f"Status: :{status_color}[{st.session_state.connection_status}]")

    st.divider()

    tab_selection = st.radio(
        "Functions",
        ["Query", "Document Summary"],
        key="tab_selector"
    )

    st.divider()

# Main content area - based on selected tab
if tab_selection == "Query":
    st.header("Query")

    if st.session_state.ollama_client is None:
        st.warning("Please establish a connection to Ollama first using the sidebar.")
    else:
        # System prompt input
        system_prompt = st.text_area(
            "System Prompt",
            height=150,
            value="You are a helpful AI assistant.",
            help="Provide context or instructions for the model"
        )

        # User prompt input
        user_prompt = st.text_area(
            "User Prompt",
            height=150,
            help="Enter your message here"
        )

        # Generate button
        if st.button("Generate Response"):
            generate_ollama_response(user_prompt, system_prompt) # Call the function here

elif tab_selection == "Document Summary":
    st.header("Document Summary")

    if st.session_state.ollama_client is None:
        st.warning("Please establish a connection to Ollama first using the sidebar.")
    else:
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=[".pdf", ".txt", ".md", ".js", ".py", ".cs", ".go", ".html", ".css", ".xml", ".json"], # Accepted formats
            help="Supported formats: .pdf, .txt, .md, .js, .py, .cs, .go, .html, .css, .xml, .json"
        )

        if uploaded_file:
            document_content = read_document_content(uploaded_file) # Use the function to read content
            if document_content: # Check if content was successfully read
                # System prompt for document summary
                system_prompt = "You are a document summarization expert. Please provide a concise and accurate summary of the document."
                # User prompt (instruction + document content)
                user_prompt = "Analyze this document and produce an accurate and concise summary:\n\n" + document_content

                st.subheader("Document Content Preview:")
                st.text_area("Document Preview", value=document_content, height=200)

                if st.button("Generate Summary"):
                    generate_ollama_response(user_prompt, system_prompt)
            # Error message is handled inside read_document_content now
        else:
            st.info("Please upload a document to summarize.")
