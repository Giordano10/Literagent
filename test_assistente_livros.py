# c:\Users\note teste\Documents\Agente RAG\test_assistente_livros.py

import unittest
from unittest.mock import patch, MagicMock
import io

# Import functions from your Streamlit app file.
# Ensure this test file is in the same directory as Assistente_livros.py
# or that the path is configured correctly.
from literagent import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversation_chain,
    handle_user_input,
)

class TestAssistenteLivros(unittest.TestCase):
    """Unit tests for the core functions of the book assistant app."""

    def test_get_pdf_text(self):
        """Tests the PDF text extraction function."""
        # Mock PyPDF2.PdfReader
        mock_pdf_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "This is page 1. "
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is page 2."
        mock_pdf_reader.pages = [mock_page1, mock_page2]

        # Patch the PdfReader class in the context of the Assistente_livros module
        with patch('Assistente_livros.PdfReader', return_value=mock_pdf_reader) as mock_pdf_reader_class:
            # Create a dummy file-like object to simulate an uploaded PDF
            pdf_docs = [io.BytesIO(b"dummy pdf content")]
            text = get_pdf_text(pdf_docs)

            # Assert that PdfReader was instantiated with the dummy file
            mock_pdf_reader_class.assert_called_once_with(pdf_docs[0])
            # Assert that the text from all pages was concatenated
            self.assertEqual(text, "This is page 1. This is page 2.")

    def test_get_text_chunks(self):
        """Tests the text chunking function."""
        # Create a long string to be chunked
        long_text = "a" * 2000
        chunks = get_text_chunks(long_text)

        # Assert that the output is a list and has more than one chunk
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 1)
        # Assert the length of the first chunk matches the chunk_size parameter
        self.assertEqual(len(chunks[0]), 1000)

    @patch('Assistente_livros.FAISS')
    @patch('Assistente_livros.GoogleGenerativeAIEmbeddings')
    def test_get_vector_store_success(self, mock_embeddings, mock_faiss):
        """Tests the vector store creation on success."""
        text_chunks = ["chunk1", "chunk2"]
        api_key = "test_api_key"

        # Mock the return values of the external libraries
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_vector_store_instance = MagicMock()
        mock_faiss.from_texts.return_value = mock_vector_store_instance

        vector_store = get_vector_store(text_chunks, api_key)

        # Assert that the embedding model was called correctly
        mock_embeddings.assert_called_once_with(model="models/embedding-001", google_api_key=api_key)
        # Assert that FAISS was used to create the vector store from texts
        mock_faiss.from_texts.assert_called_once_with(text_chunks, embedding=mock_embeddings_instance)
        # Assert that the function returns the created vector store
        self.assertEqual(vector_store, mock_vector_store_instance)

    @patch('Assistente_livros.st')
    @patch('Assistente_livros.GoogleGenerativeAIEmbeddings')
    def test_get_vector_store_failure(self, mock_embeddings, mock_st):
        """Tests the vector store creation on failure (e.g., bad API key)."""
        text_chunks = ["chunk1", "chunk2"]
        api_key = "invalid_api_key"

        # Configure the mock to raise an exception, simulating an API error
        mock_embeddings.side_effect = Exception("API Key Error")

        vector_store = get_vector_store(text_chunks, api_key)

        # Assert that the function returns None on failure
        self.assertIsNone(vector_store)
        # Assert that error messages were shown to the user via Streamlit
        self.assertEqual(mock_st.error.call_count, 2)
        mock_st.error.assert_any_call("Erro ao criar o vector store: API Key Error")

    @patch('Assistente_livros.ConversationalRetrievalChain')
    @patch('Assistente_livros.ChatGoogleGenerativeAI')
    @patch('Assistente_livros.ConversationBufferMemory')
    def test_get_conversation_chain(self, mock_memory, mock_llm, mock_chain):
        """Tests the creation of the conversational chain."""
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        api_key = "test_api_key"

        # Mock the instances created within the function
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_chain_instance = MagicMock()
        mock_chain.from_llm.return_value = mock_chain_instance

        conversation_chain = get_conversation_chain(mock_vector_store, api_key)

        # Assert that the LLM and memory were configured correctly
        mock_llm.assert_called_once_with(model="gemini-pro", google_api_key=api_key, temperature=0.3)
        mock_memory.assert_called_once_with(memory_key="chat_history", return_messages=True)
        
        # Assert that the final chain was created with the correct components
        mock_chain.from_llm.assert_called_once()
        args, kwargs = mock_chain.from_llm.call_args
        self.assertEqual(kwargs['llm'], mock_llm_instance)
        self.assertEqual(kwargs['retriever'], mock_retriever)
        self.assertEqual(kwargs['memory'], mock_memory_instance)
        self.assertIn('prompt', kwargs['combine_docs_chain_kwargs'])
        self.assertEqual(conversation_chain, mock_chain_instance)

    @patch('Assistente_livros.st')
    def test_handle_user_input_with_conversation(self, mock_st):
        """Tests user input handling when a conversation is active."""
        # Mock the conversation chain, which is a callable
        mock_conversation = MagicMock()
        mock_conversation.return_value = {'answer': 'This is the AI response.'}
        
        # Mock Streamlit's session state
        mock_st.session_state.conversation = mock_conversation
        mock_st.session_state.chat_history = []

        user_question = "What is the summary?"
        handle_user_input(user_question)

        # Assert the conversation chain was called with the user's question
        mock_conversation.assert_called_once_with({'question': user_question})

        # Assert the chat history was updated correctly (user question first, then AI answer)
        expected_history = [
            {"role": "user", "content": "What is the summary?"},
            {"role": "assistant", "content": "This is the AI response."}
        ]
        self.assertEqual(mock_st.session_state.chat_history, expected_history)

        # Assert that the chat messages were displayed
        self.assertEqual(mock_st.chat_message.call_count, 2)
        self.assertEqual(mock_st.markdown.call_count, 2)

    @patch('Assistente_livros.st')
    def test_handle_user_input_no_conversation(self, mock_st):
        """Tests user input handling when no conversation is active."""
        # Set session state to have no active conversation
        mock_st.session_state.conversation = None

        handle_user_input("A question")

        # Assert that a warning is shown to the user
        mock_st.warning.assert_called_once_with("Por favor, processe seus PDFs primeiro.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)