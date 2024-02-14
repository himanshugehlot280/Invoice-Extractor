import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import os

# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Set Streamlit page configuration
    st.set_page_config(page_title="Invoice Extractor")
    st.subheader("I can help you in Invoices Extraction process")

    # File uploader for PDF invoices
    pdf = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    # Button to trigger the publishing process
    submit = st.button("Publish")

    # Initialize an empty invoice description
    invoice_description = ''

    # Connect to Pinecone service(Requierd Only for fetching count)
    # pinecone = Pinecone(api_key=os.environ['PINECONE_API_KEY'], environment="gcp-starter") 
    pinecone = Pinecone(
    api_key=os.environ['PINECONE_API_KEY'], environment="gcp-starter"
    )
    index = pinecone.Index("csv")  
    # vector_count contain a dictionary 
    vector_count = index.describe_index_stats() 
    # total_vector_count is a key and its value is pincone vectors( In your pinecone database)
    count = vector_count["total_vector_count"]

    # Initialize an empty string for extracted data
    extracted_data = ""

    # Create embeddings instance
    embeddings = create_embeddings_load_data()

    if submit:
        with st.spinner('Wait for it...'):

            # Create a unique ID to query and retrieve only user-uploaded documents from Pinecone vector store
            st.session_state['unique_id'] = uuid.uuid4().hex

            # Create a list of documents from user-uploaded PDF files
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])

            # Display the count of uploaded documents
            st.write("*Documents uploaded* :" + str(len(final_docs_list)))

            # Push data to Pinecone
            push_to_pinecone(os.environ['PINECONE_API_KEY'], "gcp-starter", "csv", embeddings, final_docs_list)

    # Button to trigger the extraction process
    extract = st.button("Extract")

    if extract:
        # Fetch relevant documents from Pinecone
        relevant_docs = similar_docs(invoice_description, count, os.environ['PINECONE_API_KEY'], "gcp-starter", "csv", embeddings)

        st.write(":heavy_minus_sign:" * 30)

        # Create an empty DataFrame to store data from all CSVs
        combined_df = pd.DataFrame()

        # Iterate through relevant documents and extract data 
        # Basically Iterate all over pinecone vector and convert it to df 
        for item in range(len(relevant_docs)):
            result = relevant_docs[item][0].page_content
            extracted_data = extract_data(result)
            df = to_df(extracted_data)

            # Append the current DataFrame to the combined DataFrame 
            # Combined Final data frame
            combined_df = combined_df._append(df, ignore_index=True)

        # Display the combined DataFrame
        st.write(combined_df)

        # Save the combined data to a CSV file
        # combined_df.to_csv("combined_data.csv", index=False)
        st.write("Combined data saved to 'combined_data.csv'")
        st.success("Hope I was able to save your time❤️")

# Invoking the main function
if __name__ == '__main__':
    main()
