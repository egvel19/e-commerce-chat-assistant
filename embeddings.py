import openai
import pandas as pd

MODEL_FOR_EMBEDDING = "text-embedding-ada-002"
BATCH_SIZE = 100

# Function to generate embedding for user input
def get_embedding(input_text):
    response = openai.embeddings.create(
        model=MODEL_FOR_EMBEDDING,
        input=input_text
    )

    if response.data[0].embedding:
        return response.data[0].embedding
    else:
        return []


# Send batches to API to optimize embedding process 
def get_embeddings_batch(inputs):
    embeddings = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[i:i + BATCH_SIZE]
        response = openai.embeddings.create(
            model=MODEL_FOR_EMBEDDING,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings


def update_product_data_with_embeddings(product_data_file):
    product_df = pd.read_csv(product_data_file)
    
    # If 'embedded' does not exist, add it 
    if 'embedded' not in product_df.columns:
        combined_text = lambda row: f"{row['Product Name']} {row['About Product']} {row['Product Specification']}"
        inputs = [combined_text(row) for _, row in product_df.iterrows()]
        embeddings = get_embeddings_batch(inputs)
        product_df['embedded'] = embeddings
        product_df.to_csv(product_data_file, index=False)
    
    return product_df
