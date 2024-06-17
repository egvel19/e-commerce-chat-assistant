import openai
import numpy as np
import json
import logging
from numpy.linalg import norm
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import copy
from contextlib import asynccontextmanager
from embeddings import get_embedding, update_product_data_with_embeddings

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

openai.api_key = 'MY_API_KEY'
MAIN_MODEL = "gpt-3.5-turbo"
MAXIMUM_TOKENS = 150

with open('initial_prompt.txt', 'r') as file:
    initial_prompt = file.read()

with open('second_prompt.txt', 'r') as file:
    second_prompt = file.read()

def find_product_by_name(product_name, product_data):
    product_name_cleaned = product_name.strip().lower()
    for product in product_data:
        if product_name_cleaned in product['Product Name'].strip().lower():
            return product
    return None

def handle_price_inquiry(product_data, product_name):
    product = find_product_by_name(product_name, product_data)
    if product:
        return f"The price of {product['Product Name']} is ${product['Selling Price']}."
    else:
        return "Sorry, I couldn't find the price for that product."

# Function to find similar products according to user's input
def cosine_similarity(user_embedding, data, threshold, limit):
    data_cp = copy.deepcopy(data)
    for item in data_cp:
        product_embedding = item["embedded"]
        dot_product = np.dot(product_embedding, user_embedding)
        norm_product = norm(product_embedding) * norm(user_embedding)
        cosine_sim = dot_product / norm_product if norm_product != 0 else 0.0
        item["cosine"] = cosine_sim

    sorted_data = sorted(data_cp, key=lambda x: x["cosine"], reverse=True)
    top_matches = [item for item in sorted_data if item["cosine"] > threshold][:limit]
    return top_matches

def api_initial_request(conversation_history):
    response = openai.chat.completions.create(
        model=MAIN_MODEL,
        messages=conversation_history,
        max_tokens=MAXIMUM_TOKENS
    )
    return response.choices[0].message.content.strip()


def api_generate_response(system_message):
    response = openai.chat.completions.create(
        model=MAIN_MODEL,
        messages=system_message,
        max_tokens=MAXIMUM_TOKENS
    )
    return response.choices[0].message.content.strip()


class UserRequest(BaseModel):
    message: str

# Lifespan context manager to handle startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global product_data
    product_df = update_product_data_with_embeddings('sample_data_10k.csv')
    product_data = product_df.to_dict(orient='records')
    yield

app = FastAPI(lifespan=lifespan, startup_timeout=10000)

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    conversation_history = [{"role": "system", "content": initial_prompt}]

    while True:
        try:
            user_message = await websocket.receive_text()
            conversation_history.append({"role": "user", "content": user_message})

            # First API call
            base_requests = api_initial_request(conversation_history)
            base_requests_json = json.loads(base_requests)

            # If user is asking product information
            if base_requests_json.get("asks_for_product"):

                # if intent is price inquiry of specific product, handle differently 
                if base_requests_json.get("intent") == "Asking about specific product price":
                    product_name = base_requests_json.get("product_name") 
                    response_text = handle_price_inquiry(product_data, product_name)
                
                # If the user is searching for a product or asking about a product:
                else:
                    user_embedding = get_embedding(user_message)
                    similar_products = cosine_similarity(user_embedding, product_data, threshold=0.6, limit=5)

                    if similar_products:
                        products_list = [
                            {
                                "Product Name": product['Product Name'], 
                                "Selling Price": product['Selling Price'], 
                                "Uniq Id": product['Uniq Id'], 
                                "About Product": product['About Product']
                            } 
                            for product in similar_products
                        ]

                        conversation_input_for_api = conversation_history.copy()

                        system_message = [{
                            "role": "system",
                            "content": (
                                "Here is the prompt according which you should generate response:\n"
                                f"{second_prompt}\n\n"
                                "Here is the user intent:\n"
                                f"{base_requests_json.get('intent')}\n\n"
                                "Here is the conversation history:\n"
                                f"{conversation_input_for_api}\n\n"
                                "Here is the product_list. you can find the Product Name, Selling Price, Unique id, and Description of relevant products here:\n"
                                f"{json.dumps(products_list, indent=2)}\n\n"
                            )
                        }]

                        # Second API call for generating final response 
                        response_text = api_generate_response(system_message)
                    else: 
                        response_text = "Sorry, no products match your request. Could you provide more specific information?"

            else: # If the user has started general conversation, chat bot should respond accordingly 
                response_text = base_requests_json.get("general_conversation_answer", "How can I assist you today?")

            conversation_history.append({"role": "assistant", "content": response_text})
            await websocket.send_text(json.dumps({"role": "assistant", "content": response_text}))
            
        except Exception as e:
            logging.error(f"Error: {e}")
            await websocket.send_text(json.dumps({"role": "error", "content": "An error occurred. Please try again later."}))
            break


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)