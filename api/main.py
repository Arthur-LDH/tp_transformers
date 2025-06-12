from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Add more endpoints for your TP, for example:
# @app.post("/predict/")
# async def predict(text: str):
#     # Placeholder for your model prediction logic
#     # You will need to load your trained model here
#     # and use it to make predictions on the input text.
#     prediction = f"Prediction for: {text}" # Replace with actual prediction
#     return {"prediction": prediction}
