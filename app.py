from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import uvicorn
from House_pred import House_predict

# Load your model
regressor = joblib.load("price_pipeline.pkl")

app = FastAPI()
#pickle_in = open("price_pipeline.pkl","rb")
#regressor=pickle.load(pickle_in)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS (optional if frontend is same host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Boston house price prediction': f'{name}'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted house price with the confidence
@app.post("/predict")
def predict_price(input_data: House_predict):
    data_dict = input_data.dict()
    arr = np.array(list(data_dict.values())).reshape(1, -1)
    prediction = regressor.predict(arr)[0]
    return {"prediction": float(prediction)}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload




#study pickle and joblib uvicorn 
#how to do it for scaled/large which is more effcient