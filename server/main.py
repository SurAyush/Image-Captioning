from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from model.predict import Predictor

app = FastAPI()
model_path = '../model/trained_model/model_epoch_4.pt'
predictor = Predictor(model_path)

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"], 
)

# just a test route
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate_caption")
async def generate_caption(file: UploadFile = File(...), technique: str = 'greedy', beam_width: int = 4):
    
    if technique not in ['greedy' , 'beam']:
        return JSONResponse(
            stus_code=400,
            content={"error":"Only Greedy and Beam Search is available"}
        )
    
    if beam_width < 2 or beam_width > 9:
        return JSONResponse(
            status_code=400,
            content="Beam size is limited from 2 to 9 for limiting computational cost"
        )
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPEG or PNG images are supported."},
        )
    
    try:
        file_bytes = await file.read()  
        image = BytesIO(file_bytes)
        if technique == 'beam':
            out = predictor.predict(image,True,beam_width)
        else:
            out = predictor.predict(image,False)
        return {"captions" : out}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process the image. Error: {str(e)}"},
        )
    





