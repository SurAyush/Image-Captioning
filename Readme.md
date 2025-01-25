# **Image Captioning Web App**  
🌟 A full-stack web application for generating image captions, combining the power of modern web technologies with state-of-the-art AI models. Built using **React.js** (frontend) and **FastAPI** (backend), this project integrates a custom-trained model leveraging **CLIP**, a Mapping Network, and **GPT-2** for caption generation.

---

## **Features**  
- 🖼️ **Image Captioning**: Generate captions for images using either greedy search or beam search.  
- ⚙️ **Interactive Interface**: A responsive and intuitive UI powered by React.js.  
- 🚀 **AI-Powered Backend**: FastAPI serves as the backend for efficient model inference.  
- 🔧 **Custom Training**: Train the model on your dataset by making minimal configuration changes.

---

## **Getting Started**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/SurAyush/Image-Captioning.git
cd Image-Captioning
```

### **2. Download the Model**
Download the pre-trained model from Hugging Face and save it in the model/trained_model directory.
[Trained-Model](https://huggingface.co/SurAyush/ImageCaptioning)

### **3. Install Dependencies**

#### Backend
```bash
cd server/
pip install -r requirements.txt
fastapi run
```

#### Frontend
```bash
cd client/
npm install
npm run dev
```

## **4. Try Image Captioning**  
Once everything is set up, you can use the web app to:  
- Upload an image.  
- Generate captions using **greedy search** or **beam search**.  

---

## **Training the Model**  

1. **Prepare the Dataset**:  
   - Make the necessary changes in `parse_coco.py` and `train.py` to locate the **MS COCO 2017** dataset.  

2. **Generate Intermediate Dataset**:  
   ```bash
   python parse_coco.py
   ```

### **Train the Model:**
```bash
python train.py
```

### **Adjust Hyperparameters:**

Modify Config.py to fine-tune the hyperparameters for training.

## **Notes**

💡 The current trained model demonstrates promising results but is limited by resource constraints during training. Despite this, it generates captions related to the input image and shows significant potential for improvement with further training.
BLEU is not yet been evaluated as the model is not fully trained so evaluation does not seem very meaningful.

Inconvenience you might face (whole-hearted apologies for that):

Please install any more python libraries if required (and not specified in requirements.txt)
You may use the python venv if you like (not used as the heavy packages were pre-installed in my local machine like pytorch)


For more details, check out my blog: [My Blog Post](https://medium.com/@ayushsur26/implementing-an-image-caption-model-c990cb620d14).

## **Contributions**
Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests.

## **License**
This project is licensed under the MIT License.
