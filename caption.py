'''from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokeninzer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
        
    pixel_values = feature_extractor(
        images = images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values)
    
    preds = tokeninzer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print("Final Caption is:",preds)
    return preds

predict_caption(['sisters.png'])'''

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import cv2

# Initialize model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define prediction function
def predict_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
        
    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values)
    
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Create GUI application
class CaptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioning App")
        self.geometry("1200x800")
        
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)
        
        self.caption_label = tk.Label(self, text="Predicted Caption:")
        self.caption_label.pack(pady=10)
        
        self.caption_text = tk.Text(self, height=5, width=50)
        self.caption_text.pack(pady=10)
        
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)
        
        self.camera_button = tk.Button(self, text="Use Camera", command=self.use_camera)
        self.camera_button.pack(pady=5)
        
        self.predict_button = tk.Button(self, text="Predict Caption", command=self.predict_caption)
        self.predict_button.pack(pady=10)
        
        self.camera = cv2.VideoCapture(0)
        
    def upload_image(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png"), ("Image files", "*.jpg"), ("Image files", "*.jpeg")])        
        if file_paths:
            for file_path in file_paths:
                self.display_image(file_path)

    def use_camera(self):
        ret, frame = self.camera.read()
        if ret:
            cv2.imwrite("camera_image.png", frame)
            self.display_image("camera_image.png")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        
        self.image_path = file_path
        
    def predict_caption(self):
        try:
            caption = predict_caption([self.image_path])
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, caption[0])
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the application
if __name__ == "__main__":
    app = CaptionApp()
    app.mainloop()
