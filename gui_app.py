import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import sys
import os
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath("."))

from src.inference import YearPredictor
from src.train import run_training

# Configuration
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PhotoDate AI - Year Estimator")
        self.geometry("900x700")

        # Grid configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create Tabview
        self.tabview = ctk.CTkTabview(self, width=800, height=600)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.tab_inference = self.tabview.add("Inference")
        self.tab_training = self.tabview.add("Training")
        
        self.setup_inference_tab()
        self.setup_training_tab()

        # Load Model for Inference
        self.predictor = None
        self.load_model()

    def load_model(self):
        model_path = "models/best_model.pth"
        if os.path.exists(model_path):
            try:
                self.predictor = YearPredictor(model_path)
                self.status_label.configure(text="Model Loaded Successfully", text_color="green")
            except Exception as e:
                self.status_label.configure(text=f"Error Loading Model: {e}", text_color="red")
        else:
            self.status_label.configure(text="Model not found. Please train or download one.", text_color="orange")

    # ==========================================================================
    # INFERENCE TAB
    # ==========================================================================
    def setup_inference_tab(self):
        self.tab_inference.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_label = ctk.CTkLabel(self.tab_inference, text="Photo Year Estimator", font=ctk.CTkFont(size=24, weight="bold"))
        self.header_label.grid(row=0, column=0, padx=20, pady=10)
        
        # Image Display
        self.image_frame = ctk.CTkFrame(self.tab_inference, width=400, height=400)
        self.image_frame.grid(row=1, column=0, padx=20, pady=10)
        self.image_label = ctk.CTkLabel(self.image_frame, text="No Image Selected")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Buttons
        self.btn_frame = ctk.CTkFrame(self.tab_inference, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, padx=20, pady=10)
        
        self.upload_btn = ctk.CTkButton(self.btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10)
        
        self.predict_btn = ctk.CTkButton(self.btn_frame, text="Estimate Year", command=self.predict_year, state="disabled")
        self.predict_btn.grid(row=0, column=1, padx=10)
        
        # Result
        self.result_label = ctk.CTkLabel(self.tab_inference, text="Estimated Year: --", font=ctk.CTkFont(size=20))
        self.result_label.grid(row=3, column=0, padx=20, pady=20)
        
        # Status
        self.status_label = ctk.CTkLabel(self.tab_inference, text="", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=4, column=0, padx=20, pady=5)
        
        self.current_image = None
        self.current_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.current_image_path = file_path
            
            # Display Image
            img = Image.open(file_path)
            # Resize for display
            img.thumbnail((400, 400))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            
            self.image_label.configure(image=ctk_img, text="")
            self.predict_btn.configure(state="normal")
            self.result_label.configure(text="Estimated Year: --")
            
            # Keep reference to avoid garbage collection
            self.image_label.image = ctk_img
            
            # Store numpy array for prediction
            # Re-open to ensure we get the raw data
            pil_image = Image.open(file_path).convert('RGB')
            self.current_image = np.array(pil_image)

    def predict_year(self):
        if self.predictor and self.current_image is not None:
            try:
                year = self.predictor.predict(self.current_image)
                self.result_label.configure(text=f"Estimated Year: {year:.0f}")
            except Exception as e:
                self.status_label.configure(text=f"Prediction Error: {e}", text_color="red")

    # ==========================================================================
    # TRAINING TAB
    # ==========================================================================
    def setup_training_tab(self):
        self.tab_training.grid_columnconfigure(0, weight=1)
        
        # Credentials Frame
        self.cred_frame = ctk.CTkFrame(self.tab_training)
        self.cred_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.cred_frame, text="AWS Credentials").grid(row=0, column=0, columnspan=2, pady=5)
        
        self.entry_access_key = ctk.CTkEntry(self.cred_frame, placeholder_text="AWS Access Key ID", width=300)
        self.entry_access_key.grid(row=1, column=0, padx=10, pady=5)
        
        self.entry_secret_key = ctk.CTkEntry(self.cred_frame, placeholder_text="AWS Secret Access Key", width=300, show="*")
        self.entry_secret_key.grid(row=1, column=1, padx=10, pady=5)
        
        self.entry_bucket = ctk.CTkEntry(self.cred_frame, placeholder_text="S3 Bucket Name", width=300)
        self.entry_bucket.grid(row=2, column=0, padx=10, pady=5)
        
        # Start Button
        self.train_btn = ctk.CTkButton(self.tab_training, text="Start Training", command=self.start_training_thread)
        self.train_btn.grid(row=1, column=0, padx=20, pady=10)
        
        # Logs
        self.log_box = ctk.CTkTextbox(self.tab_training, width=700, height=400)
        self.log_box.grid(row=2, column=0, padx=20, pady=10)
        self.log_box.configure(state="disabled")

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def start_training_thread(self):
        access_key = self.entry_access_key.get()
        secret_key = self.entry_secret_key.get()
        bucket_name = self.entry_bucket.get()
        
        if not all([access_key, secret_key, bucket_name]):
            self.log("❌ Error: Please provide all AWS credentials.")
            return
            
        self.train_btn.configure(state="disabled", text="Training in Progress...")
        self.log("Starting training thread...")
        
        # Run in separate thread
        thread = threading.Thread(target=self.run_training_process, args=(access_key, secret_key, bucket_name))
        thread.start()

    def run_training_process(self, access_key, secret_key, bucket_name):
        # This runs in a separate thread
        try:
            # Use a lambda to schedule the log update on the main thread (tkinter is not thread safe)
            # But ctk often handles this okay, or we can use .after. 
            # For simplicity, we'll try direct call, if it crashes we need a queue.
            # Actually, let's use a safe wrapper.
            
            def safe_log(msg):
                # Schedule log update on main thread
                self.after(0, lambda: self.log(msg))
                
            run_training(access_key, secret_key, bucket_name, log_callback=safe_log)
            
        except Exception as e:
            self.after(0, lambda: self.log(f"Critical Error: {e}"))
        finally:
            self.after(0, lambda: self.train_btn.configure(state="normal", text="Start Training"))
            # Reload model if training finished
            self.after(0, self.load_model)

if __name__ == "__main__":
    app = App()
    app.mainloop()
