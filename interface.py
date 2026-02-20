import customtkinter as ctk
from tkinter import filedialog
import cv2 as cv
from PIL import Image, ImageTk
from detection import YoloTracker

# Standard Professional Theme
ctk.set_appearance_mode("dark") 
ctk.set_default_color_theme("dark-blue") # "blue" or "dark-blue" works best

class App(ctk.CTk): # Inherit from ctk.CTk for the modern window frame
    def __init__(self):
        super().__init__()
        
        self.title("Khel AI for object Segmentation")
        self.geometry("1400x800")
        
        # Initialize Tracker
        self.tracker = YoloTracker("yolo26x-seg.pt")
        self.cap = None
        self.current_tracks = []
        self.is_live = True

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Navigation)
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="YOLO v26", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=30, padx=20)

        self.btn_live = ctk.CTkButton(self.sidebar_frame, text="LIVE FEED", 
                                     fg_color="#1f538d", hover_color="#14375e",
                                     command=self.start_live)
        self.btn_live.pack(pady=10, padx=20)

        self.btn_file = ctk.CTkButton(self.sidebar_frame, text="UPLOAD VIDEO", 
                                     command=self.start_recording)
        self.btn_file.pack(pady=10, padx=20)

        self.status_indicator = ctk.CTkLabel(self.sidebar_frame, text="● Disconnected", text_color="grey")
        self.status_indicator.pack(side="bottom", pady=20)

        # 2. MAIN VIEWPORT (Video)
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#101010")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.canvas = ctk.CTkCanvas(self.main_frame, bg="black", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.update_loop()

    def start_live(self):
        self.is_live = True
        self.tracker.switch_model("yolo26x-seg.pt")
        if self.cap: self.cap.release()
        self.cap = cv.VideoCapture(0)
        self.status_indicator.configure(text="● LIVE ACTIVE", text_color="#2fa572")

    def start_recording(self):
        path = filedialog.askopenfilename()
        if path:
            self.is_live = False
            self.tracker.switch_model("yolo26n-seg.pt")
            if self.cap: self.cap.release()
            self.cap = cv.VideoCapture(path)
            filename = path.split('/')[-1]
            self.status_indicator.configure(text=f"● FILE: {filename}", text_color="#3b8ed0")

    def on_click(self, event):
        x, y = event.x, event.y
        for tid, poly in self.current_tracks:
            if cv.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                self.tracker.selected_track_id = tid
                return
        self.tracker.selected_track_id = None

    def update_loop(self):
        if self.cap and self.cap.isOpened():
            if not self.is_live: self.cap.grab() 
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                # Maintain aspect ratio in canvas
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                
                if canvas_w > 1 and canvas_h > 1:
                    scale = min(canvas_w/w, canvas_h/h)
                    frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

                display_frame, self.current_tracks = self.tracker.process_frame(frame)
                
                rgb_frame = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                self.photo = ImageTk.PhotoImage(image=img)
                
                # Center the image on the canvas
                self.canvas.delete("all")
                self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.photo, anchor="center")
            else:
                if not self.is_live: self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        self.after(10, self.update_loop) # Use .after() from ctk.CTk

if __name__ == "__main__":
    app = App()
    app.mainloop()