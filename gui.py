import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import sv_ttk
import threading
import logging
from PIL import Image, ImageTk, ImageDraw
from universal_background_remover import BackgroundRemover, setup_logging

# Setup detailed logging configuration
setup_logging()


class TextHandler(logging.Handler):
    """
    A custom logging handler that sends logs to a tkinter Text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state="disabled")
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


class UniversalBackgroundRemoverApp:
    def __init__(self, root):
        """
        Initialize the Universal Background Remover application.
        """
        self.root = root
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()

        # Initialize BackgroundRemover with default models
        obj_detector_name = "hustvl/yolos-base"
        seg_model_name = "briaai/RMBG-1.4"
        cache_dir = "cache"
        self.remover = BackgroundRemover(
            obj_detector_name, seg_model_name, cache_dir=cache_dir
        )
        self.image_paths = []
        self.bg_color = (0, 0, 0)
        self.roi_confirmation_event = threading.Event()
        self.roi = None  # Initialize ROI

        # Initialize selection attributes
        self.rect = None
        self.oval = None

    def setup_bindings(self):
        """
        Setup key bindings or other event bindings.
        """
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def setup_window(self):
        """
        Setup the main application window.
        """
        logging.debug("Initializing the main application window.")
        self.root.title("Universal Background Remover")
        self.root.geometry("1100x800")
        self.root.resizable(True, True)
        sv_ttk.set_theme("light")

    def create_widgets(self):
        """
        Create and place widgets in the application window.
        """
        self.setup_layout()
        self.setup_controls()
        self.setup_canvas()
        self.setup_progress_bar()
        self.setup_log()

    def setup_layout(self):
        """
        Setup the layout of the application window.
        """
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(
            7, weight=1
        )  # Canvas row with higher weight for expansion
        self.root.grid_rowconfigure(
            8, weight=1
        )  # Log text row with higher weight for expansion

    def setup_controls(self):
        """
        Setup the controls like buttons, dropdowns, and checkboxes.
        """
        controls_frame = ttk.Frame(self.root)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.toggle_theme_button = ttk.Button(
            controls_frame, text="Toggle Theme", command=self.toggle_theme
        )
        self.toggle_theme_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.load_images_button = ttk.Button(
            controls_frame, text="Load Images", command=self.load_images
        )
        self.load_images_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.selection_method_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=tk.StringVar(value="full"),
            values=["full", "rectangle", "oval", "freehand", "AI"],
        )
        self.selection_method_dropdown.grid(
            row=0, column=2, padx=5, pady=5, sticky="ew"
        )
        self.selection_method_dropdown.bind(
            "<<ComboboxSelected>>", self.on_selection_method_change
        )

        self.bg_color_button = ttk.Button(
            controls_frame, text="Select Background Color", command=self.select_bg_color
        )
        self.bg_color_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.start_processing_button = ttk.Button(
            controls_frame, text="Start Processing", command=self.start_processing
        )
        self.start_processing_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        self.start_processing_button.config(state=tk.DISABLED)

        self.save_image_button = ttk.Button(
            controls_frame, text="Save Image", command=self.save_image
        )
        self.save_image_button.grid(row=0, column=5, padx=5, pady=5, sticky="ew")
        self.save_image_button.config(state=tk.DISABLED)

        self.remove_inside_checkbox = ttk.Checkbutton(
            controls_frame, text="Remove Inside", variable=tk.BooleanVar(value=True)
        )
        self.remove_inside_checkbox.grid(row=0, column=6, padx=5, pady=5, sticky="ew")

        self.model_selector_frame = ttk.LabelFrame(self.root, text="Model Selection")
        self.model_selector_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.obj_model_label = ttk.Label(
            self.model_selector_frame, text="Object Detector Model"
        )
        self.obj_model_label.grid(row=0, column=0, padx=10, pady=5)
        self.obj_model_entry = ttk.Entry(self.model_selector_frame)
        self.obj_model_entry.insert(0, "hustvl/yolos-base")
        self.obj_model_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.seg_model_label = ttk.Label(
            self.model_selector_frame, text="Segmentation Model"
        )
        self.seg_model_label.grid(row=1, column=0, padx=10, pady=5)
        self.seg_model_entry = ttk.Entry(self.model_selector_frame)
        self.seg_model_entry.insert(0, "briaai/RMBG-1.4")
        self.seg_model_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.update_model_button = ttk.Button(
            self.model_selector_frame, text="Update Models", command=self.update_models
        )
        self.update_model_button.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )

    def setup_canvas(self):
        """
        Setup the image display area.
        """
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.grid(row=8, column=0, padx=10, pady=5, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar_x = ttk.Scrollbar(
            self.image_frame, orient="horizontal", command=self.canvas.xview
        )
        self.scrollbar_y = ttk.Scrollbar(
            self.image_frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas.config(
            xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set
        )
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")

    def setup_progress_bar(self):
        """
        Setup the progress bar for visual feedback during processing.
        """
        self.progress_bar = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate"
        )
        self.progress_bar.grid(row=9, column=0, padx=10, pady=5, sticky="ew")

    def setup_log(self):
        """
        Setup the logging area to display processing details.
        """
        self.log_text = tk.Text(self.root, height=10, state="disabled")
        self.log_text.grid(row=10, column=0, padx=10, pady=5, sticky="nsew")
        self.log_handler = TextHandler(self.log_text)
        logging.getLogger().addHandler(self.log_handler)

    def toggle_theme(self):
        """
        Toggle between light and dark theme.
        """
        if sv_ttk.get_theme() == "light":
            sv_ttk.set_theme("dark")
        else:
            sv_ttk.set_theme("light")

    def load_images(self):
        """
        Load images from file dialog.
        """
        file_types = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        self.image_paths = filedialog.askopenfilenames(
            title="Select Images", filetypes=file_types
        )
        if self.image_paths:
            messagebox.showinfo(
                "Images Loaded", f"{len(self.image_paths)} images loaded successfully."
            )
            self.display_image_on_canvas(self.canvas, self.image_paths[0])
            self.enable_processing_buttons()
        else:
            messagebox.showwarning(
                "No Images Selected", "No image files were selected."
            )

    def enable_processing_buttons(self):
        """
        Enable processing buttons when images are loaded.
        """
        self.start_processing_button.config(state=tk.NORMAL)
        self.save_image_button.config(state=tk.NORMAL)

    def on_selection_method_change(self, event):
        """
        Handle selection method change.
        """
        method = self.selection_method_dropdown.get()
        if method == "rectangle":
            self.open_selection_window("rectangle")
        elif method == "oval":
            self.open_selection_window("oval")
        elif method == "freehand":
            self.open_selection_window("freehand")
        elif method == "AI":
            threading.Thread(target=self.detect_and_confirm_roi, daemon=True).start()

    def open_selection_window(self, shape):
        """
        Open a window to select ROI.
        """
        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.title("Select ROI")
        self.selection_window.grab_set()
        self.selection_canvas = tk.Canvas(self.selection_window, cursor="cross")
        self.selection_canvas.pack(fill="both", expand=True)
        self.display_image_on_canvas(self.selection_canvas, self.image_paths[0])
        self.bind_selection_events(shape)

    def bind_selection_events(self, shape="rectangle"):
        """
        Bind mouse events for selecting ROI.
        """
        self.roi = None
        self.start_x = None
        self.start_y = None
        self.rect = None  # Initialize rect to None
        self.oval = None  # Initialize oval to None
        self.selection_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.selection_canvas.bind(
            "<B1-Motion>", lambda e: self.on_move_press(e, shape)
        )
        self.selection_canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        """
        Handle mouse button press event.
        """
        self.start_x = event.x
        self.start_y = event.y
        if not self.rect:
            self.rect = self.selection_canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y, outline="red"
            )

    def on_move_press(self, event, shape):
        """
        Update the shape as the mouse moves.
        """
        cur_x, cur_y = event.x, event.y
        if shape == "rectangle":
            if not self.rect:
                self.rect = self.selection_canvas.create_rectangle(
                    self.start_x, self.start_y, cur_x, cur_y, outline="red"
                )
            else:
                self.selection_canvas.coords(
                    self.rect, self.start_x, self.start_y, cur_x, cur_y
                )
        elif shape == "oval":
            if not self.oval:
                self.oval = self.selection_canvas.create_oval(
                    self.start_x, self.start_y, cur_x, cur_y, outline="red"
                )
            else:
                self.selection_canvas.coords(
                    self.oval, self.start_x, self.start_y, cur_x, cur_y
                )

    def on_button_release(self, event):
        """
        Finalize the selection on mouse release.
        """
        self.roi = (self.start_x, self.start_y, event.x, event.y)
        image_width, image_height = (
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )

        # Ensure the ROI is within image bounds
        if (
            self.roi[0] < 0
            or self.roi[1] < 0
            or self.roi[2] > image_width
            or self.roi[3] > image_height
            or self.roi[0] >= self.roi[2]
            or self.roi[1] >= self.roi[3]
        ):
            messagebox.showerror(
                "Invalid ROI", "The selected ROI is out of image bounds or not valid."
            )
            self.roi = None
        else:
            self.selection_window.destroy()
            messagebox.showinfo("ROI Selected", f"ROI selected: {self.roi}")

    def display_image_on_canvas(self, canvas, image_path):
        """
        Display an image on the given canvas.
        """
        image = Image.open(image_path)
        image.thumbnail((self.root.winfo_width() - 50, self.root.winfo_height() - 250))
        self.tk_image_canvas = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_canvas)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def reset_canvas(self):
        """
        Clear the canvas.
        """
        self.canvas.delete("all")

    def start_processing(self):
        """
        Start the image processing.
        """
        if not self.image_paths:
            messagebox.showwarning(
                "No Images", "Please load image files before starting processing."
            )
            return

        try:
            self.progress_bar["value"] = 0
            total_images = len(self.image_paths)

            threading.Thread(
                target=self.run_background_processing,
                args=(self.image_paths, self.bg_color, total_images),
            ).start()

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def run_background_processing(self, image_paths, bg_color, total_images):
        """
        Run background processing of images.
        """
        self.processed_images = []
        for index, image_path in enumerate(image_paths):
            processed_image = self.remover.process_image(image_path, self.bg_color)
            if processed_image:
                self.processed_images.append(processed_image)
                self.progress_bar["value"] = (index + 1) / total_images * 100
                self.root.update_idletasks()

        if self.processed_images:
            messagebox.showinfo(
                "Processing Complete", "Image processing completed successfully."
            )
            self.display_image_from_memory(self.processed_images[0])
        else:
            logging.warning("No images were processed.")
            messagebox.showwarning("Processing Incomplete", "No images were processed.")

    def detect_and_confirm_roi(self):
        """
        Detect ROI using AI and confirm with user.
        """
        def task():
            logging.debug("Thread for detecting ROI started")
            try:
                detected_objects = self.remover.detect_objects_with_yolo(
                    self.image_paths[0]
                )
                if detected_objects:
                    roi = detected_objects[0]["box"]
                    logging.debug(f"Detected ROI: {roi}")
                    if self.is_valid_roi(roi):
                        self.roi = roi
                        logging.debug(f"ROI detected and set: {self.roi}")
                        self.root.after(0, self.confirm_or_adjust_roi)
                    else:
                        logging.error(f"Invalid ROI detected: {roi}")
                        messagebox.showerror("ROI Error", "Invalid ROI detected.")
                else:
                    logging.debug("No ROI detected, warning shown")
                    messagebox.showwarning(
                        "Instance Segmentation Detection", "No object detected."
                    )
            except Exception as e:
                logging.error(f"Error in detect_and_confirm_roi task: {e}")

        threading.Thread(target=task, daemon=True).start()

    def is_valid_roi(self, roi):
        """
        Check if the detected ROI is valid.
        """
        if not isinstance(roi, list) or len(roi) != 4:
            return False
        x, y, x2, y2 = roi
        if x < 0 or y < 0 or x2 <= x or y2 <= y:
            return False
        return True

    def confirm_or_adjust_roi(self):
        """
        Confirm or adjust the detected ROI.
        """
        if self.roi:
            self.show_roi_on_image(self.image_paths[0], self.roi)

            # Add buttons to confirm or adjust the ROI
            self.confirm_button = ttk.Button(
                self.root, text="Confirm ROI", command=self.confirm_roi
            )
            self.confirm_button.grid(row=10, column=0, padx=10, pady=5, sticky="ew")

            self.adjust_button = ttk.Button(
                self.root, text="Adjust ROI", command=self.adjust_roi
            )
            self.adjust_button.grid(row=11, column=0, padx=10, pady=5, sticky="ew")

    def show_roi_on_image(self, image_path, roi):
        """
        Show the selected ROI on the image.
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([roi[0], roi[1], roi[2], roi[3]], outline="red", width=3)

        def display_image():
            """
            Display image in a non-blocking way.
            """
            tk_image = ImageTk.PhotoImage(image)
            display_window = tk.Toplevel(self.root)
            label = tk.Label(display_window, image=tk_image)
            label.image = tk_image  # Keep a reference to avoid garbage collection
            label.pack()

        self.root.after(0, display_image)  # Schedule the display to be non-blocking

    def confirm_roi(self):
        """
        Proceed with the detected ROI.
        """
        self.confirm_button.grid_forget()
        self.adjust_button.grid_forget()
        messagebox.showinfo("ROI Confirmed", "The detected ROI has been confirmed.")
        self.start_processing()

    def adjust_roi(self):
        """
        Allow the user to adjust the ROI manually.
        """
        self.confirm_button.grid_forget()
        self.adjust_button.grid_forget()
        self.open_selection_window("rectangle")

    def select_bg_color(self):
        """
        Select a background color using a color chooser.
        """
        color_code = colorchooser.askcolor(title="Choose background color")
        if color_code[0] is not None:
            self.bg_color = tuple(map(int, color_code[0]))

    def display_image_from_memory(self, processed_image):
        """
        Display a processed image from memory.
        """
        self.reset_canvas()
        image = Image.open(processed_image)
        width, height = image.size
        image.thumbnail((width // 3, height // 3))  # Resize image for display
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.adjust_main_window(width // 3, height // 3)

    def adjust_main_window(self, width, height):
        """
        Adjust the main window size based on the image dimensions.
        """
        self.root.geometry(f"{width + 200}x{height + 300}")

    def save_image(self):
        """
        Save the processed images.
        """
        if not self.processed_images:
            messagebox.showwarning(
                "No Processed Images", "Please process images before saving."
            )
            return

        for idx, processed_image_path in enumerate(self.processed_images):
            file_path = filedialog.asksaveasfilename(
                title=f"Save Processed Image {idx + 1}",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if file_path:
                processed_image = Image.open(processed_image_path)
                processed_image.save(file_path)
                messagebox.showinfo(
                    "Image Saved", f"Processed image saved as {file_path}"
                )

    def update_models(self):
        """
        Update the models based on user input.
        """
        obj_detector_name = self.obj_model_entry.get()
        seg_model_name = self.seg_model_entry.get()
        self.remover = BackgroundRemover(
            obj_detector_name, seg_model_name, cache_dir="cache"
        )
        messagebox.showinfo(
            "Models Updated",
            "The object detector and segmentation models have been updated.",
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalBackgroundRemoverApp(root)
    root.mainloop()
