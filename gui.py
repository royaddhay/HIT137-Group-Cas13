import tkinter as tk
from tkinter import filedialog, scrolledtext
from models import SentimentModel, ImageClassificationModel
from oop_concepts import OOPExplanation, C

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HIT137 Assignment 3 - AI GUI")
        self.geometry("800x600")

        # Dropdown menu
        self.input_type = tk.StringVar(value="Text")
        tk.Label(self, text="Select Input Type:").pack()
        tk.OptionMenu(self, self.input_type, "Text", "Image").pack()

        # Input field
        self.input_entry = tk.Entry(self, width=50)
        self.input_entry.pack()

        tk.Button(self, text="Run Model", command=self.run_model).pack()

        # Output area
        self.output = scrolledtext.ScrolledText(self, width=80, height=15)
        self.output.pack()

        # Explanations
        self.oop_box = scrolledtext.ScrolledText(self, width=80, height=10)
        self.oop_box.pack()
        self.load_explanations()

    def run_model(self):
        input_type = self.input_type.get()
        text = self.input_entry.get()
        self.output.delete("1.0", tk.END)

        if input_type == "Text":
            model = SentimentModel()
            result = model.run(text)
        else:
            file_path = filedialog.askopenfilename()
            model = ImageClassificationModel()
            result = model.run(file_path)

        self.output.insert(tk.END, str(result))

    def load_explanations(self):
        oop = OOPExplanation()
        oop.add_detail("Encapsulation", "Private variables used in OOPExplanation.")
        oop.add_detail("Polymorphism", "Both models override run() method differently.")
        oop.add_detail("Inheritance", "Class C inherits from A and B.")
        oop.add_detail("Decorators", "log_function and uppercase_output in utils.py")
        oop.add_detail("Overriding", "C overrides greet() method.")

        for k, v in oop.get_details().items():
            self.oop_box.insert(tk.END, f"{k}: {v}\n\n")
