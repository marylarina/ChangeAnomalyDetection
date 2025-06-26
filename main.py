import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import cv2
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score

from DefectVisionFormer import DefectVisionFormer
from FocalDiceLoss import FocalDiceLoss
from ModelCompilation import ModelCompilation
from torchvision import transforms


# Загрузка преобученной модели
def load_pretrained_model(model_path, model_name):
    num_classes = 1
    device = "cpu"
    mode = "binary"
    loss_function = FocalDiceLoss(mode=mode,
                                  gamma=2.0,
                                  alpha=None,
                                  beta=10.0,
                                  smooth=0.1)
    optimizer = torch.optim.AdamW
    learning_rate = 0.0001
    lr_schedule = 'ReduceLROnPlateau'
    epochs = 40
    metrics = MetricCollection(
        {
            'F1': BinaryF1Score(threshold=0.5).to(device),
        },
    )
    network = DefectVisionFormer(encoder_name='efficientnet_b4',
                                 align_type=model_name,
                                 fusion_type='concat',
                                 decoder_name='u_net',
                                 num_classes=num_classes)
    model = ModelCompilation(model=network,
                             epochs=epochs,
                             mode=mode,
                             metrics=metrics,
                             loss_function=loss_function,
                             optimizer=optimizer,
                             learning_rate=learning_rate,
                             lr_schedule=lr_schedule,
                             accelerator=device)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model


# Формирование маски с аномалией
def testModel(normal_img, anomaly_img, model):
    model.eval()
    with torch.no_grad():
        predicted_mask = model.forward(x_n=normal_img, x_a=anomaly_img).to("cpu")
    predicted_mask = predicted_mask.cpu()[0]
    predicted_mask = torch.sigmoid(predicted_mask)
    return predicted_mask


class AnomalyDetectionApp:
    def __init__(self, root):
        # Интерфейс
        self.root = root
        self.root.iconbitmap("machinery_equipment_maintenance_management_service_tools_construction_icon_261632.ico")
        self.root.title("ChangeAnomalyDetection")
        self.root.geometry("1300x500")
        self.root.configure(bg='#f0f0f0')

        self.style = {
            'font': ('Helvetica', 10),
            'bg': '#C5C4C1',
            'button': {
                'bg': '#646C84',
                'fg': 'white',
                'activebackground': '#3a5a80',
                'font': ('Helvetica', 10, 'bold'),
                'relief': 'flat',
                'borderwidth': 0,
                'padx': 15,
                'pady': 8
            },
            'label': {
                'bg': '#C5C4C1',
                'fg': '#333333',
                'font': ('Helvetica', 10)
            },
            'frame': {
                'bg': '#ffffff',
                'bd': 1,
                'relief': 'solid',
                'highlightbackground': '#F5EFDA'
            }
        }

        self.model = None
        self.input_image = None
        self.input_image_tf = None
        self.output_image = None

        self.models = {
            "multihead": {"path": "detectors/loss_model-epoch=34-val_loss=0.32-val_F1=0.78.ckpt"},
            "soft": {"path": "detectors/loss_model-epoch=38-val_loss=0.45-val_F1=0.72.ckpt"},
            "hard": {"path": "detectors/loss_model-epoch=24-val_loss=0.36-val_F1=0.77.ckpt"}
        }

        self.create_widgets()

    # Создание виджетов
    def create_widgets(self):
        control_frame = tk.Frame(self.root, **self.style['frame'])
        control_frame.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky='ns')

        tk.Label(control_frame,
                 text="ChangeAnomalyDetection",
                 font=('Helvetica', 14, 'bold'),
                 fg='#383C57',
                 bg=self.style['frame']['bg']).pack(pady=(20, 30))

        self.load_normal_btn = tk.Button(control_frame,
                                         text="Загрузить нормальное изображение",
                                         command=self.load_normal_image,
                                         **self.style['button'])
        self.load_normal_btn.pack(pady=5, fill='x', padx=10)

        self.load_anomaly_btn = tk.Button(control_frame,
                                          text="Загрузить аномальное изображение",
                                          command=self.load_anomaly_image,
                                          **self.style['button'])
        self.load_anomaly_btn.pack(pady=5, fill='x', padx=10)

        tk.Label(control_frame,
                 text="Способ согласования:",
                 bg=self.style['frame']['bg'],
                 fg=self.style['label']['fg'],
                 font=self.style['label']['font']).pack(pady=(20, 5))

        self.model_var = tk.StringVar(self.root)
        self.model_var.set("multihead")

        self.model_menu = tk.OptionMenu(control_frame,
                                        self.model_var,
                                        *self.models.keys())
        self.model_menu.config(bg='#9088A0',
                               fg='#333333',
                               font=self.style['font'],
                               relief='flat',
                               highlightthickness=0)
        self.model_menu.pack(fill='x', padx=10, pady=5)

        self.segment_btn = tk.Button(control_frame,
                                     text="Сегментировать",
                                     command=self.segment_image,
                                     **self.style['button'])
        self.segment_btn.pack(pady=(20, 5), fill='x', padx=10)

        self.clear_btn = tk.Button(control_frame,
                                   text="Сбросить маску",
                                   command=self.clear_mask,
                                   bg='#CA969A',
                                   fg='white',
                                   activebackground='#c0392b',
                                   font=self.style['button']['font'],
                                   relief='flat')
        self.clear_btn.pack(pady=5, fill='x', padx=10)

        image_frame = tk.Frame(self.root, bg=self.style['bg'])
        image_frame.grid(row=0, column=1, columnspan=2, padx=10, pady=20, sticky='nsew')

        tk.Label(image_frame,
                 text="Нормальное изображение",
                 **self.style['label']).grid(row=0, column=0, pady=(0, 10))
        tk.Label(image_frame,
                 text="Аномальное изображение",
                 **self.style['label']).grid(row=0, column=1, pady=(0, 10))

        self.normal_image_label = tk.Label(image_frame, bg='white', bd=1, relief='solid')
        self.normal_image_label.grid(row=1, column=0, padx=10, pady=5)

        self.anomaly_image_label = tk.Label(image_frame, bg='white', bd=1, relief='solid')
        self.anomaly_image_label.grid(row=1, column=1, padx=10, pady=5)

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    # Загрузка нормального изображения
    def load_normal_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # предобработка входного изображения для подачи на вход модели
            self.normal_image_torch = cv2.imread(file_path)
            self.normal_image_torch = cv2.cvtColor(self.normal_image_torch, cv2.COLOR_BGR2RGB)
            self.normal_image_torch = cv2.resize(self.normal_image_torch, (512, 512), interpolation=cv2.INTER_LINEAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.normal_image_torch = transform(self.normal_image_torch).unsqueeze(0)

            # Предобработка изображения для отображения
            self.normal_input_image = Image.open(file_path)
            self.normal_input_image = self.normal_input_image.resize((416, 416))
            self.normal_input_image_tk = ImageTk.PhotoImage(self.normal_input_image)
            self.normal_image_label.config(image=self.normal_input_image_tk)
            self.normal_image_label.image = self.normal_input_image_tk

    # Загрузка аномального изображения
    def load_anomaly_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # предобработка входного изображения для подачи на вход модели
            self.anomaly_image_torch = cv2.imread(file_path)
            self.anomaly_image_torch = cv2.cvtColor(self.anomaly_image_torch, cv2.COLOR_BGR2RGB)
            self.anomaly_image_torch = cv2.resize(self.anomaly_image_torch, (512, 512), interpolation=cv2.INTER_LINEAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.anomaly_image_torch = transform(self.anomaly_image_torch).unsqueeze(0)

            # Предобработка изображения для отображения
            self.anomaly_input_image = Image.open(file_path)
            self.anomaly_input_image = self.anomaly_input_image.resize((416, 416))
            self.original_anomaly_image = self.anomaly_input_image.copy()
            self.anomaly_input_image_tk = ImageTk.PhotoImage(self.anomaly_input_image)
            self.anomaly_image_label.config(image=self.anomaly_input_image_tk)
            self.anomaly_image_label.image = self.anomaly_input_image_tk

    # Удаление маски с изображения
    def clear_mask(self):
        if self.original_anomaly_image:
            self.anomaly_input_image = self.original_anomaly_image.copy()
            self.anomaly_input_image_tk = ImageTk.PhotoImage(self.anomaly_input_image)
            self.anomaly_image_label.config(image=self.anomaly_input_image_tk)
            self.anomaly_image_label.image = self.anomaly_input_image_tk

    # Основная функция для сегментации аномалии
    def segment_image(self):
        if self.normal_input_image is None or self.anomaly_input_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        model_name = self.model_var.get()
        model_info = self.models.get(model_name)
        if model_info is None:
            messagebox.showerror("Error", "Invalid model selected.")
            return

        model_path = model_info["path"]

        self.model = load_pretrained_model(model_path, model_name)

        mask_orig = testModel(self.normal_image_torch, self.anomaly_image_torch, self.model)

        mask_orig = mask_orig.squeeze().cpu().numpy()
        mask_orig = cv2.resize(mask_orig, (416, 416), interpolation=cv2.INTER_LINEAR)

        colored_mask = np.zeros((416, 416, 3), dtype=np.uint8)
        colored_mask[:, :, 0] = 255
        colored_mask[:, :, 1:3] = 0

        alpha_mask = (mask_orig > 0.5).astype(np.uint8) * 120
        colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=alpha_mask)

        mask_rgba = np.dstack((colored_mask, alpha_mask))
        mask_img = Image.fromarray(mask_rgba, 'RGBA')

        self.anomaly_input_image = self.original_anomaly_image.convert('RGBA')
        self.output_image = Image.alpha_composite(self.anomaly_input_image, mask_img)
        self.output_image_tk = ImageTk.PhotoImage(self.output_image)
        self.anomaly_image_label.config(image=self.output_image_tk)
        self.anomaly_image_label.image = self.output_image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()
