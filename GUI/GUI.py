print("CIAOOOOO")

import albumentations as albu
import copy
import cv2
import numpy as np
import os
import shutil
import segmentation_models_pytorch as smp
import skimage
import tkinter
import torch
from skimage.transform import resize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from tkinter import IntVar, filedialog, Frame, messagebox
from matplotlib.figure import Figure
from roipoly import RoiPoly
from collections import Counter
from skimage.morphology import thin
from torch.utils.data import Dataset as BaseDataset
from tkinter import messagebox
BACKBONE = 'timm-resnest101e'
ENCODER = BACKBONE
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class Dataset(BaseDataset):    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class filopodia():
    filename = None
    dir_name = None
    original_img = None
    # image processing params
    threshold_image = 35
    erosion = 20
    dilation = 20
    filopodia_minimum_length = 10
    filopodia_width_threshold = 40
    gamma = 1.0
    model = None
    # GUI params
    tkTop = tkinter.Tk()
    already_view = 0
    histogram = plt.Figure()
    hide_ImProc_steps = tkinter.IntVar()
    # ROI selection params
    roi_bbox_x = np.zeros((1, 4))
    roi_bbox_y = np.zeros((1, 4))
    ROI_n_clicks = 0
    already_opened_an_image = False

    def __init__(self):
        self.build_root_layout()
        self.build_menu_bar()
        self.build_left_panel()  # build parameters controls (left panel)
        self.build_right_panel()  # build analysis panel (right panel)
        self.tkTop.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.load_DL_model("best_model.pth")

    def adjust_gamma(self, image, gamma=1.0):
        inverse_gamma = 1.0 / gamma
        table = np.array([((l / 255.0) ** inverse_gamma) *
                         255 for l in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def browse_file(self):
        self.dir_name = None
        temp = self.filename
        self.filename = filedialog.askopenfilename()
        if self.filename != temp:
            self.hide_me(self.tkButtonCompute)

        self.original_img = cv2.imread(self.filename)
        self.show_single_image(self.original_img)

        # init variable for roi selection
        self.img_for_roi = np.zeros((1, np.shape(self.original_img)[2]))
        for i in range(0, np.shape(self.original_img)[2]):
            if cv2.countNonZero(self.original_img[:, :, i]) != 0:
                a = cv2.blur(self.original_img[:, :, i], (15, 15))
                self.img_for_roi[0, i] = cv2.countNonZero(a)
        self.img_for_roi = np.argsort(self.img_for_roi)
 
    def browse_dir(self):
        self.dir_name = filedialog.askdirectory()
        self.hide_me(self.tkButtonCompute)

    def show_single_image(self, img):
        if self.already_opened_an_image:
            self.canvas.get_tk_widget().destroy()
        figure = Figure()
        figure.add_subplot(111).imshow(img)
        self.canvas = FigureCanvasTkAgg(figure, self.central_frame)
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        self.already_opened_an_image = True

    def FiloAnalyzer_full(self, img, threshold_method, erosion, dilation, filopodia_area_threshold, gamma):

        # PREPROCESSING
        if len(img.shape) > 2:
            body = img[:,:,0]
        else:
            body = img
        body[body < 0.3 * np.mean((body))] = 0
        body = self.adjust_gamma(body, gamma=gamma)

        # SEGMENTATION
        if threshold_method == "automatic":
            test = copy.copy(body)
            body = cv2.adaptiveThreshold(test, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 655, 5)
            body = cv2.morphologyEx(body, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            body = ndi.binary_fill_holes(body).astype(np.uint8)
            x, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            body = np.zeros_like(body)
            for i in x:
                mom = cv2.moments(i)
                area = mom['m00']
                if area > 10000 and area < 2000000:
                    cv2.drawContours(body, [i], -1, (255, 144, 255), -1)
        elif threshold_method == "triangle":
            test = copy.copy(body)
            hist = np.histogram(body.ravel(), bins=256, range=(0.0, 255))
            ret, body = cv2.threshold(body, 42, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            s = hist[0][int(ret):int(ret) + 12]
            s[np.where(s == 0)] = np.max(s)
            t = np.argmin(s) + int(ret)
            ret, body = cv2.threshold(test, t, 255, cv2.THRESH_BINARY)
            body = ndi.binary_fill_holes(body).astype(np.uint8)
        else:
            ret, body = cv2.threshold(body, threshold_method, 255, cv2.THRESH_BINARY)
        
        thresholded_image = copy.deepcopy(body)
        
        nucleus = cv2.morphologyEx(body, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)))
        nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)))
        body[nucleus != 0] = 0

        opened_image = copy.deepcopy(body)

        # EXTRACT ONLY BIGGEST AND ELONGATED CONTOURS
        result = np.zeros_like(body)
        x, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in x:
            moments = cv2.moments(i)
            area = moments['m00']
            if area > filopodia_area_threshold and area < 200000:
                if moments['m00'] != 0.0:
                    ellip = cv2.fitEllipse(i)
                    (_, axes, _) = ellip
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    if np.sqrt(1 - (minor_axis * minor_axis) / (major_axis * major_axis)) > 0.7:
                        cv2.drawContours(result, [i], -1, 1, -1)

        return thresholded_image, opened_image, result
    
    def DL_predict(self, img):
        img = resize(img, (320, 320), preserve_range=True, mode='reflect')
        if os.path.isdir("tmp"):
            shutil.rmtree("tmp")
        os.mkdir("tmp")
        cv2.imwrite(img=img, filename="./tmp/tmp.png")

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        tmp_dataset = Dataset(
            "tmp", 
            "tmp", 
            augmentation=None, 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=["car"],
        )
        for i in range(len(tmp_dataset)):
            n = i
            
            image, gt_mask = tmp_dataset[n]
            
            gt_mask = gt_mask.squeeze().astype(bool)
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = self.DL_model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy() > 0.999)

            return pr_mask

    def start(self):
        self.gather_image_processing_params()

        if self.dir_name != None: # if batch mode
            self.batch_compute_imgs()
            return
        plt.close("all")
        self.canvas.get_tk_widget().destroy()
        self.hide_me(self.tkButtonCompute)
        img = self.original_img

        if self.deep.get():
            if self.DL_model is None:
                self.load_DL_model()
            result = self.DL_predict(img).astype(np.uint8)
            # shape (320, 320) dtype uint8 min 0 max 1

        else:
            thresholded, opened, result = self.FiloAnalyzer_full(img, self.threshold_type, self.erosion, self.dilation, 10, self.gamma)
            # shape (1093, 1093) dtype uint8 min 0 max 1
        self.clean_result = copy.deepcopy(result)
        annotations = self.annotations(result)
        fused_result = self.detect_fused(result)

        # adjust shapes and colors, compose final result
        fused_result = np.dstack([fused_result, fused_result, fused_result])
        fused_result[fused_result[:,:,0] == 1] = [100, 100, 250] # light blue -> single
        fused_result[fused_result[:,:,0] == 2] = [210, 210, 50] # yellow -> fused
        result = annotations * np.dstack([(annotations.mean(-1) != 0), (annotations.mean(-1) != 0), (annotations.mean(-1) != 0)]) \
                + fused_result * np.dstack([(annotations.mean(-1) == 0), (annotations.mean(-1) == 0), (annotations.mean(-1) == 0)])

        figure = Figure()
        print(self.hide_ImProc_steps.get(), self.deep.get() == 1)
        if (self.hide_ImProc_steps.get() == 1) or (self.deep.get() == 1):
            figure.add_subplot(111).imshow(result)
            self.canvas = FigureCanvasTkAgg(figure, self.central_frame)
        else:
            figure.add_subplot(221).imshow(img)
            figure.add_subplot(222).imshow(thresholded)
            figure.add_subplot(223).imshow(opened)
            figure.add_subplot(224).imshow(result)
        self.canvas = FigureCanvasTkAgg(figure, self.central_frame)
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        self.tkButtonCompute.pack()

    def batch_compute_imgs(self):
        dst_dir = filedialog.askdirectory(title='Destination for batch mode results')
        for filename in os.listdir(self.dir_name):
            img = cv2.imread(os.path.join(self.dir_name, filename))
            _, _, result = self.FiloAnalyzer_full(img, self.threshold_type, self.erosion, self.dilation, 10, self.gamma)
            result = (result > 0).astype(np.uint8) * 255
            plt.imshow(result)
            plt.show()
            print("WRITING FROM", os.path.join(self.dir_name, filename), "TO", os.path.join(dst_dir, filename))
            cv2.imwrite(os.path.join(dst_dir, filename), result)

    def annotations(self, img):
        contours, _ = cv2.findContours(img.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if img.max() <= 2:
            img = img.astype(np.uint8) * 255
        if len(img.shape) <= 2:
            img = np.dstack([img, img, img])
        result = np.zeros_like(img)
        skipped = 0
        for i, contour in enumerate(contours):
            # Find the centroid of the contour
            M = cv2.moments(contour)
            if M['m00'] == 0:
                skipped += 1
                continue
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            # Add the index number near the centroid
            cv2.putText(result, str(i + 1 - skipped), (centroid_x + 10, centroid_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170,3,3), 3)
            cv2.putText(result, str(i + 1 - skipped), (centroid_x + 10, centroid_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,170,3), 2)
        return result

    def detect_fused(self, img):

        def find_branchpoints(skeleton):
            #skeleton = skeleton.astype(int)
            return find_endpoints(skeleton) - 2

        def find_endpoints(img):
            # Find row and column locations that are non-zero
            (rows,cols) = np.nonzero(img)

            # Initialize empty list of co-ordinates
            skel_coords = []

            # For each non-zero pixel...
            for (r,c) in zip(rows,cols):

                # Extract an 8-connected neighbourhood
                (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

                # Cast to int to index into image
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                # Convert into a single 1D array and check for non-zero locations
                pix_neighbourhood = img[row_neigh,col_neigh].ravel() != 0

                # If the number of non-zero locations equals 2, add this to 
                # our list of co-ordinates
                if np.sum(pix_neighbourhood) == 2:
                    skel_coords.append((r,c))

            return len(skel_coords)

        img_thinned = thin(img) # or skeletonize, small difference
        img_thinned[0,:] = 0
        img_thin_labeled = skimage.measure.label(img_thinned.astype(np.uint8), connectivity=2)
        img_labeled = skimage.measure.label(img.astype(np.uint8), connectivity=2)
        stats_bbox = skimage.measure.regionprops(img_thin_labeled.astype(np.uint8))
        # results to fill
        fused_image = np.zeros_like(img)
        singles_image = np.zeros_like(img)
        finish = np.zeros_like(img)

        for i in range(0, len(stats_bbox)):

            bbox = stats_bbox[i].bbox
            # take thinned branch region
            bbox_region = img_thin_labeled[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # take its largest connected component in case multiple accidentally are in that bounding box
            value_counts = Counter(bbox_region.flatten()).most_common()
            most_frequent_value = value_counts[1][0] if len(value_counts) > 1 else value_counts[0][0]
            bbox_region = (bbox_region == most_frequent_value) * 1

            # if into that bounding box #branchpoints > 1 AND #endpoints >= 4, it is a FUSED filopodia
            bbox_region_padded = np.pad(bbox_region, pad_width=4, mode='constant', constant_values=0)
            n_endpoints = find_endpoints(bbox_region_padded)
            n_branchpoints = find_branchpoints(bbox_region_padded)
            is_fused = n_branchpoints > 1 and n_endpoints >= 4

            # mark FUSED and SINGLE regions with 2 different values
            if is_fused:
                fused_image += (img_labeled == (i + 1))
            else:
                singles_image += (img_labeled == (i + 1))
            finish = singles_image + fused_image * 2

        return finish

    def gather_image_processing_params(self):
        self.threshold_type = "triangle" if self.checked_triangle.get() else \
                        "automatic" if self.checked_automatic.get() else \
                        self.tkScale_threshold.get()
        self.threshold_image = self.tkScale_threshold.get()
        self.erosion = self.tkScale_erosion.get()
        self.dilation = self.tkScale_dilation.get()
        self.filopodia_width_threshold = self.tkScale_filopodia.get()
        self.gamma = self.tkScale_gamma.get()

    def build_root_layout(self):
        self.tkTop.title("Filopodia Segmentation")
        self.left_frame = Frame(self.tkTop,
                                background="light blue",
                                borderwidth=5,
                                width=200)
        self.central_frame = Frame(self.tkTop,
                                   background="#FFF8DC",
                                   borderwidth=0,
                                   width=1000)
        self.right_frame = Frame(self.tkTop,
                                 background="light blue",
                                 borderwidth=5,
                                 width=200)
        self.left_frame.pack(side='left', expand=False, fill='y')
        self.central_frame.pack(side='left', expand=True, fill='both')
        self.right_frame.pack(side='right', expand=False, fill='y')
        #self.tkTop.geometry('2000x1000')
        self.tkTop.state("zoomed")

    def build_right_panel(self):
        self.labelr1 = tkinter.Label(self.right_frame, text="Result", bg='#005000', fg='#ffffff')
        self.labelr2 = tkinter.Label(self.right_frame, text="# filopodia (demerged)", bg='#91ee91', fg='#000000')
        self.labelr22 = tkinter.Label(self.right_frame, text="", bg='#005000', fg='#ffffff')
        self.labelr3 = tkinter.Label(self.right_frame, text="Average filopodia length",  bg='#91ee91', fg='#000000')
        self.labelr33 = tkinter.Label(self.right_frame, text="", bg='#005000', fg='#ffffff')
        self.labelr33.config(font=("Courier", 20))
        self.labelr3.config(font=("Courier", 20))
        self.labelr22.config(font=("Courier", 20))
        self.labelr2.config(font=("Courier", 20))
        self.labelr1.config(font=("Courier", 20))

    def build_left_panel(self):
        self.tkButtonStart = tkinter.Button(self.left_frame, text="Start", command=lambda: self.start())
        self.tkScale_threshold = tkinter.Scale(self.left_frame, from_=1, to=255, orient=tkinter.HORIZONTAL)
        self.tkScale_erosion = tkinter.Scale(self.left_frame, from_=1, to=50, orient=tkinter.HORIZONTAL)
        self.tkScale_dilation = tkinter.Scale(self.left_frame, from_=1, to=40, orient=tkinter.HORIZONTAL)
        self.tkScale_filopodia = tkinter.Scale(self.left_frame, from_=1, to=100, orient=tkinter.HORIZONTAL)
        self.tkScale_gamma = tkinter.Scale(self.left_frame, from_=0.1, to=5.0, orient=tkinter.HORIZONTAL, resolution=0.1)
        self.tkScale_threshold.set(self.threshold_image)
        self.tkScale_erosion.set(self.erosion)
        self.tkScale_dilation.set(self.dilation)
        self.tkScale_filopodia.set(40)
        self.checked_automatic = IntVar()
        self.checked_triangle = IntVar()
        self.deep = IntVar()
        self.tkScale_gamma.set(self.gamma)
        self.auto_thresh_checkbox = tkinter.Checkbutton(self.left_frame, text="Automatic threshold (Adaptive)", variable=self.checked_automatic)
        self.auto_thresh_checkbox.pack()
        self.triang_thresh_checkbox = tkinter.Checkbutton(self.left_frame, text="Automatic threshold (Triangle)", variable=self.checked_triangle)
        self.triang_thresh_checkbox.pack()
        self.deeplearning = tkinter.Checkbutton(self.left_frame, text="Deep learning prediction", variable=self.deep, state='disabled')
        self.deeplearning.pack()
        self.label = tkinter.Label(self.left_frame, text="threshold for image edge extraction")
        self.label.pack()
        self.tkScale_threshold.pack()
        self.label2 = tkinter.Label(self.left_frame, text="erosion factor")
        self.label2.pack()
        self.tkScale_erosion.pack()
        self.label3 = tkinter.Label(self.left_frame, text="dilation factor")
        self.label3.pack()
        self.tkScale_dilation.pack()
        self.label4 = tkinter.Label(self.left_frame, text="filopodia width factor")
        self.label4.pack()
        self.tkScale_filopodia.pack()
        self.label5 = tkinter.Label(self.left_frame, text="gamma for highlight filopodia")
        self.label5.pack()
        self.tkScale_gamma.pack()
        self.onlyresult = tkinter.Checkbutton(self.left_frame, text="Hide intermediate steps", variable=self.hide_ImProc_steps)
        self.onlyresult.pack()
        self.tkButtonStart.pack()
        self.tkButtonROI = tkinter.Button(self.left_frame, text="Select ROI", command=lambda: self.getROI())
        self.tkButtonROI.pack()
        self.tkButtonCompute = tkinter.Button(self.left_frame, text="Compute", command=lambda: self.extract_statistics())

    def build_menu_bar(self):
        self.menuBar = tkinter.Menu(self.tkTop)
        self.fileMenu = tkinter.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Open image", command=lambda: self.browse_file())
        self.fileMenu.add_command(label="Open directory", command=lambda: self.browse_dir())  # TODO
        self.fileMenu.add_command(label="Save", command=lambda: self.save())

        self.DLMenu = tkinter.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="DL Model", menu=self.DLMenu)
        self.DLMenu.add_command(label="Load model", command=lambda: self.load_DL_model())
        self.DLMenu.add_command(label="Fine tune", command=lambda: self.finetune_DL_model())

    def load_DL_model(self, path=None):
        if path is not None:
            model_filename = path
        else:
            model_filename = filedialog.askopenfilename()
        print(model_filename)
        try:
            self.DL_model = torch.load(model_filename, map_location=torch.device(DEVICE))
            messagebox.showinfo("OK", "Model loaded")
            self.deeplearning['state'] = tkinter.NORMAL
        except Exception as e:
            messagebox.showinfo("Error", "Model could not be loaded: " + str(e))

    def finetune_DL_model(self):
        if self.model is None:
            messagebox.showerror('Deep Learing model missing', 'Deep learning model missing: load a model with [DL Model > Load model] before fine tuning')
        imgs_dir = filedialog.askdirectory(title='Images directory')
        masks_dir = filedialog.askdirectory(title='Masks directory')
        # TODO
        
    def hide_me(self, widget):
        """Hides the specified widget"""
        widget.pack_forget()

    def on_closing(self):
        """What happens on click on top-right X"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.tkTop.destroy()

    def startloop(self):
        """Starts tkinter main loop"""
        self.tkTop.config(menu=self.menuBar)
        self.tkTop.mainloop()

    def save(self):
        f = filedialog.asksaveasfilename()
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        self.figure_result.savefig(f + '_complete')
        self.histogram.savefig(f + '_histogram')
        self.figure.savefig(f + '_final')
        text_file = open(f + 'writer.txt', "w")
        text_file_single = open(f + 'single_length.txt', "w")
        for i in self.txtlist:
            text_file.write(i + "\n")
        text_file.close()
        for i in self.filopodia_single:
            text_file_single.write(str(i) + "\n")
        text_file_single.close()

    def extract_statistics(self):

        figg = plt.figure() # istogramma di lunghezza o angolo orientamento filopodi? TODO fa restare il programma vivo, usare Figure?
        n_bins = 20
        thinned_image = (thin(self.clean_result) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(thinned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filopodia_lenghts = []
        # Iterate through the contours
        for contour in contours:
            # Create a mask for the current contour
            mask = np.zeros_like(thinned_image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            # Count the white pixels within the contour using the mask
            white_pixel_count = np.sum(mask == 255)
            filopodia_lenghts.append(white_pixel_count)

        plt.hist(filopodia_lenghts, bins=n_bins, edgecolor='black', color='#91ee91')
        plt.xlabel('Length in pixels')
        plt.ylabel('Frequency')
        plt.title('Filopodia length distribution')
        self.histogram = figg
        try:
            self.canvas2.get_tk_widget().destroy()
        except:
            pass
        self.canvas2 = FigureCanvasTkAgg(self.histogram, self.right_frame)
        self.labelr1.pack()
        self.labelr2.pack()
        self.labelr22['text'] = str(self.num_filopodia_demerged(self.clean_result)) # n filopodi demerged
        self.labelr22.pack()
        self.labelr3.pack()
        self.labelr33['text'] = str(np.round(np.mean(filopodia_lenghts), 2)) + " px" # avg length
        self.labelr33.pack()
        self.canvas2.get_tk_widget().pack(side=tkinter.BOTTOM, expand=False)

    def getROI(self):
        img = self.original_img.mean(axis=2)
        plt.imshow(img)
        my_roi = RoiPoly(color='r') # draw new ROI in red color
        plt.show()
        mask = my_roi.get_mask(img) * 255
        self.original_img = np.min([mask, img], axis=0).astype(np.uint8)
        self.original_img = np.dstack([self.original_img, self.original_img, self.original_img])
        self.show_single_image(self.original_img)

    def num_filopodia_demerged(self, mask):
        thinned = thin(mask)
        img_thin_labeled = skimage.measure.label(thinned.astype(np.uint8), connectivity=2)
        stats_bbox = skimage.measure.regionprops(img_thin_labeled.astype(np.uint8))
        filopodia_count = 0
        for i in range(0, len(stats_bbox)):
            bbox = stats_bbox[i].bbox
            bbox_region = img_thin_labeled[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            value_counts = Counter(bbox_region.flatten()).most_common()
            most_frequent_value = value_counts[1][0] if len(value_counts) > 1 else value_counts[0][0]
            bbox_region = (bbox_region == most_frequent_value) * 1

            # if into that bounding box #branchpoints > 1 AND #endpoints >= 4, it is a FUSED filopodia
            bbox_region_padded = np.pad(bbox_region, pad_width=4, mode='constant', constant_values=0)
            n_endpoints = self.find_endpoints(bbox_region_padded)
            
            filopodia_count += n_endpoints - 1
        return filopodia_count

    def find_endpoints(self, img):
        # Find row and column locations that are non-zero
        (rows,cols) = np.nonzero(img)

        # Initialize empty list of co-ordinates
        skel_coords = []

        # For each non-zero pixel...
        for (r,c) in zip(rows,cols):

            # Extract an 8-connected neighbourhood
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

            # Cast to int to index into image
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = img[row_neigh,col_neigh].ravel() != 0

            # If the number of non-zero locations equals 2, add this to 
            # our list of co-ordinates
            if np.sum(pix_neighbourhood) == 2:
                skel_coords.append((r,c))

        return len(skel_coords)


filop = filopodia()
filop.startloop()