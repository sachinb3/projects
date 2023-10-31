import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

coefficients_left = 0
image = None
        

def choose_image():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512,512))
        print(image.shape)


def perform_dct_idct():
    if image is None:
        messagebox.showerror("Error", "Please choose an image.")
        return
    try:

        mask_values = []
        for i in range(8):
            row_values = []
            for j in range(8):
                value = int(mask_entries[i][j].get())
                row_values.append(value)
            mask_values.append(row_values)
        
        mask = np.array(mask_values, dtype=np.uint8)
            
        block_size = 8
        
        num_blocks_height = image.shape[0] // block_size
        num_blocks_width = image.shape[1] // block_size
        
        reconstructed_image = np.zeros_like(image)
        
        for i in range(num_blocks_width):
            for j in range(num_blocks_height):
                block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                dct_block = dct(dct(block , axis=0, norm='ortho'), axis=1, norm='ortho') 
               #print(np.abs(dct_block))
                dct_block = dct_block * mask
                
                
                idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                
                reconstructed_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = idct_block
        
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Reconstructed Image')
        plt.show()
        
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter integer values.")

root = tk.Tk()
#root.state("zoomed")
root.title("DCT & IDCT GUI")

choose_image_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_image_button.config(fg="black", bg="grey")

choose_image_button.grid(row=8, columnspan=1, padx=1)

def update_mask_entries(new_mask):
    for i in range(8):
        for j in range(8):
            mask_entries[i][j].delete(0, tk.END)
            mask_entries[i][j].insert(0, str(new_mask[i][j]))

coefficient_loss_var = tk.StringVar()
coefficient_loss_var.set("Coefficient Loss: 0")

coefficient_loss_label = tk.Label(root, textvariable=coefficient_loss_var)
coefficient_loss_label.grid(row=9, column=0, columnspan=8)


def perform_sum(preset):
    coefficients_left = np.sum(preset)
    #print(coefficients_left)
    update_mask_entries(preset)
    coefficient_loss_var.set("Coefficient Percent Loss: {}".format(100*(64 - coefficients_left)/64)+"%")

mask_entries = []
for i in range(8):
    row_entries = []
    for j in range(8):
        entry = tk.Entry(root, width=10)
        entry.grid(row=i, column=j)
        entry.insert(0, "0")  # Default value
        row_entries.append(entry)
    mask_entries.append(row_entries)


def zero_matrix():
    preset_mask = np.zeros((8, 8), dtype=np.uint8)
    perform_sum(preset_mask)

def apply_preset_one():
    preset_mask = np.zeros((8, 8), dtype=np.uint8)
    for i in range(4):
        preset_mask[i, :4] = 1
    perform_sum(preset_mask)

def apply_preset_two():
    preset_mask = np.ones((8, 8), dtype=np.uint8)
    for i in range(4):
        preset_mask[i, :4] = 0
    perform_sum(preset_mask)

def apply_preset_three():
    preset_mask = np.identity(8, dtype=np.uint8)
    perform_sum(preset_mask)

def apply_preset_four():
    preset_mask = np.ones((8,8), dtype=np.uint8)
    perform_sum(preset_mask)

def apply_preset_five():
    preset_mask = np.zeros((8, 8), dtype=np.uint8)
    for i in range(4):
        preset_mask[i, :6] = 1
    perform_sum(preset_mask)

def apply_preset_six():
    preset_mask = np.ones((8, 8), dtype=np.uint8)
    for i in range(4):
        preset_mask[i, :2] = 0
    perform_sum(preset_mask)

def apply_preset_seven():
    preset_mask = np.ones((8, 8), dtype=np.uint8)
    for i in range(4):
        preset_mask[i, :3] = 1    
    perform_sum(preset_mask)

def apply_preset_eight():
    preset_mask =  np.random.randint(2,size = (8,8))
    perform_sum(preset_mask)

perform_button = tk.Button(root, text="Perform DCT & IDCT", command=perform_dct_idct)
perform_button.config(fg="black", bg="grey")
perform_button.grid(row=8, columnspan=8)

preset_one_button = tk.Button(root, text="Preset One", command=apply_preset_one)
preset_one_button.grid(row=0, column=8)

preset_two_button = tk.Button(root, text="Preset Two", command=apply_preset_two)
preset_two_button.grid(row=1, column=8)

preset_three_button = tk.Button(root, text="Preset Three", command=apply_preset_three)
preset_three_button.grid(row=2, column=8)

preset_four_button = tk.Button(root, text="Preset Four", command=apply_preset_four)
preset_four_button.grid(row=3, column=8)

preset_five_button = tk.Button(root, text="Preset Five", command=apply_preset_five)
preset_five_button.grid(row=4, column=8)

preset_six_button = tk.Button(root, text="Preset Six", command=apply_preset_six)
preset_six_button.grid(row=5, column=8)

preset_seven_button = tk.Button(root, text="Ones Matrix", command=apply_preset_seven)
preset_seven_button.grid(row=6, column=8)

preset_eight_button = tk.Button(root, text="Randomize", command=apply_preset_eight)
perform_button.config(fg="black", bg="grey")
preset_eight_button.grid(row=8, column=8)

zero_button = tk.Button(root, text="Zero Matrix", command=zero_matrix)
zero_button.grid(row=7, column=8)

root.mainloop()
