import os
import tkinter.messagebox
import math
import time
from collections import Counter
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, StringVar, Label, messagebox, simpledialog
from tkinter.ttk import Button
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# 创建 Tk 窗口
root = tk.Tk()
root.title("图片查看器")
root.geometry("1600x900")

# 初始化变量
image_names = []  # 图片文件名列表
image_index = 0   # 当前浏览的图片的索引
image_label = None  # 显示图片的标签
folder_path = ""   # 当前浏览的文件夹路径
new_image = None


# 定义打开文件夹的函数
def open_folder():
    global image_names, image_index, image_label, folder_path
    # 选择文件夹
    folder_path = filedialog.askdirectory()
    # 读取文件夹中的所有文件名
    image_names = [name for name in os.listdir(folder_path)
                   if name.lower().endswith(".jpg") or
                   name.lower().endswith(".png") or
                   name.lower().endswith(".bmp") or
                   name.lower().endswith(".webp")
                   ]
    # 将当前图片索引设为 0
    image_index = 0
    # 显示第一张图片
    show_image()


# 定义显示图片的函数
def show_image():
    global image_label, image_index, image_names
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    # 打开图片
    image = Image.open(image_path)
    image.thumbnail((1024, 768))
    # 将图片转换为 PhotoImage 对象
    image = ImageTk.PhotoImage(image)
    # 如果没有图片标签，则创建一个
    if not image_label:
        image_label = tk.Label(root, image=image)
        image_label.pack()
    # 否则直接更新图片
    else:
        image_label.config(image=image)
    # 将图片对象保存在标签上，防止图片被垃圾回收
    image_label.image = image
    # 更新当前图片文件名标签
    label_text.set(image_names[image_index])


# 定义显示上一张图片的函数
def prev_image(event=None):
    global image_index, image_names
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 将当前图片索引减 1
    image_index = (image_index - 1) % len(image_names)
    # 显示图片
    show_image()


# 定义显示下一张图片的函数
def next_image(event=None):
    global image_index, image_names
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 将当前图片索引加 1
    image_index = (image_index + 1) % len(image_names)
    # 显示图片
    show_image()


# 定义图像复原的函数
def image_restoration(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    # 读取图片
    pic = cv2.imread(image_path)
    # 将图片转换为浮点类型
    f = pic.astype(float)
    # 进行傅立叶变换
    g = fft2(f)
    # 将图像移至中间
    g = fftshift(g)
    if g.shape.__len__() != 2:
        messagebox.showerror("错误", "图像复原仅能应用于灰度图")
        return
    m, n = g.shape[:2]
    # 参数值设定
    K = 0.0025
    k = 80
    # 求距离的中心点
    start = time.time()
    u0 = round(m/2)
    v0 = round(n/2)
    H = np.zeros((m, n))
    # 逆滤波还原
    for i in range(m):
        for j in range(n):
            d = (i-u0)**2 + (j-v0)**2
            H[i, j] = np.exp(-k*(d**(5/6)))
            if H[i, j] < 0.0001:
                H[i, j] = 0.78
    F1 = g / H
    F1 = ifftshift(F1)
    F1 = ifft2(F1)
    F1 = np.real(F1).astype(np.uint8)
    # 维纳滤波还原
    d = np.abs(H)
    d = d**2
    F2 = g / H * (d / (d + K))
    F2 = ifftshift(F2)
    F2 = ifft2(F2)
    F2 = np.real(F2).astype(np.uint8)
    choice = tkinter.messagebox.askquestion("请选择图像复原方式", "使用逆滤波还原请点击“是”，使用维纳滤波还原请点击“否”",
                                            type='yesnocancel')
    end = time.time()
    print("图像复原用时：", end - start)
    global new_image
    if choice == "yes":
        # 读取复原图
        pil_image = Image.fromarray(F1)
        new_image_path = os.path.join(folder_path, image_names[image_index] + "-逆滤波还原图像.jpg")
        # 将 PIL 图像转换为 PhotoImage 图像
        new_image = ImageTk.PhotoImage(pil_image)

        # 创建新窗口
        new_window = tk.Toplevel(root)
        new_window.title("逆滤波复原")
        new_window.geometry("1600x900")

        # 创建 Label 组件来显示图像
        new_image_label = tk.Label(new_window, image=new_image)
        new_image_label.pack()

        # 创建保存按钮
        def save_image():
            pil_image.save(new_image_path)
            messagebox.showinfo("提示", "已成功保存")

        save_button = tk.Button(new_window, text="保存", command=save_image)
        save_button.pack()

        # 创建关闭按钮
        close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
        close_button.pack()

    elif choice == "no":
        # 读取复原图
        pil_image = Image.fromarray(F2)
        new_image_path = os.path.join(folder_path, image_names[image_index] + "-维纳滤波还原图像.jpg")
        # 将 PIL 图像转换为 PhotoImage 图像
        new_image = ImageTk.PhotoImage(pil_image)

        # 创建新窗口
        new_window = tk.Toplevel(root)
        new_window.title("维纳滤波复原")
        new_window.geometry("1600x900")
        # 创建 Label 组件来显示图像
        new_image_label = tk.Label(new_window, image=new_image)
        new_image_label.pack()

        # 创建保存按钮
        def save_image():
            pil_image.save(new_image_path)
            messagebox.showinfo("提示", "已成功保存")
        save_button = tk.Button(new_window, text="保存", command=save_image)
        save_button.place(relx=0.3, rely=0.92, anchor="center")
        save_button.pack()

        # 创建关闭按钮
        close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
        close_button.pack()

    else:
        return


# 定义傅里叶变换
def fourier_transform(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    # 读取图片
    pic = cv2.imread(image_path)
    start = time.time()
    # 将图像转换为灰度图像
    if len(pic.shape) == 3:
        gray_image = np.mean(pic, axis=2)
    else:
        gray_image = pic
    # 计算傅里叶变换
    fft_image = np.fft.fft2(gray_image)
    # 将复数转换为实数
    fft_image = np.abs(fft_image)
    # 调整动态范围
    fft_image = np.log(fft_image + 1)
    fft_image = np.uint8(fft_image * 255)
    # 读取变换图
    pil_image = Image.fromarray(fft_image)
    end = time.time()
    print("傅里叶变换用时：", end - start)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-傅里叶变换频谱.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("傅里叶变换频谱")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义高斯低通滤波
def lowpass_filter(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    flag = 0
    # 读入图像
    pic = cv2.imread(image_path)
    if len(pic.shape) == 3:
        flag = 1
    start = time.time()
    # 获取图像大小
    M, N = pic.shape[:2]
    # 参数值设定
    D0 = 20
    # 中心点四舍五入，求距离
    u0 = round(M / 2)
    v0 = round(N / 2)
    # 高斯低通滤波
    if flag == 0:
        # 创建滤波器
        y = np.zeros((M, N), dtype=complex)
        # 将输入图像进行傅立叶变换
        f = np.fft.fft2(pic)
        # 移至中间
        f = np.fft.fftshift(f)
        for i in range(M):
            for j in range(N):
                d = np.sqrt((i - u0) ** 2 + (j - v0) ** 2)
                y[i, j] = np.exp(-(d ** 2) / (2 * (D0 ** 2)))
    # 使用滤波器滤波变换后的图像
        y = y * f
    # 反变换
        y = np.fft.ifftshift(y)
        y = np.fft.ifft2(y)
    # 转换为实数部分，并转换为 uint8 类型
        y = np.abs(y.real)
        y = np.real(y).astype(np.uint8)
    # 读取滤波图
    else:
        t = pic
        # 创建滤波器
        fi = np.zeros((M, N), dtype=complex)
        for i in range(M):
            for j in range(N):
                d = np.sqrt((i - u0) ** 2 + (j - v0) ** 2)
                fi[i, j] = np.exp(-(d ** 2) / (2 * (D0 ** 2)))
        for k in range(0, 3):
            # 将输入图像进行傅立叶变换
            f = np.fft.fft2(pic[:, :, k])
            # 移至中间
            f = np.fft.fftshift(f)
            # 使用滤波器滤波变换后的图像
            y = fi * f
            # 反变换
            y = np.fft.ifftshift(y)
            y = np.fft.ifft2(y)
            # 转换为实数部分，并转换为 uint8 类型
            y = np.abs(y.real)
            y = np.real(y).astype(np.uint8)
            # 给单通道赋值
            t[:, :, k] = y
        # 其实是通道合并
        y = t
        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])

    end = time.time()
    print("高斯低通滤波用时：", end - start)
    pil_image = Image.fromarray(y.astype(np.uint8))
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-高斯低通滤波结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("高斯低通滤波")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义高斯高通滤波
def highpass_filter(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    flag = 0
    # 读入图像
    pic = cv2.imread(image_path)
    if len(pic.shape) == 3:
        flag = 1
    start = time.time()
    # 获取图像大小
    M, N = pic.shape[:2]
    # 参数值设定
    D0 = 20
    # 中心点四舍五入，求距离
    u0 = round(M / 2)
    v0 = round(N / 2)
    # 高斯低通滤波
    if flag == 0:
        # 创建滤波器
        y = np.zeros((M, N), dtype=complex)
        # 将输入图像进行傅立叶变换
        f = np.fft.fft2(pic)
        # 移至中间
        f = np.fft.fftshift(f)
        for i in range(M):
            for j in range(N):
                d = np.sqrt((i - u0) ** 2 + (j - v0) ** 2)
                y[i, j] = 1 - np.exp(-(d ** 2) / (2 * (D0 ** 2)))
    # 使用滤波器滤波变换后的图像
        y = y * f
    # 反变换
        y = np.fft.ifftshift(y)
        y = np.fft.ifft2(y)
    # 转换为实数部分，并转换为 uint8 类型
        y = np.abs(y.real)
        y = np.real(y).astype(np.uint8)
    # 读取滤波图
    else:
        t = pic
        # 创建滤波器
        fi = np.zeros((M, N), dtype=complex)
        for i in range(M):
            for j in range(N):
                d = np.sqrt((i - u0) ** 2 + (j - v0) ** 2)
                fi[i, j] = 1 - np.exp(-(d ** 2) / (2 * (D0 ** 2)))
        for k in range(0, 3):
            # 将输入图像进行傅立叶变换
            f = np.fft.fft2(pic[:, :, k])
            # 移至中间
            f = np.fft.fftshift(f)
            # 使用滤波器滤波变换后的图像
            y = fi * f
            # 反变换
            y = np.fft.ifftshift(y)
            y = np.fft.ifft2(y)
            # 转换为实数部分，并转换为 uint8 类型
            y = np.abs(y.real)
            y = np.real(y).astype(np.uint8)
            # 给单通道赋值
            t[:, :, k] = y
        # 其实是通道合并
        y = t
        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])

    end = time.time()
    print("高斯高通滤波用时：", end - start)
    pil_image = Image.fromarray(y.astype(np.uint8))
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-高斯低通滤波结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("高斯低通滤波")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义指数灰度变换
def mi_change(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    num = simpledialog.askinteger("输入", "请输入一个不为 0 的整数：", minvalue=1)
    if num is None:
        return
    start = time.time()
    mi_img = np.zeros(image.shape, dtype=np.float32)
    store_data1 = []
    for number in range(0, 256):
        store_data1.append(math.pow(number, num))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            mi_img[i, j, 0] = store_data1[image[i, j, 0]]
            mi_img[i, j, 1] = store_data1[image[i, j, 1]]
            mi_img[i, j, 2] = store_data1[image[i, j, 2]]
    # normalize将矩阵线性归一到0-255
    cv2.normalize(mi_img, mi_img, 0, 255, cv2.NORM_MINMAX)
    # 转回uint8类型
    y = np.array(mi_img, dtype=np.uint8)
    end = time.time()
    print("指数灰度变换用时：", end - start)

    b, g, r = cv2.split(y)
    y = cv2.merge([r, g, b])
    pil_image = Image.fromarray(y)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-指数灰度变换结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("彩色图像指数灰度变换")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义伽马校正
def gama_change(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    gama = simpledialog.askfloat("输入", "请输入一个大于0的数作为伽马值γ：", minvalue=0.001)
    if gama is None:
        return
    start = time.time()

    gama_img = np.zeros(image.shape, dtype=np.float32)
    # store_data = {}
    store_data1 = []

    for num in range(0, 256):
        store_data1.append(256 * math.pow(num / 256, 1 / gama))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gama_img[i, j, 0] = store_data1[image[i, j, 0]]
            gama_img[i, j, 1] = store_data1[image[i, j, 1]]
            gama_img[i, j, 2] = store_data1[image[i, j, 2]]
    # 转回uint8类型
    gama_img = np.array(gama_img, dtype=np.uint8)
    end = time.time()
    print("伽马校正用时：", end - start)
    b, g, r = cv2.split(gama_img)
    gama_img = cv2.merge([r, g, b])
    pil_image = Image.fromarray(gama_img)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-" + str(gama) + "伽马校正变换结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("伽马校正")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")

    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义均值滤波
def mean_filtering(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    start = time.time()
    h, w, c = image.shape
    size = 3
    # 零填充
    pad = (size - 1) // 2
    pad_array = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float32)
    pad_array[pad:pad + h, pad:pad + w] = image.copy().astype(np.float32)
    # 卷积的过程
    tmp_array = image.copy()
    for ci in range(c):
        tmp_array[:, :, ci] = np.mean(
            np.lib.stride_tricks.sliding_window_view(pad_array[:, :, ci], (size, size)).reshape(h, w, -1), axis=2)
    tmp_array = tmp_array.astype(np.uint8)
    end = time.time()
    print("均值滤波用时：", end - start)
    b, g, r = cv2.split(tmp_array)
    output_image = cv2.merge([r, g, b])
    pil_image = Image.fromarray(output_image)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-均值滤波结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("3*3均值滤波")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义中值滤波
def median_filtering(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    start = time.time()
    size = 3
    h, w, c = image.shape
    # 零填充
    pad = (size - 1) // 2
    pad_array = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float32)
    pad_array[pad:pad + h, pad:pad + w] = image.copy().astype(np.float32)

    # 卷积的过程
    tmp_array = image.copy()
    for ci in range(c):
        tmp_array[:, :, ci] = np.median(
            np.lib.stride_tricks.sliding_window_view(pad_array[:, :, ci], (size, size)).reshape(h, w, -1), axis=2)
    tmp_array = tmp_array.astype(np.uint8)

    end = time.time()
    print("中值滤波用时：", end - start)
    b, g, r = cv2.split(tmp_array)
    output_image = cv2.merge([r, g, b])
    pil_image = Image.fromarray(output_image)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-中值滤波结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("3*3中值滤波")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义直方图均衡化
def histogram_equalization(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    start = time.time()
    image_array = np.array(image, dtype=np.uint8)
    output_image = np.zeros(image.shape, dtype=np.uint8)
    size = image_array.shape[0] * image_array.shape[1]
    for ci in range(image_array.shape[2]):
        counter_channel = Counter(image_array[:, :, ci].flatten())
        a = {k: v / size for k, v in counter_channel.items()}
        a = sorted(a.items(), key=lambda item: item[0])
        length = max(counter_channel.keys()) - min(counter_channel.keys())
        b = [[a[i][0], 0] for i in range(len(a))]
        b[0][1] = a[0][1]
        for i in range(1, len(a)):
            b[i][1] = b[i - 1][1] + a[i][1]
        c = {}
        error = 0
        for i in range(len(b)):
            c[b[i][0]] = round(b[i][1] * length)
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                try:
                    output_image[i][j][ci] = c[image[i][j][ci]]
                except KeyError:
                    output_image[i][j][ci] = 0
                    error += 1
        if error > 30:
            messagebox.showerror("错误", "该图片受损，无法进行直方图均衡化")
            return
    end = time.time()
    print("直方图均衡化用时：", end - start)
    b, g, r = cv2.split(output_image)
    output_image = cv2.merge([r, g, b])
    pil_image = Image.fromarray(output_image)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-直方图均衡化结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("直方图均衡化")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义彩色负片
def color_negative(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    start = time.time()
    input_image = np.array(image, dtype=np.uint8)
    output_image = np.ones(image.shape, dtype=np.uint8) * 255
    output_image = output_image - input_image
    end = time.time()
    print("负片化用时：", end - start)
    b, g, r = cv2.split(output_image)
    output_image = cv2.merge([r, g, b])
    pil_image = Image.fromarray(output_image)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-彩色负片化结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("彩色负片")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 定义拉普拉斯锐化
def laplace_sharpening(event=None):
    global image_label, folder_path, image_names, image_index
    if not image_names:
        messagebox.showerror("错误", "尚未选择文件夹或该文件夹中无任何图片")
        return
    if len(image_names) == 0:
        messagebox.showerror("错误", "该文件夹中无任何图片")
        return
    # 获取当前图片的路径
    image_path = os.path.join(folder_path, image_names[image_index])
    image = cv2.imread(image_path)
    if len(image.shape) != 3:
        messagebox.showerror("错误", "请选择彩色图片")
        return
    start = time.time()
    image_array = np.array(image, dtype=np.uint8)
    h, w, c = image_array.shape
    laplace_array = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    output_image = np.zeros(image.shape, dtype=np.uint8)
    for ci in range(c):
        output_image[1:h - 1, 1:w - 1, ci] = abs(np.dot(
            np.lib.stride_tricks.sliding_window_view(image_array[:, :, ci], (3, 3)).reshape(h - 2, w - 2, -1),
            laplace_array))
    end = time.time()
    print("拉普拉斯锐化用时：", end - start)
    b, g, r = cv2.split(output_image)
    output_image = cv2.merge([r, g, b])
    pil_image = Image.fromarray(output_image)
    new_image_path = os.path.join(folder_path, image_names[image_index] + "-拉普拉斯锐化结果.png")
    # 将 PIL 图像转换为 PhotoImage 图像
    global new_image
    pil_image.thumbnail((1024, 768))
    new_image = ImageTk.PhotoImage(pil_image)

    # 创建新窗口
    new_window = tk.Toplevel(root)
    new_window.title("拉普拉斯锐化")
    new_window.geometry("1600x900")
    # 创建 Label 组件来显示图像
    new_image_label = tk.Label(new_window, image=new_image)
    new_image_label.pack()

    # 创建保存按钮
    def save_image():
        pil_image.save(new_image_path)
        messagebox.showinfo("提示", "已成功保存")
    save_button = tk.Button(new_window, text="保存", command=save_image)
    save_button.place(relx=0.3, rely=0.92, anchor="center")
    save_button.pack()

    # 创建关闭按钮
    close_button = tk.Button(new_window, text="关闭", command=new_window.destroy)
    close_button.place(relx=0.5, rely=0.92, anchor="center")
    close_button.pack()


# 创建当前图片文件名的标签
label_text = StringVar()
label = Label(root, textvariable=label_text)
label.pack()

# 创建 "打开文件夹" 按钮
open_button = Button(root, text="打开文件夹", command=open_folder)
open_button.place(relx=0.5, rely=0.92, anchor="center")

# 创建 "上一张" 按钮
prev_button = Button(root, text="上一张")
prev_button.place(relx=0.3, rely=0.92, anchor="center")
prev_button.bind("<Button-1>", prev_image)
root.bind_all("<Left>", prev_image)

# 创建 "下一张" 按钮
next_button = Button(root, text="下一张")
next_button.place(relx=0.7, rely=0.92, anchor="center")
next_button.bind("<Button-1>", next_image)
root.bind_all("<Right>", next_image)

# 创建 "指数灰度变换" 按钮
mi_change_button = Button(root, text="指数灰度变换", command=mi_change)
mi_change_button.place(relx=0.95, rely=0.20, anchor="center")

# 创建 "伽马校正变换" 按钮
gama_change_button = Button(root, text="伽马校正", command=gama_change)
gama_change_button.place(relx=0.95, rely=0.25, anchor="center")

# 创建 "均值滤波" 按钮
mean_filtering_button = Button(root, text="均值滤波", command=mean_filtering)
mean_filtering_button.place(relx=0.95, rely=0.30, anchor="center")

# 创建 "中值滤波" 按钮
median_filtering_button = Button(root, text="中值滤波", command=median_filtering)
median_filtering_button.place(relx=0.95, rely=0.35, anchor="center")

# 创建 "直方图均衡化" 按钮
histogram_equalization_button = Button(root, text="直方图均衡化", command=histogram_equalization)
histogram_equalization_button.place(relx=0.95, rely=0.40, anchor="center")

# 创建 "负片化" 按钮
color_negative_button = Button(root, text="负片化", command=color_negative)
color_negative_button.place(relx=0.95, rely=0.45, anchor="center")

# 创建 "拉普拉斯锐化" 按钮
laplace_sharpening_button = Button(root, text="拉普拉斯锐化", command=laplace_sharpening)
laplace_sharpening_button.place(relx=0.95, rely=0.50, anchor="center")

# 创建 "傅里叶变换" 按钮
fourier_button = Button(root, text="傅里叶变换频谱", command=fourier_transform)
fourier_button.place(relx=0.95, rely=0.55, anchor="center")

# 创建 "高斯低通滤波" 按钮
lowpass_button = Button(root, text="高斯低通滤波", command=lowpass_filter)
lowpass_button.place(relx=0.95, rely=0.60, anchor="center")

# 创建 "高斯高通滤波" 按钮
highpass_button = Button(root, text="高斯高通滤波", command=highpass_filter)
highpass_button.place(relx=0.95, rely=0.65, anchor="center")

# 创建 "图像复原" 按钮
restore_button = Button(root, text="图像复原", command=image_restoration)
restore_button.place(relx=0.95, rely=0.70, anchor="center")

# 启动主窗口
root.mainloop()
