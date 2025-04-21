import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Scale
from threading import Thread
import queue

class ColorExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像颜色提取工具")
        
        # 设置窗口全屏
        self.root.state('zoomed')  # Windows系统
        # 对于Mac/Linux可以使用：
        # self.root.attributes('-zoomed', True)  # Linux
        # self.root.attributes('-fullscreen', True)  # 全屏模式
        
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.update_queue = queue.Queue()
        self.update_pending = False
        
        # 阈值参数
        self.r_min = tk.IntVar(value=90)
        self.r_max = tk.IntVar(value=255)
        self.g_min = tk.IntVar(value=0)
        self.g_max = tk.IntVar(value=90)
        self.b_min = tk.IntVar(value=0)
        self.b_max = tk.IntVar(value=100)
        
        # 创建UI
        self.create_widgets()
        
        # 检查更新队列
        self.root.after(100, self.process_queue)
    
    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        # 按钮
        tk.Button(control_frame, text="打开图片", command=self.open_image).pack(pady=5, fill=tk.X)
        tk.Button(control_frame, text="保存结果", command=self.save_image).pack(pady=5, fill=tk.X)
        
        # 阈值调节滑块
        self.create_slider(control_frame, "红色最小值:", self.r_min, 0, 255)
        self.create_slider(control_frame, "红色最大值:", self.r_max, 0, 255)
        self.create_slider(control_frame, "绿色最小值:", self.g_min, 0, 255)
        self.create_slider(control_frame, "绿色最大值:", self.g_max, 0, 255)
        self.create_slider(control_frame, "蓝色最小值:", self.b_min, 0, 255)
        self.create_slider(control_frame, "蓝色最大值:", self.b_max, 0, 255)
        
        # 图像显示区域
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定滑块变化事件
        self.r_min.trace_add("write", lambda *_: self.schedule_update())
        self.r_max.trace_add("write", lambda *_: self.schedule_update())
        self.g_min.trace_add("write", lambda *_: self.schedule_update())
        self.g_max.trace_add("write", lambda *_: self.schedule_update())
        self.b_min.trace_add("write", lambda *_: self.schedule_update())
        self.b_max.trace_add("write", lambda *_: self.schedule_update())
    
    def create_slider(self, parent, label, variable, from_, to):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label, width=12, anchor="w").pack(side=tk.LEFT)
        tk.Scale(frame, variable=variable, from_=from_, to=to, 
                orient=tk.HORIZONTAL, command=lambda x: self.schedule_update()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            self.display_image = self.original_image.copy()
            self.show_image()
            self.schedule_update()  # 打开图片后立即更新
    
    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if save_path:
                self.processed_image.save(save_path)
    
    def schedule_update(self):
        """将更新请求放入队列，避免频繁更新"""
        if not self.update_pending and self.original_image is not None:
            self.update_pending = True
            self.root.after(200, self.process_update)
    
    def process_update(self):
        """处理图像更新"""
        if self.original_image is None:
            self.update_pending = False
            return
            
        # 获取当前阈值
        r_min = self.r_min.get()
        r_max = self.r_max.get()
        g_min = self.g_min.get()
        g_max = self.g_max.get()
        b_min = self.b_min.get()
        b_max = self.b_max.get()
        
        # 在后台线程中处理图像
        Thread(target=self.process_image_async, 
              args=(r_min, r_max, g_min, g_max, b_min, b_max), 
              daemon=True).start()
        
        self.update_pending = False
    
    def process_image_async(self, r_min, r_max, g_min, g_max, b_min, b_max):
        """在后台线程中处理图像"""
        try:
            img_array = np.array(self.original_image)
            
            # 分离RGB通道
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            # 创建颜色掩膜
            mask = (
                (r >= r_min) & (r <= r_max) &
                (g >= g_min) & (g <= g_max) &
                (b >= b_min) & (b <= b_max))
            
            # 创建结果图像（非目标区域设为白色）
            result = img_array.copy()
            result[~mask] = [255, 255, 255]
            
            # 将结果放入队列
            self.update_queue.put(Image.fromarray(result))
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            self.update_queue.put(None)  # 确保队列不会阻塞
    
    def process_queue(self):
        """处理队列中的更新请求"""
        try:
            while True:
                try:
                    result = self.update_queue.get_nowait()
                    if result is not None:
                        self.processed_image = result
                        self.display_image = result.copy()
                        self.show_image()
                        print(f"图像已更新")
                except queue.Empty:
                    break
        finally:
            self.root.after(100, self.process_queue)
    
    def show_image(self):
        if self.display_image is not None:
            # 清除画布
            self.canvas.delete("all")
            
            # 获取画布尺寸
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
                
            # 计算缩放比例
            img_width, img_height = self.display_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_size = (int(img_width*ratio), int(img_height*ratio))
            
            # 缩放图像
            display_img = self.display_image.resize(new_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            
            # 在画布上显示图像
            self.canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=photo, anchor=tk.CENTER
            )
            
            # 保持引用防止被垃圾回收
            self.canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorExtractorApp(root)
    root.mainloop()