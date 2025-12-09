import streamlit as st
import cv2
import numpy as np
import io
import os
import sqlite3
from datetime import datetime

# 假设您已将 MelanomaDetectorAdvanced 类和相关定义复制到此文件中或导入

# --- (将 calculate.py 中所有的 import 和 MelanomaDetectorAdvanced 类定义复制到此处) ---

# 定义参考物的实际直径 (5 毫米)
REFERENCE_DIAMETER_MM = 5.0


# ... (复制整个 MelanomaDetectorAdvanced 类的定义)

class MelanomaDetectorAdvanced:
    # (省略了类定义，请将您的整个类粘贴到这里)
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 保持原本的 resize 逻辑
        h, w = self.original_img.shape[:2]
        self.scale_factor = 1.0
        if w > 800:
            self.scale_factor = 800 / w
            self.original_img = cv2.resize(self.original_img, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        self.mask = None
        self.contour = None
        self.pixels_per_mm = None  # 新增：比例尺（每毫米有多少像素）
        self.melanoma_color_analysis = {}  # 新增：颜色分析结果
        self.melanoma_size_mm = {}  # 新增：尺寸分析结果

    def apply_gamma_correction(self, img, gamma=3.5):
        """
        关键步骤：Gamma 校正。
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def optimized_kmeans(self, img):
        # ... (此方法保持不变，用于痣的分割)
        # 1. 强力预处理：高斯模糊去噪 + Gamma校正增加对比度
        blur = cv2.GaussianBlur(img, (9, 9), 0)
        gamma_img = self.apply_gamma_correction(blur, gamma=1.5)

        # 2. 转换颜色空间：LAB (亮度+红绿+蓝黄)
        lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)

        # reshape
        pixel_values = lab.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # 3. 增加 K 值到 4
        k = 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 4. 找到最“暗”的簇（L通道值最小的）
        l_values = centers[:, 0]
        sorted_indices = np.argsort(l_values)
        target_label = sorted_indices[0]  # 取最暗的一个簇

        # 生成 Mask
        labels = labels.flatten()
        mask = (labels == target_label).astype(np.uint8) * 255
        mask = mask.reshape(img.shape[:2])

        return mask

    def post_process_mask(self, mask):
        # ... (此方法保持不变，用于后处理和连通域选择)
        # 1. 形态学操作：先开运算（去噪点），再闭运算（填孔）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. 空间约束：只保留靠近图像中心的那个连通域 (假设痣在中央)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        h, w = mask.shape
        center_x, center_y = w // 2, h // 2

        best_label = 0
        min_dist = float('inf')

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            dist_to_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue

            if dist_to_center < min_dist:
                min_dist = dist_to_center
                best_label = i

        new_mask = np.zeros_like(mask)
        new_mask[labels == best_label] = 255

        return new_mask

    ## 新增功能 1：识别蓝色贴纸并计算比例尺
    def _detect_scale_reference(self):
        """
        在图像左上角寻找蓝色圆形参考物，并计算每毫米对应的像素数。
        """
        img = self.original_img.copy()

        # 1. 颜色空间转换到 HSV，以便更好地分割蓝色
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 蓝色在 HSV 中的范围 (需要包含深蓝和浅蓝)
        # 注意：OpenCV 的 H 范围是 [0, 179]
        # 蓝色通常在 100-140 附近
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 寻找轮廓
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("警告：未找到蓝色参考物。无法进行尺寸分析。")
            return

        # 寻找面积最大的轮廓 (假设它是 5mm 贴纸)
        ref_contour = max(contours, key=cv2.contourArea)

        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(ref_contour)

        # 圆的像素直径
        pixel_diameter = 2 * radius

        # 计算比例尺： 像素直径 / 实际直径 (5mm)
        self.pixels_per_mm = pixel_diameter / REFERENCE_DIAMETER_MM

        print(f"✅ 比例尺检测完成：每毫米约 {self.pixels_per_mm:.2f} 像素。")

    ## 新增功能 2：分析痣的尺寸和颜色
    def analyze_melanoma_properties(self):
        if self.contour is None:
            print("错误：未检测到痣的轮廓，无法分析属性。")
            return

        # --- 尺寸分析 ---
        if self.pixels_per_mm:
            # 1. 计算最小外接圆/椭圆的尺寸
            # 最小外接圆直径
            (x, y), radius = cv2.minEnclosingCircle(self.contour)
            pixel_diameter_min_circle = 2 * radius

            # 最小外接矩形 (得到长轴和短轴)
            rect = cv2.minAreaRect(self.contour)
            width_pixel, height_pixel = rect[1]

            # 确保 width 是短轴，height 是长轴
            long_axis_pixel = max(width_pixel, height_pixel)
            short_axis_pixel = min(width_pixel, height_pixel)

            # 2. 转换为实际尺寸 (mm)
            long_axis_mm = long_axis_pixel / self.pixels_per_mm
            short_axis_mm = short_axis_pixel / self.pixels_per_mm

            # 3. 计算面积
            pixel_area = cv2.contourArea(self.contour)
            # 面积转换需要除以比例尺的平方 (像素/mm * 像素/mm)
            area_mm2 = pixel_area / (self.pixels_per_mm ** 2)

            self.melanoma_size_mm = {
                "longer axis (mm)": f"{long_axis_mm:.2f}",
                "shorter axis (mm)": f"{short_axis_mm:.2f}",
                "Minimum Enclosing Circle Diameter (mm)": f"{long_axis_mm:.2f}",  # 通常取长轴作为最大的尺寸
                "square (mm^2)": f"{area_mm2:.2f}"
            }
            print(f"✅ 痣的尺寸分析完成: 长轴 {long_axis_mm:.2f} mm, 面积 {area_mm2:.2f} mm^2。")
        else:
            print("⚠️ 无法进行尺寸分析，因为未检测到比例尺。")

        # --- 颜色分析 ---
        # 1. 提取痣区域的 BGR 像素值
        img_bgr = self.original_img
        # 使用 mask 提取原图中的像素
        pixels_bgr = img_bgr[self.mask == 255]

        if len(pixels_bgr) < 10:  # 如果分割区域太小，则跳过
            self.melanoma_color_analysis = {"主要颜色": "分割区域太小"}
            return

        # 2. 对 BGR 像素进行 K-Means 聚类，找出主要颜色
        # 聚类数量 K=3 或 K=4 即可，用于识别核心、边缘、和高光/阴影
        n_colors = 3
        # 转换为 float32 for K-Means
        pixels_bgr = np.float32(pixels_bgr)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        # centers 即是聚类中心，即代表的颜色 (BGR)
        _, labels, centers = cv2.kmeans(pixels_bgr, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 3. 计算每种颜色的比例
        unique, counts = np.unique(labels, return_counts=True)
        counts = counts.flatten()

        total_pixels = len(pixels_bgr)

        color_results = []
        for i in range(n_colors):
            b, g, r = centers[i].astype(int)
            percentage = (counts[i] / total_pixels) * 100

            color_results.append({
                "BGR": (b, g, r),
                "HEX": f"#{r:02x}{g:02x}{b:02x}",
                "比例 (%)": f"{percentage:.1f}"
            })

        # 4. 按比例排序，并存储结果
        color_results.sort(key=lambda x: float(x["比例 (%)"]), reverse=True)
        self.melanoma_color_analysis = color_results

        print("✅ 痣的颜色分析完成。")

    # 为了在 Streamlit 中运行，请务必移除 cv2.imshow(), cv2.waitKey(0), cv2.destroyAllWindows()
    # 并且将 run 方法改为返回结果字典和处理后的图像，而不是直接显示。
    # ...

    def run(self):
        print("--- 1. 图像分割 ---")
        raw_mask = self.optimized_kmeans(self.original_img)
        self.mask = self.post_process_mask(raw_mask)

        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ★ 关键修复：检测 contours，而不是 self.contour
        if contours:
            self.contour = max(contours, key=cv2.contourArea)
            print("轮廓提取成功。")
        else:
            print("未检测到病灶")
            return {"status": "Analysis Failed: Contour not found"}, self.original_img

        print("--- 2. 比例尺检测与尺寸分析 ---")
        self._detect_scale_reference()
        self.analyze_melanoma_properties()

        print("--- 3. 可视化和结果输出 ---")
        output = self.original_img.copy()
        cv2.drawContours(output, [self.contour], -1, (0, 255, 0), 2)

        # 返回 Streamlit 需要的结果
        results = {
            "size_analysis": self.melanoma_size_mm,
            "color_analysis": self.melanoma_color_analysis,
            "pixels_per_mm": self.pixels_per_mm,
            "status": "Analysis Success"
        }

        return results, output


# --- Streamlit 应用程序 ---

DB_NAME = 'melanoma_data.db'


def init_db():
    """初始化 SQLite 数据库"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            filename TEXT,
            long_axis_mm REAL,
            area_mm2 REAL,
            main_color_hex TEXT,
            full_results TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_result_to_db(filename, results):
    """将分析结果保存到数据库"""
    if results["status"] != "Analysis Success":
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 提取关键数据点
    long_axis = results["size_analysis"].get("longer axis (mm)", "N/A")
    area = results["size_analysis"].get("square (mm^2)", "N/A")
    main_color_hex = results["color_analysis"][0]["HEX"] if results["color_analysis"] else "N/A"

    import json
    import numpy as np

    # 🔥 核心修复：把 numpy 类型全部转 Python 类型
    full_results_json = json.dumps(
        results,
        default=lambda x: float(x) if isinstance(x, np.generic) else x
    )

    c.execute(
        "INSERT INTO analysis_results (timestamp, filename, long_axis_mm, area_mm2, main_color_hex, full_results) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, filename, long_axis, area, main_color_hex, full_results_json)
    )

    conn.commit()
    conn.close()



def main():
    st.set_page_config(page_title="🔬 黑色素瘤图像分析工具", layout="wide")
    st.title("🔬 黑色素瘤图像分析工具")

    init_db()

    # --- 文件上传 ---
    uploaded_file = st.file_uploader("上传皮肤镜图像 (要求包含蓝色 5mm 参考贴纸)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.subheader(f"🖼️ 正在分析文件: **{file_name}**")

        # 1. 保存文件到临时路径
        with open(os.path.join("./", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 运行分析
        detector = MelanomaDetectorAdvanced(uploaded_file.name)

        with st.spinner('正在进行图像分割和特性分析...'):
            results, processed_img = detector.run()

        st.success("✅ 分析完成！")

        # 3. 结果展示
        col1, col2 = st.columns(2)

        with col1:
            st.image(detector.original_img, caption="原始图像 (已缩放)", use_column_width=True, channels="BGR")

        with col2:
            st.image(processed_img, caption="分析结果 (绿色轮廓)", use_column_width=True, channels="BGR")

        # --- 详细数据表格 ---
        st.subheader("📊 尺寸分析")
        if results["status"] == "Analysis Success" and results["size_analysis"]:
            size_data = results["size_analysis"]
            st.table(size_data)

        st.subheader("🎨 颜色分析")
        if results["status"] == "Analysis Success" and results["color_analysis"]:
            # Streamlit 可以直接显示字典列表
            st.dataframe(results["color_analysis"], use_container_width=True)

        # 4. 数据存储
        if st.button("💾 保存本次分析结果到数据库"):
            save_result_to_db(file_name, results)
            st.balloons()
            st.success("数据已成功保存到 SQLite 数据库!")

    # --- 数据库查看 ---
    st.header("历史分析记录")
    if st.button("🔄 查看历史数据"):
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            "SELECT id, timestamp, filename, long_axis_mm, area_mm2, main_color_hex FROM analysis_results ORDER BY id DESC",
            conn)
        conn.close()
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    import pandas as pd  # 确保您在环境中安装了 pandas: pip install pandas

    main()