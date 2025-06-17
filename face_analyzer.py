import cv2
import numpy as np
import mediapipe as mp
from matplotlib.colors import rgb2hex
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import colorsys


class FaceAnalyzer:
    def __init__(self):
        """Yüz analizi için gerekli araçları başlatır."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)

        # Genişletilmiş renk sözlüğü
        self.color_names = {
            # Ten renkleri
            "açık": np.array([255, 223, 196]),
            "orta": np.array([210, 180, 140]),
            "koyu": np.array([150, 111, 77]),
            # Göz renkleri - genişletilmiş
            "mavi": np.array([79, 119, 189]),
            "mavi-gri": np.array([96, 112, 122]),  # #60707a gibi gri-mavi tonlar için
            "açık-mavi": np.array([128, 185, 216]),
            "yeşil": np.array([79, 156, 102]),
            "açık-yeşil": np.array([145, 195, 150]),
            "kahverengi": np.array([101, 67, 33]),
            "açık-kahverengi": np.array([150, 100, 60]),
            "koyu-kahverengi": np.array([60, 40, 20]),
            "kehribar": np.array([196, 142, 14]),  # Amber
            "gri": np.array([128, 128, 128]),
            "siyah": np.array([45, 40, 40])
        }

        # Yüz şekilleri tanımları
        self.face_shapes = {
            "oval": "Oval",
            "yuvarlak": "Yuvarlak",
            "kare": "Kare",
            "kalp": "Kalp",
            "dikdörtgen": "Dikdörtgen",
            "elmas": "Elmas"
        }

        # Yüz şekli için önemli landmark indeksleri
        self.face_shape_landmarks = {
            "çene": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234],  # Çene hattı
            "alın": [10, 8, 109, 67, 103, 54, 21, 162, 139],  # Alın
            "çene_ucu": [152, 175, 199, 200, 18, 313, 400, 379, 378],  # Çene ucu
            "yanaklar": [123, 50, 36, 206, 203, 182, 42, 267, 356, 389, 264]  # Yanaklar
        }

        # Analiz tarihi ve kullanıcı bilgisi
        self.current_date = datetime.now()
        self.current_user = "admin"

    def rgb_to_hsv(self, rgb):
        """RGB değerini HSV'ye çevirir"""
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

    def get_color_category(self, rgb_array, color_type="eye"):
        """
        RGB değerini HSV'ye çevirip renk kategorisini belirler
        HSV ile renk tonu (hue) daha doğru tespit edilir
        """
        h, s, v = self.rgb_to_hsv(rgb_array)

        # Ten rengi için basit kümeleme
        if color_type == "skin":
            if v > 75:  # Yüksek parlaklık (açık)
                return "açık"
            elif v > 50:  # Orta parlaklık
                return "orta"
            else:  # Düşük parlaklık (koyu)
                return "koyu"

        # Göz rengi için kapsamlı değerlendirme
        elif color_type == "eye":
            # Düşük doygunluk - gri, siyah
            if s < 15:
                if v < 30:
                    return "siyah"
                else:
                    return "gri"

            # Mavi tonları (160-240 hue)
            elif 160 <= h <= 240:
                if s < 30 and v < 60:
                    return "mavi-gri"  # #60707a gibi tonlar için
                elif v > 70:
                    return "açık-mavi"
                else:
                    return "mavi"

            # Yeşil tonları (80-160 hue)
            elif 80 <= h <= 160:
                if v > 70:
                    return "açık-yeşil"
                else:
                    return "yeşil"

            # Kahverengi ve amber tonları (0-60 hue)
            elif 0 <= h <= 60:
                if s > 60 and v > 60:
                    return "kehribar"
                elif v > 60:
                    return "açık-kahverengi"
                elif v < 30:
                    return "koyu-kahverengi"
                else:
                    return "kahverengi"

            # Varsayılan
            else:
                return "kahverengi"

        # Varsayılan olarak en yakın rengi bul
        return self.find_closest_color_name(rgb_array, color_type)

    def find_closest_color_name(self, rgb, color_type="eye"):
        """RGB değerine en yakın renk adını bulur"""
        min_distance = float('inf')
        closest_color = None

        # Ten rengi ve göz rengi için farklı renk grupları
        if color_type == "skin":
            colors = ["açık", "orta", "koyu"]
        else:  # eye
            colors = ["mavi", "mavi-gri", "açık-mavi", "yeşil", "açık-yeşil",
                      "kahverengi", "açık-kahverengi", "koyu-kahverengi", "kehribar",
                      "gri", "siyah"]

        for name in colors:
            sample = self.color_names[name]
            distance = np.sqrt(np.sum((rgb - sample) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        return closest_color

    def get_average_color(self, image, mask=None):
        """Belirtilen bölgenin ortalama rengini döndürür."""
        if mask is not None:
            # Maskeleme kullanarak ortalama rengi bulma
            mean_color = cv2.mean(image, mask=mask)[:3]
        else:
            # Tüm görüntünün ortalamasını alma
            mean_color = np.mean(image, axis=(0, 1))[:3]

        # BGR'den RGB'ye çevirme
        rgb_color = np.array(mean_color[::-1], dtype=np.int32)
        return rgb_color

    def create_mask_from_landmarks(self, image, landmarks, indices):
        """Landmark indekslerine göre bir maske oluşturur."""
        h, w = image.shape[:2]
        points = []

        for idx in indices:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))

        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(points, np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    def analyze_face_shape(self, landmarks, image_shape):
        """Yüz şeklini analiz eder"""
        h, w = image_shape[:2]

        # Önemli yüz noktalarını topla
        face_points = {}
        for region, indices in self.face_shape_landmarks.items():
            points = []
            for idx in indices:
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
            face_points[region] = np.array(points)

        # Yüz oranlarını hesapla
        face_width = 0
        face_height = 0

        # Yüz genişliği (kulaklar arası)
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        face_width = int(abs(right_ear.x - left_ear.x) * w)

        # Yüz uzunluğu (alın üstünden çene ucuna)
        forehead_top = landmarks.landmark[10]
        chin_bottom = landmarks.landmark[152]
        face_height = int(abs(chin_bottom.y - forehead_top.y) * h)

        # Çene genişliği
        left_jaw = landmarks.landmark[93]
        right_jaw = landmarks.landmark[323]
        jaw_width = int(abs(right_jaw.x - left_jaw.x) * w)

        # Alın genişliği
        left_forehead = landmarks.landmark[103]
        right_forehead = landmarks.landmark[332]
        forehead_width = int(abs(right_forehead.x - left_forehead.x) * w)

        # Çene açısı
        left_cheekbone = landmarks.landmark[123]
        right_cheekbone = landmarks.landmark[352]
        cheekbone_width = int(abs(right_cheekbone.x - left_cheekbone.x) * w)

        # En/boy oranı
        ratio = face_width / face_height if face_height > 0 else 0

        # Oranlardan yüz şeklini belirle
        face_shape = "belirsiz"

        if ratio > 0.8 and ratio < 0.9:
            # Oval yüz: Uzunluk genişlikten biraz fazla, çene köşeli değil
            if jaw_width < cheekbone_width and forehead_width > jaw_width:
                face_shape = "oval"

        elif ratio >= 0.9 and ratio < 1:
            # Yuvarlak yüz: Neredeyse kare, ama köşeleri daha yumuşak
            if jaw_width < cheekbone_width and forehead_width > jaw_width:
                face_shape = "yuvarlak"
            # Kare yüz: Neredeyse kare ve çene köşeli
            elif jaw_width >= cheekbone_width * 0.9:
                face_shape = "kare"

        elif ratio < 0.8:
            # Uzun yüz: Belirgin şekilde uzun
            if forehead_width > jaw_width:
                if cheekbone_width > forehead_width and cheekbone_width > jaw_width:
                    face_shape = "elmas"
                else:
                    face_shape = "dikdörtgen"
            # Kalp şeklinde yüz: Geniş alın, ince çene
            elif forehead_width > jaw_width * 1.2:
                face_shape = "kalp"

        # Eğer belirsiz kaldıysa, en genel şekil olan ovali döndür
        if face_shape == "belirsiz":
            face_shape = "oval"

        # Detaylı veri döndür
        face_shape_data = {
            "shape": face_shape,
            "width": face_width,
            "height": face_height,
            "ratio": ratio,
            "jaw_width": jaw_width,
            "forehead_width": forehead_width,
            "cheekbone_width": cheekbone_width
        }

        return face_shape_data

    def analyze_face(self, image_path):
        """Görselden yüz analizi yaparak ten, göz rengi ve yüz şeklini belirler."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görsel yüklenemedi: {image_path}")

        # RGB'ye çevirme (mediapipe RGB kullanır)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, image, None, None, None

        # İlk yüzü al
        face_landmarks = results.multi_face_landmarks[0]

        # Yüz şeklini analiz et
        face_shape_data = self.analyze_face_shape(face_landmarks, image.shape)

        # Yüz bölgesi indeksleri (yanak ve alın)
        skin_indices = [
            # Yanaklar
            123, 50, 101, 36, 206, 94, 139, 137, 262, 359, 356, 389,
            # Alın
            66, 69, 109, 10, 338, 297, 299, 296
        ]

        # Göz bölgesi indeksleri (sağ ve sol göz)
        right_eye_indices = [33, 133, 159, 145, 153, 154, 155, 133]
        left_eye_indices = [362, 263, 386, 374, 380, 381, 382, 362]

        # Maskeler oluştur
        skin_mask = self.create_mask_from_landmarks(image, face_landmarks, skin_indices)
        right_eye_mask = self.create_mask_from_landmarks(image, face_landmarks, right_eye_indices)
        left_eye_mask = self.create_mask_from_landmarks(image, face_landmarks, left_eye_indices)

        # Renkleri analiz et
        try:
            skin_color = self.get_average_color(image, skin_mask)
            right_eye_color = self.get_average_color(image, right_eye_mask)
            left_eye_color = self.get_average_color(image, left_eye_mask)

            # Ortalama göz rengi - float hesaplamalarını int'e dönüştür
            eye_color = np.array(((right_eye_color + left_eye_color) / 2), dtype=np.int32)

            # HEX değerlerini hesapla
            skin_hex = rgb2hex(skin_color / 255)
            eye_hex = rgb2hex(eye_color / 255)

            # HSV tabanlı geliştirilmiş renk sınıflandırması
            skin_color_name = self.get_color_category(skin_color, "skin")
            eye_color_name = self.get_color_category(eye_color, "eye")

            # Analiz sonuçlarını döndür
            result = {
                "ten_rengi": {
                    "rgb": skin_color.tolist(),
                    "hex": skin_hex,
                    "tahmini_renk": skin_color_name
                },
                "goz_rengi": {
                    "rgb": eye_color.tolist(),
                    "hex": eye_hex,
                    "tahmini_renk": eye_color_name
                },
                "yuz_sekli": {
                    "sekil": face_shape_data["shape"],
                    "oran": round(face_shape_data["ratio"], 2)
                }
            }

            return result, image, skin_mask, right_eye_mask, left_eye_mask

        except Exception as e:
            print(f"Renk analizi hatası: {e}")
            # Hata durumunda varsayılan değerlerle sonuç döndür
            default_result = {
                "ten_rengi": {
                    "rgb": [150, 111, 77],
                    "hex": "#774b96",
                    "tahmini_renk": "koyu"
                },
                "goz_rengi": {
                    "rgb": [96, 112, 122],
                    "hex": "#60707a",
                    "tahmini_renk": "mavi-gri"
                },
                "yuz_sekli": {
                    "sekil": "belirsiz",
                    "oran": 0
                }
            }
            return default_result, image, skin_mask, right_eye_mask, left_eye_mask

    def visualize_results(self, image, skin_mask, right_eye_mask, left_eye_mask, result):
        """Analiz sonuçlarını görselleştirir ve Türkçe karakter desteği ile metin ekler."""
        h, w = image.shape[:2]

        # Maskeleri görselleştir
        viz_image = image.copy()

        if skin_mask is not None:
            # Ten rengi maskesi (kırmızı)
            viz_image[skin_mask > 0] = viz_image[skin_mask > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3

        if right_eye_mask is not None:
            # Sağ göz maskesi (yeşil)
            viz_image[right_eye_mask > 0] = viz_image[right_eye_mask > 0] * 0.7 + np.array([0, 255, 0],
                                                                                           dtype=np.uint8) * 0.3

        if left_eye_mask is not None:
            # Sol göz maskesi (mavi)
            viz_image[left_eye_mask > 0] = viz_image[left_eye_mask > 0] * 0.7 + np.array([255, 0, 0],
                                                                                         dtype=np.uint8) * 0.3

        # OpenCV görüntüsünü PIL görüntüsüne dönüştür (Türkçe karakter desteği için)
        pil_image = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Türkçe karakterleri destekleyen font ayarla
        try:
            # Farklı fontları dene (sistem fontlarına göre)
            possible_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "C:/Windows/Fonts/segoeui.ttf"  # Windows
            ]

            font_path = None
            for path in possible_fonts:
                if os.path.exists(path):
                    font_path = path
                    break

            if font_path:
                # Font boyutu küçültüldü
                font = ImageFont.truetype(font_path, 10)
            else:
                # Varsayılan font
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Font yükleme hatası: {e}")
            font = ImageFont.load_default()

        # Ten ve göz rengi değerlerini analiz sonuçlarından al
        ten_renk = result['ten_rengi']['tahmini_renk']
        goz_renk = result['goz_rengi']['tahmini_renk']
        yuz_sekli = result['yuz_sekli']['sekil']

        # Direk HEX değerlerini kullanarak metin renklendirme
        ten_hex = result['ten_rengi']['hex']
        goz_hex = result['goz_rengi']['hex']

        # HEX -> RGB dönüşümü
        ten_hex_rgb = tuple(int(ten_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        goz_hex_rgb = tuple(int(goz_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

        # Metin renkleri - doğrudan HEX'ten alınan renkler
        ten_text_color = ten_hex_rgb
        goz_text_color = goz_hex_rgb
        black_color = (0, 0, 0)  # Diğer metinler için siyah

        # Sol üst köşeye bilgileri ekle (opak katman olmadan)
        margin = 5
        y_pos = margin + 3

        # Ten rengi bilgisi - HEX'ten alınan renk ile
        draw.text((margin, y_pos), f"Ten: {ten_renk}", font=font, fill=ten_text_color)
        y_pos += 12

        # Göz rengi bilgisi - HEX'ten alınan renk ile
        draw.text((margin, y_pos), f"Göz: {goz_renk}", font=font, fill=goz_text_color)
        y_pos += 12

        # Yüz şekli bilgisi - siyah renk ile
        draw.text((margin, y_pos), f"Yüz Şekli: {yuz_sekli}", font=font, fill=black_color)
        y_pos += 12

        # RGB ve HEX bilgileri - siyah renk ile
        draw.text((margin, y_pos), f"Ten RGB: {result['ten_rengi']['rgb']}", font=font, fill=black_color)
        y_pos += 12
        draw.text((margin, y_pos), f"Ten HEX: {result['ten_rengi']['hex']}", font=font, fill=black_color)
        y_pos += 12
        draw.text((margin, y_pos), f"Göz HEX: {result['goz_rengi']['hex']}", font=font, fill=black_color)

        # Alt bilgi - siyah renk ile
        # datetime formatını string'e çevir
        formatted_date = self.current_date.strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"{formatted_date} - {self.current_user}"
        info_width = draw.textlength(info_text, font=font) if hasattr(draw, "textlength") else font.getsize(info_text)[
            0]
        draw.text((w - info_width - 7, h - 17), info_text, font=font, fill=black_color)

        # PIL görüntüsünü OpenCV görüntüsüne geri dönüştür
        viz_image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return viz_image_with_text