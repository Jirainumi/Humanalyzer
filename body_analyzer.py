import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime


class BodyAnalyzer:
    """
    Görüntüdeki vücut oranlarını analiz eden sınıf
    """

    def __init__(self):
        """Vücut analizi için gerekli araçları başlatır."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)

        # Vücut analiz değerlendirme kriterleri
        self.body_types = {
            "ektomorf": "İnce ve uzun yapılı vücut tipi",
            "mezomorf": "Atletik ve kaslı vücut tipi",
            "endomorf": "Yuvarlak hatlı vücut tipi"
        }

        # Analiz tarihi ve kullanıcı bilgisi
        self.current_date = datetime.now()
        self.current_user = "Admin"

    def calculate_body_ratios(self, landmarks, image_shape):
        """
        Vücut landmark'larından oran hesaplamaları yapar
        """
        h, w = image_shape[:2]
        landmark_points = {}

        # Önemli noktaların koordinatlarını al
        for idx, landmark in enumerate(landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            landmark_points[idx] = (x, y, visibility)

        # Yeterli görünürlük kontrolü
        valid_points = sum(1 for _, _, v in landmark_points.values() if v > 0.7)
        if valid_points < 20:  # En az 20 nokta istiyoruz
            return None

        # Ölçümleri hesapla
        measurements = {}

        # Boy
        if 0 in landmark_points and 27 in landmark_points:
            nose = landmark_points[0]
            foot = landmark_points[27] if landmark_points[27][2] > landmark_points[28][2] else landmark_points[28]
            height = abs(foot[1] - nose[1])
            measurements['height'] = height

        # Omuz genişliği
        if 11 in landmark_points and 12 in landmark_points:
            left_shoulder = landmark_points[11]
            right_shoulder = landmark_points[12]
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            measurements['shoulder_width'] = shoulder_width

        # Göğüs genişliği
        if 23 in landmark_points and 24 in landmark_points:
            left_hip = landmark_points[23]
            right_hip = landmark_points[24]
            hip_width = abs(right_hip[0] - left_hip[0])
            measurements['hip_width'] = hip_width

        # Bel genişliği (yaklaşık)
        if 11 in landmark_points and 12 in landmark_points and 23 in landmark_points and 24 in landmark_points:
            left_shoulder = landmark_points[11]
            right_shoulder = landmark_points[12]
            left_hip = landmark_points[23]
            right_hip = landmark_points[24]

            # Bel noktalarını hesapla (omuz ve kalça arasında)
            left_waist_x = left_shoulder[0] + (left_hip[0] - left_shoulder[0]) * 0.6
            left_waist_y = left_shoulder[1] + (left_hip[1] - left_shoulder[1]) * 0.6

            right_waist_x = right_shoulder[0] + (right_hip[0] - right_shoulder[0]) * 0.6
            right_waist_y = right_shoulder[1] + (right_hip[1] - right_shoulder[1]) * 0.6

            waist_width = abs(right_waist_x - left_waist_x)
            measurements['waist_width'] = waist_width

        # Bacak uzunluğu
        if 23 in landmark_points and 27 in landmark_points:
            hip = landmark_points[23]
            ankle = landmark_points[27]
            leg_length = abs(ankle[1] - hip[1])
            measurements['leg_length'] = leg_length

        return measurements

    def determine_body_type(self, measurements):
        """
        Vücut ölçülerine göre vücut tipini belirler
        """
        if not measurements or len(measurements) < 4:
            return None, None

        # Oranları hesapla
        shoulder_hip_ratio = measurements.get('shoulder_width', 0) / measurements.get('hip_width',
                                                                                      1) if 'hip_width' in measurements and 'shoulder_width' in measurements else 0
        waist_hip_ratio = measurements.get('waist_width', 0) / measurements.get('hip_width',
                                                                                1) if 'hip_width' in measurements and 'waist_width' in measurements else 0
        height_width_ratio = measurements.get('height', 0) / measurements.get('shoulder_width',
                                                                              1) if 'height' in measurements and 'shoulder_width' in measurements else 0

        # Vücut tipi skorları
        ecto_score = 0
        meso_score = 0
        endo_score = 0

        # Ektomorf (ince ve uzun)
        if height_width_ratio > 3.0:
            ecto_score += 3
        elif height_width_ratio > 2.5:
            ecto_score += 2
        elif height_width_ratio > 2.0:
            ecto_score += 1

        # Mezomorf (atletik ve geniş omuzlu)
        if shoulder_hip_ratio > 1.4:
            meso_score += 3
        elif shoulder_hip_ratio > 1.2:
            meso_score += 2
        elif shoulder_hip_ratio > 1.0:
            meso_score += 1

        if waist_hip_ratio < 0.85:
            meso_score += 2

        # Endomorf (yuvarlak hatlı)
        if waist_hip_ratio > 0.95:
            endo_score += 3
        elif waist_hip_ratio > 0.9:
            endo_score += 2
        elif waist_hip_ratio > 0.85:
            endo_score += 1

        if shoulder_hip_ratio < 1.1:
            endo_score += 2

        # En yüksek skoru alan vücut tipi
        scores = {
            "ektomorf": ecto_score,
            "mezomorf": meso_score,
            "endomorf": endo_score
        }

        primary_type = max(scores, key=scores.get)

        # İkinci baskın tipi bul
        scores[primary_type] = -1  # İlk tipi geçici olarak çıkar
        secondary_type = max(scores, key=scores.get)
        scores[
            primary_type] = ecto_score if primary_type == "ektomorf" else meso_score if primary_type == "mezomorf" else endo_score

        if scores[secondary_type] < 1:
            body_type = primary_type
        else:
            body_type = f"{primary_type}-{secondary_type}"

        # Detaylı açıklamalar için
        body_type_info = {
            "type": body_type,
            "primary": primary_type,
            "secondary": secondary_type if scores[secondary_type] >= 1 else None,
            "description": self.body_types.get(primary_type, ""),
            "shoulder_hip_ratio": round(shoulder_hip_ratio, 2),
            "waist_hip_ratio": round(waist_hip_ratio, 2),
            "height_width_ratio": round(height_width_ratio, 2)
        }

        return body_type, body_type_info

    def analyze_body(self, image_path):
        """Görselden vücut analizi yapar."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görsel yüklenemedi: {image_path}")

        # RGB'ye çevirme (mediapipe RGB kullanır)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, image, None

        # Vücut oranlarını hesapla
        measurements = self.calculate_body_ratios(results.pose_landmarks, image.shape)

        if not measurements:
            return None, image, None

        # Vücut tipini belirle
        body_type, body_type_info = self.determine_body_type(measurements)

        if not body_type:
            return None, image, None

        # Analiz sonuçları
        result = {
            "vucut_tipi": body_type,
            "aciklama": body_type_info["description"],
            "oranlar": {
                "omuz_kalca_orani": body_type_info["shoulder_hip_ratio"],
                "bel_kalca_orani": body_type_info["waist_hip_ratio"],
                "boy_genislik_orani": body_type_info["height_width_ratio"]
            }
        }

        return result, image, results.pose_landmarks

    def visualize_results(self, image, landmarks, result):
        """Analiz sonuçlarını görselleştirir."""
        h, w = image.shape[:2]

        # Görüntüyü kopyala
        viz_image = image.copy()

        # Landmark'ları çiz
        if landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            # BGR -> RGB dönüşümü (mediapipe RGB kullanır)
            viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)

            # Landmark'ları çiz
            mp_drawing.draw_landmarks(
                viz_image_rgb,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # RGB -> BGR dönüşümü
            viz_image = cv2.cvtColor(viz_image_rgb, cv2.COLOR_RGB2BGR)

        # OpenCV görüntüsünü PIL görüntüsüne dönüştür
        viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(viz_image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Font ayarla
        try:
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
                # Normal ve büyük font
                font_small = ImageFont.truetype(font_path, 12)
                font_large = ImageFont.truetype(font_path, 18)
            else:
                font_small = ImageFont.load_default()
                font_large = ImageFont.load_default()
        except Exception as e:
            print(f"Font yükleme hatası: {e}")
            font_small = ImageFont.load_default()
            font_large = ImageFont.load_default()

        # Renk tanımları
        text_color = (0, 0, 0)  # Siyah
        highlight_color = (255, 120, 0)  # Turuncu

        # Sonuçları görüntü üzerine ekle
        # Sağ üst köşeye sonuç kutusu
        margin = 10
        box_width = 240
        box_height = 160
        text_margin = 5

        # Yarı saydam beyaz arka plan
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            [(w - margin - box_width, margin), (w - margin, margin + box_height)],
            fill=(255, 255, 255, 200)
        )
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)

        # Başlık
        y_pos = margin + text_margin
        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            "VÜCUT ANALİZ SONUÇLARI",
            font=font_large,
            fill=highlight_color
        )
        y_pos += 25

        # Vücut tipi
        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            f"Vücut Tipi: {result['vucut_tipi'].upper()}",
            font=font_small,
            fill=text_color
        )
        y_pos += 20

        # Açıklama
        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            f"Açıklama: {result['aciklama']}",
            font=font_small,
            fill=text_color
        )
        y_pos += 20

        # Oranlar
        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            f"Omuz/Kalça: {result['oranlar']['omuz_kalca_orani']}",
            font=font_small,
            fill=text_color
        )
        y_pos += 20

        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            f"Bel/Kalça: {result['oranlar']['bel_kalca_orani']}",
            font=font_small,
            fill=text_color
        )
        y_pos += 20

        draw.text(
            (w - margin - box_width + text_margin, y_pos),
            f"Boy/Genişlik: {result['oranlar']['boy_genislik_orani']}",
            font=font_small,
            fill=text_color
        )

        # Alt bilgi - tarih ve kullanıcı
        formatted_date = self.current_date.strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"{formatted_date} - {self.current_user}"
        info_width = draw.textlength(info_text, font=font_small) if hasattr(draw, "textlength") else \
        font_small.getsize(info_text)[0]
        draw.text((w - info_width - 12, h - 25), info_text, font=font_small, fill=text_color)

        # PIL görüntüsünü OpenCV görüntüsüne dönüştür
        viz_image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return viz_image_with_text