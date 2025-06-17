import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import datetime
import os
import threading
import time


class CaptureAnalyzer:
    """Kamera görüntüsünden gerçek zamanlı yüz analizi yapan sınıf"""

    def __init__(self, face_analyzer=None):
        """Kamera yakalama ve analiz sistemi için araçları başlatır."""
        # Face analyzer bağlantısı
        self.face_analyzer = face_analyzer

        # MediaPipe yüz mesh modeli
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Kamera ayarları
        self.camera = None
        self.camera_id = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30

        # Analiz ve gösterim ayarları
        self.running = False
        self.show_landmarks = True
        self.show_results = True
        self.analysis_interval = 0.5  # Analiz sıklığı (saniye)
        self.last_analysis_time = 0
        self.current_results = None

        # Font ayarları
        self.font = None
        self.load_font()

        # Analiz tarihi ve kullanıcı bilgisi
        self.current_date = datetime.datetime.now()
        self.current_user = "Jirainumi"

    def load_font(self):
        """Türkçe karakter desteği için font yükler"""
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
                self.font = ImageFont.truetype(font_path, 14)
            else:
                self.font = ImageFont.load_default()
        except Exception as e:
            print(f"Font yükleme hatası: {e}")
            self.font = ImageFont.load_default()

    def initialize_camera(self, camera_id=0, width=640, height=480, fps=30):
        """Kamerayı başlatır ve ayarları yapılandırır"""
        self.camera_id = camera_id
        self.frame_width = width
        self.frame_height = height
        self.fps = fps

        # Kamera nesnesi oluştur ve ayarla
        self.camera = cv2.VideoCapture(camera_id)

        if not self.camera.isOpened():
            raise ValueError(f"Kamera {camera_id} açılamadı")

        # Kamera çözünürlüğü ve FPS ayarları
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        # Gerçek ayarları kontrol et
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

        print(f"Kamera başlatıldı: {actual_width}x{actual_height} @ {actual_fps}fps")

        return self.camera.isOpened()

    def analyze_frame(self, frame):
        """Bir kare üzerinde yüz analizi yapar"""
        if self.face_analyzer is None:
            return None

        # RGB'ye dönüştür (MediaPipe RGB formatı kullanır)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe ile yüz işaretleri tespit et
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # İlk yüzü al
        face_landmarks = results.multi_face_landmarks[0]

        # Yüz analizi için mask oluşturma işlemi
        h, w = frame.shape[:2]

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

        # Maskeleri oluştur
        skin_mask = self.face_analyzer.create_mask_from_landmarks(frame, face_landmarks, skin_indices)
        right_eye_mask = self.face_analyzer.create_mask_from_landmarks(frame, face_landmarks, right_eye_indices)
        left_eye_mask = self.face_analyzer.create_mask_from_landmarks(frame, face_landmarks, left_eye_indices)

        # Renkleri analiz et
        try:
            skin_color = self.face_analyzer.get_average_color(frame, skin_mask)
            right_eye_color = self.face_analyzer.get_average_color(frame, right_eye_mask)
            left_eye_color = self.face_analyzer.get_average_color(frame, left_eye_mask)

            # Ortalama göz rengi
            eye_color = np.array(((right_eye_color + left_eye_color) / 2), dtype=np.int32)

            # Renk isimlerini bul - gelişmiş HSV sınıflandırması kullan
            skin_color_name = self.face_analyzer.get_color_category(skin_color, "skin")
            eye_color_name = self.face_analyzer.get_color_category(eye_color, "eye")

            # Yüz şeklini analiz et
            face_shape_data = self.face_analyzer.analyze_face_shape(face_landmarks, frame.shape)

            # Analiz sonuçları
            result = {
                "ten_rengi": {
                    "rgb": skin_color.tolist(),
                    "hex": rgb2hex(
                        skin_color / 255) if 'rgb2hex' in globals() else f"#{skin_color[0]:02x}{skin_color[1]:02x}{skin_color[2]:02x}",
                    "tahmini_renk": skin_color_name
                },
                "goz_rengi": {
                    "rgb": eye_color.tolist(),
                    "hex": rgb2hex(
                        eye_color / 255) if 'rgb2hex' in globals() else f"#{eye_color[0]:02x}{eye_color[1]:02x}{eye_color[2]:02x}",
                    "tahmini_renk": eye_color_name
                },
                "yuz_sekli": {
                    "sekil": face_shape_data["shape"],
                    "oran": round(face_shape_data["ratio"], 2)
                },
                "landmarks": face_landmarks,
                "masks": {
                    "skin": skin_mask,
                    "right_eye": right_eye_mask,
                    "left_eye": left_eye_mask
                }
            }

            return result

        except Exception as e:
            print(f"Kare analiz hatası: {e}")
            return None

    def visualize_frame(self, frame, analysis_result):
        """Analiz sonuçlarını kare üzerine görselleştirir"""
        if frame is None or analysis_result is None:
            return frame

        h, w = frame.shape[:2]
        viz_frame = frame.copy()

        # Landmark'ları göster
        if self.show_landmarks and "landmarks" in analysis_result:
            landmarks = analysis_result["landmarks"]
            # MediaPipe Drawing kullanarak landmark'ları çiz
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

            # RGB'ye dönüştür, çiz ve tekrar BGR'ye dönüştür
            frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
            mp_drawing.draw_landmarks(
                image=frame_rgb,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            viz_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Maskeleri göster (tercihe bağlı)
        if "masks" in analysis_result and False:  # Maskeleri varsayılan olarak gösterme
            masks = analysis_result["masks"]

            # Ten rengi maskesi (kırmızı)
            if "skin" in masks and masks["skin"] is not None:
                viz_frame[masks["skin"] > 0] = viz_frame[masks["skin"] > 0] * 0.7 + np.array([0, 0, 255],
                                                                                             dtype=np.uint8) * 0.3

            # Göz maskeleri (yeşil ve mavi)
            if "right_eye" in masks and masks["right_eye"] is not None:
                viz_frame[masks["right_eye"] > 0] = viz_frame[masks["right_eye"] > 0] * 0.7 + np.array([0, 255, 0],
                                                                                                       dtype=np.uint8) * 0.3

            if "left_eye" in masks and masks["left_eye"] is not None:
                viz_frame[masks["left_eye"] > 0] = viz_frame[masks["left_eye"] > 0] * 0.7 + np.array([255, 0, 0],
                                                                                                     dtype=np.uint8) * 0.3

        # Sonuçları göster
        if self.show_results:
            # OpenCV'den PIL'e dönüştür (Türkçe karakter desteği için)
            viz_frame_rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(viz_frame_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Ten ve göz rengi değerlerini al
            ten_renk = analysis_result['ten_rengi']['tahmini_renk']
            goz_renk = analysis_result['goz_rengi']['tahmini_renk']
            yuz_sekli = analysis_result['yuz_sekli']['sekil']

            # HEX değerleri
            ten_hex = analysis_result['ten_rengi']['hex']
            goz_hex = analysis_result['goz_rengi']['hex']

            # HEX -> RGB dönüşümü
            ten_hex_rgb = self.hex_to_rgb(ten_hex)
            goz_hex_rgb = self.hex_to_rgb(goz_hex)

            # Metin renkleri
            ten_text_color = ten_hex_rgb
            goz_text_color = goz_hex_rgb
            black_color = (0, 0, 0)  # Siyah
            white_color = (255, 255, 255)  # Beyaz

            # Arka plan için saydam dikdörtgen
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle([(10, 10), (220, 100)], fill=(255, 255, 255, 180))
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)

            # Bilgileri sol üst köşede göster
            margin = 15
            y_pos = margin

            # Başlık
            draw.text((margin, y_pos), "YÜZ ANALİZ SONUÇLARI", font=self.font, fill=black_color)
            y_pos += 20

            # Ten rengi
            draw.text((margin, y_pos), f"Ten: {ten_renk}", font=self.font, fill=ten_text_color)
            y_pos += 20

            # Göz rengi
            draw.text((margin, y_pos), f"Göz: {goz_renk}", font=self.font, fill=goz_text_color)
            y_pos += 20

            # Yüz şekli
            draw.text((margin, y_pos), f"Yüz Şekli: {yuz_sekli}", font=self.font, fill=black_color)

            # Alt bilgi - tarih ve kullanıcı
            formatted_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = "Canlı Analiz"
            info_width = draw.textlength(info_text, font=self.font) if hasattr(draw, "textlength") else \
            self.font.getsize(info_text)[0]
            draw.text((w - info_width - 15, h - 30), info_text, font=self.font, fill=white_color)

            # PIL'den OpenCV'ye dönüştür
            viz_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return viz_frame

    def hex_to_rgb(self, hex_color):
        """HEX renk kodunu RGB'ye çevirir"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def start_capture(self):
        """Kamera yakalamayı başlatır ve analiz döngüsünü çalıştırır"""
        # Kamera başlatılmamışsa başlat
        if self.camera is None or not self.camera.isOpened():
            self.initialize_camera()

        self.running = True
        capture_thread = threading.Thread(target=self._capture_thread)
        capture_thread.daemon = True
        capture_thread.start()

        return capture_thread

    def _capture_thread(self):
        """Kamera görüntülerini yakalar ve analiz eder (ayrı thread'de çalışır)"""
        try:
            while self.running:
                # Kare yakala
                ret, frame = self.camera.read()
                if not ret:
                    print("Kare yakalama hatası!")
                    break

                # Analiz zamanını kontrol et
                current_time = time.time()
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    # Kareyi analiz et
                    analysis_result = self.analyze_frame(frame)
                    if analysis_result:
                        self.current_results = analysis_result
                    self.last_analysis_time = current_time

                # Sonuçları görselleştir
                if self.current_results:
                    frame = self.visualize_frame(frame, self.current_results)

                # Göster
                cv2.imshow("Yüz Analizi", frame)

                # 'q' tuşuna basılırsa çık
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop_capture()

    def stop_capture(self):
        """Kamera yakalamayı durdurur"""
        self.running = False

        # Kamera nesnesini kapat
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()

        # Tüm OpenCV pencerelerini kapat
        cv2.destroyAllWindows()

    def take_snapshot(self):
        """Mevcut kamera görüntüsünün anlık görüntüsünü alır ve analiz eder"""
        if self.camera is None or not self.camera.isOpened():
            print("Kamera aktif değil!")
            return None, None

        # Kare yakala
        ret, frame = self.camera.read()
        if not ret:
            print("Kare yakalama hatası!")
            return None, None

        # Analiz et
        analysis_result = self.analyze_frame(frame)
        if analysis_result:
            # Görselleştir
            viz_frame = self.visualize_frame(frame, analysis_result)
            return viz_frame, analysis_result

        return frame, None

    def save_snapshot(self, output_path="snapshot.jpg"):
        """Anlık görüntü alır, analiz eder ve kaydeder"""
        viz_frame, analysis_result = self.take_snapshot()

        if viz_frame is not None:
            # Dosyayı kaydet
            cv2.imwrite(output_path, viz_frame)
            print(f"Anlık görüntü kaydedildi: {output_path}")
            return output_path, analysis_result

        return None, None