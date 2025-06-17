import os
from datetime import datetime

import cv2
from audio_handler import AudioHandler
from face_analyzer import FaceAnalyzer
from body_analyzer import BodyAnalyzer
from capture_analyzer import CaptureAnalyzer


class FaceAnalyzeApp:
    """Ana uygulama sınıfı - Yüz analizi, vücut analizi ve ses işlemlerini birleştirir"""

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.body_analyzer = BodyAnalyzer()
        self.capture_analyzer = CaptureAnalyzer(face_analyzer=self.face_analyzer)
        self.audio_handler = AudioHandler()

        # Proje dizinini al
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Images klasörü
        self.images_dir = os.path.join(self.base_dir, 'images')
        # Eğer klasör yoksa oluştur
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        # Results klasörü
        self.results_dir = os.path.join(self.base_dir, 'results')
        # Eğer klasör yoksa oluştur
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # İletişim modu (True: sesli, False: yazılı)
        self.voice_mode = None

    def get_image_list(self):
        """Images klasöründeki görselleri listeler"""
        # Sadece resim dosyalarını kabul et
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []

        try:
            # Dizindeki tüm dosyaları kontrol et
            for filename in os.listdir(self.images_dir):
                # Dosya uzantısını kontrol et
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    image_files.append(filename)
        except Exception as e:
            print(f"Resim listeleme hatası: {e}")

        return image_files

    def display_image_menu(self):
        """Kullanıcıya resim seçim menüsü gösterir"""
        image_files = self.get_image_list()

        if not image_files:
            print("Images klasöründe hiç resim bulunamadı!")
            if self.voice_mode:
                self.audio_handler.speak_text("Images klasöründe hiç resim bulunamadı!")
            return None

        print("\n=== Mevcut Resimler ===")
        for i, filename in enumerate(image_files, 1):
            print(f"{i}. {filename}")

        # Kullanıcıdan seçim iste
        if self.voice_mode:
            self.audio_handler.speak_text("Lütfen analiz etmek istediğiniz resmin numarasını söyleyin")
        else:
            print("Resim numarası (veya çıkmak için 'q') girin:")

        while True:
            if self.voice_mode:
                selection = self.get_voice_input()
                # Sesli komuttan sayı elde etmeye çalış
                if selection:
                    if "çık" in selection or "iptal" in selection:
                        return None
                    # Sayıyı bul
                    for i in range(1, len(image_files) + 1):
                        if str(i) in selection:
                            selection = i
                            break
            else:
                selection = input("Seçiminiz: ")
                if selection.lower() == 'q':
                    return None

                try:
                    selection = int(selection)
                except ValueError:
                    print("Lütfen bir sayı girin.")
                    continue

            # Seçim kontrolü
            if isinstance(selection, int) and 1 <= selection <= len(image_files):
                selected_file = image_files[selection - 1]
                return os.path.join(self.images_dir, selected_file)
            else:
                print("Geçersiz seçim, lütfen tekrar deneyin.")
                if self.voice_mode:
                    self.audio_handler.speak_text("Geçersiz seçim, lütfen tekrar deneyin.")

    def get_voice_input(self):
        """Sesli komut al"""
        return self.audio_handler.listen_command()

    def get_text_input(self, prompt):
        """Metin girişi al"""
        return input(prompt)

    def get_user_input(self, prompt=""):
        """Kullanıcıdan giriş al (sesli veya yazılı moda göre)"""
        if self.voice_mode:
            if prompt:
                self.audio_handler.speak_text(prompt)
            return self.get_voice_input()
        else:
            return self.get_text_input(prompt)

    def select_interaction_mode(self):
        """Kullanıcıya sesli mi yazılı mı mod seçtirir"""
        print("\n=== YÜZ ANALİZ UYGULAMASI ===")
        print("1. Sesli mod (sesli komutlar)")
        print("2. Yazılı mod (klavye girişi)")

        while True:
            try:
                choice = input("Tercih ettiğiniz modu seçin (1/2): ")
                if choice == "1":
                    self.voice_mode = True
                    print("Sesli mod seçildi.")
                    self.audio_handler.speak_text("Sesli mod etkinleştirildi. Size nasıl yardımcı olabilirim?")
                    break
                elif choice == "2":
                    self.voice_mode = False
                    print("Yazılı mod seçildi.")
                    break
                else:
                    print("Geçersiz seçim. Lütfen 1 veya 2 girin.")
            except Exception as e:
                print(f"Hata oluştu: {e}")
                self.voice_mode = False
                break

    def run(self):
        """Ana uygulama döngüsünü çalıştırır"""
        print("Yüz analizi uygulaması başlatılıyor...")

        # İletişim modunu seç
        self.select_interaction_mode()

        while True:
            print("\nKomutlar: 'yüz analiz', 'vücut analiz', 'kamera', 'çıkış', 'yardım'")

            if self.voice_mode:
                self.audio_handler.speak_text("Bir komut söyleyin")
                command = self.get_voice_input()
                if not command:
                    self.audio_handler.speak_text("Sizi anlayamadım, lütfen tekrar deneyin.")
                    continue
                print(f"Algılanan komut: {command}")
            else:
                command = input("Komut girin: ")

            # Komut işleme
            if command and ('yüz' in command or 'face' in command or 'yuz' in command):
                self.process_face_analyze_command()
            elif command and ('vücut' in command or 'body' in command or 'vucut' in command):
                self.process_body_analyze_command()
            elif command and (
                    'kamera' in command or 'camera' in command or 'yakala' in command or 'capture' in command):
                self.process_camera_command()
            elif command and 'yardım' in command:
                self.show_help()
            elif command and ('çıkış' in command or 'kapat' in command):
                if self.voice_mode:
                    self.audio_handler.speak_text("Program kapatılıyor. Hoşçakalın.")
                print("Program kapatılıyor. Hoşçakalın.")
                break
            else:
                print("Anlaşılamadı. Yardım için 'yardım' yazın.")
                if self.voice_mode:
                    self.audio_handler.speak_text("Anlaşılamadı. Yardım için yardım diyebilirsiniz.")

    def process_face_analyze_command(self):
        """Yüz analizi komutunu işler"""
        while True:
            # Resim seçme menüsünü göster
            image_path = self.display_image_menu()

            if not image_path:
                # Eğer resim seçilmezse fonksiyondan çık ve ana menüye dön
                return

            if not os.path.exists(image_path):
                print("Belirtilen dosya bulunamadı.")
                if self.voice_mode:
                    self.audio_handler.speak_text("Belirtilen dosya bulunamadı.")
                continue  # Tekrar resim seçimi iste

            print("Resim analiz ediliyor, lütfen bekleyin...")
            if self.voice_mode:
                self.audio_handler.speak_text("Resim analiz ediliyor, lütfen bekleyin.")

            try:
                result, image, skin_mask, right_eye_mask, left_eye_mask = self.face_analyzer.analyze_face(image_path)

                if not result:
                    print("Resimde yüz tespit edilemedi!")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Resimde yüz tespit edilemedi!")
                    # Tekrar resim seçme menüsüne yönlendir
                    continue

                # Görselleştirme
                viz_image = self.face_analyzer.visualize_results(
                    image, skin_mask, right_eye_mask, left_eye_mask, result
                )

                # Sonuçları yazdır
                print("\n=== YÜZ ANALİZ SONUÇLARI ===")
                print(f"Ten Rengi: {result['ten_rengi']['tahmini_renk']}")
                print(f"RGB: {result['ten_rengi']['rgb']}")
                print(f"HEX: {result['ten_rengi']['hex']}")

                print(f"\nGöz Rengi: {result['goz_rengi']['tahmini_renk']}")
                print(f"RGB: {result['goz_rengi']['rgb']}")
                print(f"HEX: {result['goz_rengi']['hex']}")

                print(f"\nYüz Şekli: {result['yuz_sekli']['sekil']}")
                print(f"En/Boy Oranı: {result['yuz_sekli']['oran']}")

                # Sonuçları seslendir (eğer sesli modda ise)
                if self.voice_mode:
                    ten_rengi = result['ten_rengi']['tahmini_renk']
                    goz_rengi = result['goz_rengi']['tahmini_renk']
                    yuz_sekli = result['yuz_sekli']['sekil']

                    self.audio_handler.speak_text(
                        f"Analiz sonuçları: Ten rengi {ten_rengi}, Göz rengi {goz_rengi}, Yüz şekli {yuz_sekli}")

                # Görüntüyü kaydet - sonuç görsellerini results klasörüne kaydet
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(self.results_dir, f"face_analyzed_{output_filename}")
                cv2.imwrite(output_path, viz_image)
                print(f"\nAnaliz sonucu görüntü kaydedildi: {output_path}")

                # Görseli göster
                cv2.imshow("Yüz Analiz Sonucu", viz_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Analiz sonrası menü - sonuca göre devam et veya çık
                next_action = self.show_post_analysis_menu()

                if next_action == "continue":
                    # Döngüye devam et ve başka bir görsel analiz et
                    continue
                elif next_action == "return":
                    # Ana menüye dön
                    return
                elif next_action == "exit":
                    # Uygulamadan çık
                    print("\nUygulama kapatılıyor...")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Uygulama kapatılıyor. Hoşçakalın.")
                    exit(0)

                # Ek güvenlik kontrolü - varsayılan olarak ana menüye dön
                return

            except Exception as e:
                print(f"Hata: {e}")
                if self.voice_mode:
                    self.audio_handler.speak_text("Analiz sırasında bir hata oluştu.")
                # Hata durumunda ana menüye dönelim
                return

    def process_body_analyze_command(self):
        """Vücut analizi komutunu işler"""
        while True:
            # Resim seçme menüsünü göster
            image_path = self.display_image_menu()

            if not image_path:
                # Eğer resim seçilmezse fonksiyondan çık ve ana menüye dön
                return

            if not os.path.exists(image_path):
                print("Belirtilen dosya bulunamadı.")
                if self.voice_mode:
                    self.audio_handler.speak_text("Belirtilen dosya bulunamadı.")
                continue  # Tekrar resim seçimi iste

            print("Resim analiz ediliyor, lütfen bekleyin...")
            if self.voice_mode:
                self.audio_handler.speak_text("Vücut analizi yapılıyor, lütfen bekleyin.")

            try:
                result, image, pose_landmarks = self.body_analyzer.analyze_body(image_path)

                if not result:
                    print("Resimde vücut tespit edilemedi!")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Resimde vücut tespit edilemedi!")
                    # Tekrar resim seçme menüsüne yönlendir
                    continue

                # Görselleştirme
                viz_image = self.body_analyzer.visualize_results(
                    image, pose_landmarks, result
                )

                # Sonuçları yazdır
                print("\n=== VÜCUT ANALİZ SONUÇLARI ===")
                print(f"Vücut Tipi: {result['vucut_tipi']}")
                print(f"Açıklama: {result['aciklama']}")
                print(f"\nOranlar:")
                print(f"Omuz/Kalça: {result['oranlar']['omuz_kalca_orani']}")
                print(f"Bel/Kalça: {result['oranlar']['bel_kalca_orani']}")
                print(f"Boy/Genişlik: {result['oranlar']['boy_genislik_orani']}")

                # Sonuçları seslendir (eğer sesli modda ise)
                if self.voice_mode:
                    vucut_tipi = result['vucut_tipi']
                    aciklama = result['aciklama']

                    self.audio_handler.speak_text(f"Vücut analizi sonuçları: Vücut tipi {vucut_tipi}, {aciklama}")

                # Görüntüyü kaydet - sonuç görsellerini results klasörüne kaydet
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(self.results_dir, f"body_analyzed_{output_filename}")
                cv2.imwrite(output_path, viz_image)
                print(f"\nAnaliz sonucu görüntü kaydedildi: {output_path}")

                # Görseli göster
                cv2.imshow("Vücut Analiz Sonucu", viz_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Analiz sonrası menü - sonuca göre devam et veya çık
                next_action = self.show_post_analysis_menu()

                if next_action == "continue":
                    # Döngüye devam et ve başka bir görsel analiz et
                    continue
                elif next_action == "return":
                    # Ana menüye dön
                    return
                elif next_action == "exit":
                    # Uygulamadan çık
                    print("\nUygulama kapatılıyor...")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Uygulama kapatılıyor. Hoşçakalın.")
                    exit(0)

                # Ek güvenlik kontrolü - varsayılan olarak ana menüye dön
                return

            except Exception as e:
                print(f"Hata: {e}")
                if self.voice_mode:
                    self.audio_handler.speak_text("Analiz sırasında bir hata oluştu.")
                # Hata durumunda ana menüye dönelim
                return

    def process_camera_command(self):
        """Kamera ile yüz yakalama ve analiz komutunu işler"""
        print("\n=== KAMERA ANALİZ MODU ===")
        print("1. Canlı analiz başlat")
        print("2. Anlık görüntü al ve analiz et")
        print("3. Ana menüye dön")

        if self.voice_mode:
            self.audio_handler.speak_text(
                "Kamera modu seçiniz. Canlı analiz, anlık görüntü alma veya ana menüye dönme.")

        choice = None
        if self.voice_mode:
            voice_input = self.audio_handler.listen_command()
            if voice_input:
                if "canlı" in voice_input or "başlat" in voice_input or "bir" in voice_input:
                    choice = "1"
                elif "anlık" in voice_input or "görüntü" in voice_input or "foto" in voice_input or "iki" in voice_input:
                    choice = "2"
                elif "ana" in voice_input or "menü" in voice_input or "dön" in voice_input or "üç" in voice_input:
                    choice = "3"
        else:
            choice = input("Seçiminiz (1/2/3): ")

        if choice == "1":
            # Canlı analiz başlat
            print("Canlı yüz analizi başlatılıyor...")
            if self.voice_mode:
                self.audio_handler.speak_text("Canlı yüz analizi başlatılıyor. Çıkmak için Q tuşuna basın.")

            try:
                # Kamerayı başlat
                if not self.capture_analyzer.initialize_camera():
                    print("Kamera başlatılamadı!")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Kamera başlatılamadı.")
                    return

                self.capture_analyzer.running = True
                print("Canlı analiz aktif. Çıkmak için 'q' tuşuna basın.")

                last_analysis_time = datetime.now()

                while self.capture_analyzer.running:
                    # Kare yakala
                    ret, frame = self.capture_analyzer.camera.read()
                    if not ret:
                        print("Kamera görüntüsü alınamıyor!")
                        break

                    # Analiz zamanını kontrol et
                    current_time = datetime.now()
                    if current_time - last_analysis_time >= self.capture_analyzer.analysis_interval:
                        # Kareyi analiz et
                        analysis_result = self.capture_analyzer.analyze_frame(frame)
                        if analysis_result:
                            self.capture_analyzer.current_results = analysis_result
                        last_analysis_time = current_time

                    # Sonuçları görselleştir
                    if self.capture_analyzer.current_results:
                        frame = self.capture_analyzer.visualize_frame(frame, self.capture_analyzer.current_results)

                    # Pencere başlığını güncelle - kullanıcı bilgileri ile
                    window_title = f"Canlı Yüz Analizi - Kullanıcı: {self.capture_analyzer.current_user} - Çıkmak için 'q' tuşuna basın"
                    cv2.imshow(window_title, frame)

                    # Tuş kontrolü
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):  # 's' tuşu ile anlık görüntü alma
                        timestamp = datetime.date().strftime("%Y/%m/%d %H:%M:%S")
                        snapshot_path = os.path.join(self.results_dir, f"snapshot_{timestamp}.jpg")
                        cv2.imwrite(snapshot_path, frame)
                        print(f"Anlık görüntü kaydedildi: {snapshot_path}")
                        if self.voice_mode:
                            self.audio_handler.speak_text("Anlık görüntü kaydedildi.")

                # Temizle
                self.capture_analyzer.stop_capture()

            except Exception as e:
                print(f"Kamera analizi hatası: {e}")
                if self.voice_mode:
                    self.audio_handler.speak_text("Kamera başlatılırken bir hata oluştu.")
                if hasattr(self.capture_analyzer, "camera") and self.capture_analyzer.camera is not None:
                    self.capture_analyzer.stop_capture()

            # Ana menüye dön
            return

        elif choice == "2":
            # Anlık görüntü al
            print("Kamera hazırlanıyor...")
            if self.voice_mode:
                self.audio_handler.speak_text("Anlık görüntü almak için kamera hazırlanıyor.")

            try:
                # Kamerayı başlat
                if not self.capture_analyzer.initialize_camera():
                    print("Kamera başlatılamadı!")
                    if self.voice_mode:
                        self.audio_handler.speak_text("Kamera başlatılamadı.")
                    return

                print(
                    "Kamera hazır. Görüntü almak için 'space' veya 'enter' tuşuna basın. İptal etmek için 'q' tuşuna basın.")
                if self.voice_mode:
                    self.audio_handler.speak_text(
                        "Kamera hazır. Görüntü almak için boşluk veya enter tuşuna basın. İptal etmek için Q tuşuna basın.")

                capture_done = False
                countdown_active = False
                countdown_start = 0
                countdown_seconds = 3  # 3 saniyelik geri sayım

                while not capture_done:
                    # Kare yakala
                    ret, frame = self.capture_analyzer.camera.read()
                    if not ret:
                        break

                    display_frame = frame.copy()

                    # Eğer geri sayım aktifse
                    if countdown_active:
                        current_time = datetime.now()
                        elapsed = current_time - countdown_start
                        remaining = max(0, countdown_seconds - int(elapsed))

                        if remaining > 0:
                            # Geri sayım görüntüsü
                            cv2.putText(display_frame, f"{remaining}",
                                        (display_frame.shape[1] // 2 - 70, display_frame.shape[0] // 2 + 70),
                                        cv2.FONT_HERSHEY_DUPLEX, 6.0, (0, 0, 255), 10)
                            cv2.putText(display_frame, "Hazırlanın...",
                                        (display_frame.shape[1] // 2 - 150, display_frame.shape[0] // 2 - 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        else:
                            # Geri sayım bitti, fotoğraf çek
                            print("Görüntü alınıyor ve analiz ediliyor...")
                            if self.voice_mode:
                                self.audio_handler.speak_text("Görüntü alınıyor ve analiz ediliyor.")

                            # Anlık görüntü al ve analiz et
                            timestamp = datetime.date().strftime("%Y%m%d%H%M%S")
                            snapshot_path = os.path.join(self.results_dir, f"snapshot_{timestamp}.jpg")

                            # Görüntüyü kaydet
                            cv2.imwrite(snapshot_path, frame)

                            # Analiz et
                            analysis_result = self.capture_analyzer.analyze_frame(frame)

                            if analysis_result:
                                # Görselleştir
                                viz_frame = self.capture_analyzer.visualize_frame(frame, analysis_result)

                                # Analiz görselini kaydet
                                analyzed_path = os.path.join(self.results_dir, f"analyzed_snapshot_{timestamp}.jpg")
                                cv2.imwrite(analyzed_path, viz_frame)

                                print(f"Görüntü kaydedildi: {snapshot_path}")
                                print(f"Analiz edilmiş görüntü: {analyzed_path}")

                                if self.voice_mode:
                                    self.audio_handler.speak_text("Görüntü alındı ve analiz edildi.")

                                # Sonuçları göster
                                cv2.imshow("Analiz Sonucu", viz_frame)
                                cv2.waitKey(0)
                            else:
                                print("Görüntüde yüz tespit edilemedi!")
                                if self.voice_mode:
                                    self.audio_handler.speak_text("Görüntüde yüz tespit edilemedi.")

                            capture_done = True
                            break
                    else:
                        # Normal görüntüleme modu - yardım bilgisi ekle
                        cv2.putText(display_frame, "Space/Enter: Fotograf Cek | C: Geri Sayim | Q: Iptal",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Pencere başlığı
                    window_title = "Kamera Önizleme - Poz verebilirsiniz"
                    cv2.imshow(window_title, display_frame)

                    # Tuş kontrolü
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("İşlem iptal edildi.")
                        if self.voice_mode:
                            self.audio_handler.speak_text("İşlem iptal edildi.")
                        capture_done = True
                    elif key == ord(' ') or key == 13:  # Boşluk veya Enter tuşu - hemen çek
                        print("Görüntü alınıyor ve analiz ediliyor...")
                        if self.voice_mode:
                            self.audio_handler.speak_text("Görüntü alınıyor ve analiz ediliyor.")

                        timestamp = datetime.date().strftime("%Y%m%d%H%M%S")
                        snapshot_path = os.path.join(self.results_dir, f"snapshot_{timestamp}.jpg")

                        # Görüntüyü kaydet
                        cv2.imwrite(snapshot_path, frame)

                        # Analiz et
                        analysis_result = self.capture_analyzer.analyze_frame(frame)

                        if analysis_result:
                            viz_frame = self.capture_analyzer.visualize_frame(frame, analysis_result)
                            analyzed_path = os.path.join(self.results_dir, f"analyzed_snapshot_{timestamp}.jpg")
                            cv2.imwrite(analyzed_path, viz_frame)

                            print(f"Görüntü kaydedildi: {snapshot_path}")
                            print(f"Analiz edilmiş görüntü: {analyzed_path}")

                            if self.voice_mode:
                                self.audio_handler.speak_text("Görüntü alındı ve analiz edildi.")

                            cv2.imshow("Analiz Sonucu", viz_frame)
                            cv2.waitKey(0)
                        else:
                            print("Görüntüde yüz tespit edilemedi!")
                            if self.voice_mode:
                                self.audio_handler.speak_text("Görüntüde yüz tespit edilemedi.")

                        capture_done = True
                    elif key == ord('c'):  # 'c' tuşu - geri sayım başlat
                        print("Geri sayım başlıyor... Hazırlanın!")
                        if self.voice_mode:
                            self.audio_handler.speak_text("3 saniye içinde fotoğraf çekilecek. Hazırlanın!")
                        countdown_active = True
                        countdown_start = datetime.now()

                # Temizle
                cv2.destroyAllWindows()
                self.capture_analyzer.stop_capture()

            except Exception as e:
                print(f"Anlık görüntü hatası: {e}")
                if self.voice_mode:
                    self.audio_handler.speak_text("Anlık görüntü alırken bir hata oluştu.")
                cv2.destroyAllWindows()
                if hasattr(self.capture_analyzer, "camera") and self.capture_analyzer.camera is not None:
                    self.capture_analyzer.stop_capture()

            # Ana menüye dön
            return

        else:  # choice == "3" veya geçersiz
            # Ana menüye dön
            print("Ana menüye dönülüyor...")
            if self.voice_mode:
                self.audio_handler.speak_text("Ana menüye dönülüyor.")
            return

    def show_post_analysis_menu(self):
        """
        Analiz sonrası seçenekleri gösterir
        Return: "continue" (analize devam et), "return" (ana menüye dön) veya "exit" (çık)
        """
        print("\n=== Analiz tamamlandı ===")
        print("1. Başka bir görsel analiz et")
        print("2. Ana menüye dön")
        print("3. Uygulamadan çık")

        if self.voice_mode:
            self.audio_handler.speak_text(
                "Analiz tamamlandı. Başka bir görsel analiz etmek için bir, ana menüye dönmek için iki, çıkmak için üç diyebilirsiniz.")

        # Maksimum 3 deneme hakkı
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1
            choice = None

            if self.voice_mode:
                print("Seçiminizi söyleyin...")
                voice_input = self.audio_handler.listen_command()
                print(f"Algılanan komut: {voice_input}")

                if voice_input:
                    voice_input = voice_input.lower()
                    if "bir" in voice_input or "1" in voice_input or "başka" in voice_input or "yeni" in voice_input or "analiz" in voice_input:
                        choice = "1"
                    elif "iki" in voice_input or "2" in voice_input or "ana" in voice_input or "menü" in voice_input or "geri" in voice_input:
                        choice = "2"
                    elif "üç" in voice_input or "3" in voice_input or "çık" in voice_input or "kapat" in voice_input:
                        choice = "3"
            else:
                choice = input("Seçiminiz (1/2/3): ")

            if choice == "1":
                print("Başka bir görsel analiz ediliyor...")
                if self.voice_mode:
                    self.audio_handler.speak_text("Başka bir görsel analiz ediliyor.")
                return "continue"  # Analize devam et

            elif choice == "2":
                print("\nAna menüye dönülüyor...")
                if self.voice_mode:
                    self.audio_handler.speak_text("Ana menüye dönülüyor.")
                return "return"  # Ana menüye dön

            elif choice == "3":
                print("\nUygulama kapatılıyor...")
                if self.voice_mode:
                    self.audio_handler.speak_text("Uygulama kapatılıyor. Hoşçakalın.")
                return "exit"  # Uygulamadan çık

            else:
                print(f"Geçersiz seçim, lütfen 1, 2 veya 3 girin. (Deneme {attempts}/{max_attempts})")
                if self.voice_mode:
                    self.audio_handler.speak_text("Geçersiz seçim, lütfen tekrar deneyin.")

        # Maksimum deneme sayısı aşıldı, varsayılan olarak ana menüye dön
        print("\nÇok fazla geçersiz giriş. Ana menüye dönülüyor...")
        if self.voice_mode:
            self.audio_handler.speak_text("Çok fazla geçersiz giriş. Ana menüye dönülüyor.")
        return "return"

    def show_help(self):
        """Yardım bilgisini gösterir ve seslendirir"""
        help_text = """
        === YÜZ VE VÜCUT ANALİZ UYGULAMASI YARDIMI ===

        Kullanılabilir Komutlar:
        - yüz analiz: Images klasöründen bir resim seçip yüz analizi yapmanızı sağlar
                    (ten rengi, göz rengi ve yüz şekli tespiti)
        - vücut analiz: Images klasöründen bir resim seçip vücut analizi yapmanızı sağlar
                      (vücut tipi ve oran tespiti)
        - kamera: Kamera ile canlı analiz yapabilir veya anlık görüntü alabilirsiniz
        - yardım: Bu yardım mesajını gösterir
        - çıkış: Programdan çıkar

        Not: Analiz edilecek resimleri 'images' klasörüne koymanız gerekmektedir.
        Analiz sonuçları 'results' klasörüne kaydedilecektir.
        """
        print(help_text)
        if self.voice_mode:
            self.audio_handler.speak_text(
                "Yüz analiz, vücut analiz, kamera kullanımı, yardım ve çıkış komutlarını kullanabilirsiniz. "
                "Analiz edilecek resimleri images klasörüne koyun, sonuçlar results klasörüne kaydedilecektir.")


if __name__ == "__main__":
    app = FaceAnalyzeApp()
    app.run()