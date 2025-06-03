import cv2
import numpy as np
from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker # Pastikan path ini benar
from collections import deque
import time
from typing import Tuple, Dict, List, Union
import statistics
import json
import base64
import pika
import threading
import logging
import queue
import os
import torch
import psutil
import pynvml

# Impor modul settings
import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize NVIDIA GPU monitoring
try:
    pynvml.nvmlInit()
    gpu_available = True
except pynvml.NVMLError:
    gpu_available = False
    logging.warning("NVIDIA GPU not available or pynvml not initialized")

# =================== RTSPStreamManager Class ===================
class RTSPStreamManager:
    """Enhanced RTSP stream manager with optimized buffering"""

    def __init__(self, rtsp_url, buffer_size_cv2, queue_buffer_size):
        self.rtsp_url = rtsp_url
        # Menggunakan settings.MAX_QUEUE_SIZE untuk frame_queue internal
        self.frame_queue = queue.Queue(maxsize=queue_buffer_size)
        self.running = False
        self.capture_thread = None
        self.reconnect_delay = 1
        self.cap = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps_stats = []
        self.buffer_stats = []
        self.cv2_buffer_size = buffer_size_cv2 # Buffer size untuk OpenCV

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"
        )

    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logging.info("RTSP stream capture started")

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logging.info("RTSP stream capture stopped")

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_frames(self):
        while self.running:
            try:
                if not self._ensure_connection():
                    time.sleep(self.reconnect_delay)
                    continue

                buffer_fullness = self.frame_queue.qsize() / self.frame_queue.maxsize if self.frame_queue.maxsize > 0 else 0
                self.buffer_stats.append(buffer_fullness)
                if len(self.buffer_stats) > 100:
                    self.buffer_stats.pop(0)

                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to read frame, attempting to reconnect...")
                    self._close_connection()
                    time.sleep(self.reconnect_delay)
                    continue

                self.frame_count += 1
                current_time = time.time()
                if self.last_frame_time > 0:
                    time_delta = current_time - self.last_frame_time
                    if time_delta > 0:
                        actual_fps = 1 / time_delta
                        self.fps_stats.append(actual_fps)
                        if len(self.fps_stats) > 100:
                            self.fps_stats.pop(0)
                self.last_frame_time = current_time

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        logging.debug("RTSP frame queue was full, dropped oldest frame.")
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    logging.debug("RTSP frame queue is full on put, frame dropped.")

                # Menggunakan settings.VIDEO_FPS untuk logging periodik
                if self.frame_count % (settings.VIDEO_FPS * 10) == 0:
                    avg_fps = statistics.mean(self.fps_stats) if self.fps_stats else 0
                    avg_buffer = statistics.mean(self.buffer_stats) if self.buffer_stats else 0
                    logging.info(
                        f"RTSP Stream: frames_captured={self.frame_count}, avg_capture_fps={avg_fps:.1f}, avg_buffer_fill={avg_buffer:.2f}"
                    )

            except Exception as e:
                logging.error(f"Error in RTSP frame capture loop: {e}", exc_info=True)
                self._close_connection()
                time.sleep(self.reconnect_delay * 2)

    def _ensure_connection(self):
        if self.cap is None or not self.cap.isOpened():
            try:
                logging.info(f"Attempting to connect to RTSP stream: {self.rtsp_url}")
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                # Menggunakan self.cv2_buffer_size yang diinisialisasi dari settings.CV2_BUFFER_SIZE
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cv2_buffer_size)
                # self.cap.set(cv2.CAP_PROP_FPS, settings.VIDEO_FPS) # Setting FPS di sini memiliki efek terbatas

                if not self.cap.isOpened():
                    logging.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                    self.cap = None
                    return False
                logging.info(f"RTSP connection established: {self.rtsp_url}")
                return True
            except Exception as e:
                logging.error(f"Exception while connecting to RTSP stream: {e}")
                self.cap = None
                return False
        return True

    def _close_connection(self):
        if self.cap:
            logging.info("Closing RTSP stream connection.")
            self.cap.release()
            self.cap = None

    def get_stats(self):
        return {
            "captured_frames": self.frame_count,
            "capture_fps_avg": statistics.mean(self.fps_stats) if self.fps_stats else 0,
            "buffer_fill_avg": statistics.mean(self.buffer_stats) if self.buffer_stats else 0,
            "queue_size": self.frame_queue.qsize()
        }

# =================== Monitoring Function ===================
def monitor_resources(stop_event: threading.Event, inference_times: list, frames_processed: list, start_time_monitor: float):
    resource_data = {
        "cpu_percent": [], "gpu_util": [], "inference_times": inference_times, "fps": []
    }
    # Menggunakan settings.YOLO_MODEL_PATH untuk nama model di log
    model_name_log = os.path.basename(settings.YOLO_MODEL_PATH)
    gpu_device = 0

    # Menggunakan settings.RESOURCE_MONITORING_LOG_FILE
    with open(settings.RESOURCE_MONITORING_LOG_FILE, "w") as f:
        f.write("Model,Average GPU %,Average CPU %,Average Inference Time ms,Average Processing FPS\n")

    last_frame_count_monitor = 0
    last_log_time_monitor = start_time_monitor

    while not stop_event.is_set():
        current_log_time_monitor = time.time()

        cpu_percent = psutil.cpu_percent(interval=None)
        resource_data["cpu_percent"].append(cpu_percent)

        gpu_util = 0
        # Menggunakan settings.USE_CUDA
        if gpu_available and settings.USE_CUDA:
            try:
                device = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
                util = pynvml.nvmlDeviceGetUtilizationRates(device)
                gpu_util = util.gpu
                resource_data["gpu_util"].append(gpu_util)
            except pynvml.NVMLError as e:
                logging.debug(f"Debug: Error getting GPU info: {e}")
                resource_data["gpu_util"].append(0)

        current_total_frames = frames_processed[0]
        elapsed_interval_monitor = current_log_time_monitor - last_log_time_monitor

        # Menggunakan settings.MONITORING_INTERVAL_SECONDS
        if elapsed_interval_monitor >= settings.MONITORING_INTERVAL_SECONDS:
            frames_in_interval = current_total_frames - last_frame_count_monitor
            interval_fps = frames_in_interval / elapsed_interval_monitor if elapsed_interval_monitor > 0 else 0
            resource_data["fps"].append(interval_fps)

            if current_log_time_monitor - start_time_monitor >= 10.0 or stop_event.is_set():
                avg_cpu = statistics.mean(resource_data["cpu_percent"]) if resource_data["cpu_percent"] else 0
                avg_gpu = statistics.mean(resource_data["gpu_util"]) if resource_data["gpu_util"] else 0
                avg_inf_time_ms = (statistics.mean(resource_data["inference_times"]) * 1000) if resource_data["inference_times"] else 0
                avg_processing_fps = statistics.mean(resource_data["fps"]) if resource_data["fps"] else 0

                # Menggunakan settings.RESOURCE_MONITORING_LOG_FILE
                with open(settings.RESOURCE_MONITORING_LOG_FILE, "a") as f_log:
                    f_log.write(f"{model_name_log},{avg_gpu:.2f},{avg_cpu:.2f},{avg_inf_time_ms:.2f},{avg_processing_fps:.2f}\n")
                    f_log.flush()

                logging.info(
                    f"Avg Resources (10s) - GPU: {avg_gpu:.1f}%, CPU: {avg_cpu:.1f}%, Infer: {avg_inf_time_ms:.1f}ms, Proc. FPS: {avg_processing_fps:.1f}"
                )
                resource_data["cpu_percent"].clear()
                resource_data["gpu_util"].clear()
                resource_data["fps"].clear()
                start_time_monitor = current_log_time_monitor

            last_frame_count_monitor = current_total_frames
            last_log_time_monitor = current_log_time_monitor

        time.sleep(min(0.5, settings.MONITORING_INTERVAL_SECONDS / 2.0))


# =================== SpeedEstimator Class ===================
class SpeedEstimator:
    def __init__(
        self, model, debug, real_width, real_height,
        pixels_per_meter, video_fps, allowed_direction,
        max_speed_threshold, use_rabbitmq, pts1,
        outlier_threshold, tracker_args_dict, calibration_rules_list,
        min_detection_confidence, consistent_speed_buffer_seconds, min_consistent_speed_frames,
        wa_recipient, speed_violators_dir, counterflow_violators_dir, jpeg_quality,
        violation_image_bbox_expansion_factor
    ):
        self.model = model
        self.debug = debug
        self.real_width = real_width
        self.real_height = real_height
        self.pixels_per_meter = pixels_per_meter
        self.output_width = int(real_width * pixels_per_meter)
        self.output_height = int(real_height * pixels_per_meter)
        self.video_fps = video_fps if video_fps > 0 else 1.0
        self.time_per_frame = 1.0 / self.video_fps

        self.allowed_direction = allowed_direction
        self.max_speed_threshold = max_speed_threshold
        self.outlier_threshold = outlier_threshold # Menggunakan dari settings
        
        # Menggunakan settings.TrackerArgsFromSettings
        args = settings.TrackerArgsFromSettings()
        self.tracker = BYTETracker(args, frame_rate=self.video_fps)

        self.track_history = {}
        self.track_timestamps = {}
        self.track_speeds = {}
        self.speed_histories = {}
        self.centroid_history = {}
        self.kalman_filters = {}
        
        self.max_history_length = int(self.video_fps * 2)
        self.smoothing_window = 5
        self.speed_history_window = int(self.video_fps / 2)

        self.min_detection_confidence = min_detection_confidence # Menggunakan dari settings

        # Menggunakan settings.CONSISTENT_SPEED_BUFFER_SECONDS dan settings.MIN_CONSISTENT_SPEED_FRAMES
        self.consistent_speed_buffer_len = int(video_fps * consistent_speed_buffer_seconds)
        if self.consistent_speed_buffer_len < min_consistent_speed_frames:
            self.consistent_speed_buffer_len = min_consistent_speed_frames
        self.track_recent_speeds_for_violation = {}
        self.speed_violators = set()

        self.counterflow_violators = {}
        self.frame_count_internal = 0

        self.pts1 = np.array(pts1, dtype="float32")
        self.pts2 = np.array([[0,0],[self.output_width,0],
                              [self.output_width,self.output_height],[0,self.output_height]], dtype="float32")
        self.matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

        self.calibration_rules = calibration_rules_list # Menggunakan dari settings

        self.use_rabbitmq = use_rabbitmq
        self.wa_recipient = wa_recipient # Menggunakan dari settings
        self.jpeg_quality = jpeg_quality # Menggunakan dari settings
        self.violation_image_bbox_expansion_factor = violation_image_bbox_expansion_factor # Menggunakan dari settings

        if self.use_rabbitmq:
            self._init_rabbitmq()

        # Menggunakan settings.SPEED_VIOLATORS_DIR dan settings.COUNTERFLOW_VIOLATORS_DIR
        os.makedirs(speed_violators_dir, exist_ok=True)
        os.makedirs(counterflow_violators_dir, exist_ok=True)
        self.speed_violators_dir = speed_violators_dir
        self.counterflow_violators_dir = counterflow_violators_dir
        logging.info(f"SpeedEstimator: Consistent speed violation check uses buffer of {self.consistent_speed_buffer_len} frames.")


    def _init_rabbitmq(self):
        try:
            # Menggunakan variabel RabbitMQ dari settings
            credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS)
            parameters = pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=settings.RABBITMQ_PORT,
                virtual_host=settings.RABBITMQ_VHOST,
                credentials=credentials,
                heartbeat=settings.RABBITMQ_HEARTBEAT,
                blocked_connection_timeout=settings.RABBITMQ_BLOCKED_CONNECTION_TIMEOUT
            )
            self.rabbitmq_connection = pika.BlockingConnection(parameters)
            self.rabbitmq_channel = self.rabbitmq_connection.channel()
            self.rabbitmq_channel.queue_declare(queue=settings.RABBITMQ_QUEUE_NAME, durable=True)
            logging.info("RabbitMQ initialized successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize RabbitMQ: {e}")
            self.use_rabbitmq = False
            return False

    def initialize_kalman_filter(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,self.time_per_frame,0],
                                         [0,1,0,self.time_per_frame],
                                         [0,0,1,0],
                                         [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        return kf

    def smooth_centroid(self, track_id: int, current_centroid: Tuple[int, int]) -> Tuple[int, int]:
        if track_id not in self.kalman_filters:
            kf = self.initialize_kalman_filter()
            kf.statePost = np.array([current_centroid[0], current_centroid[1], 0, 0], dtype=np.float32)
            self.kalman_filters[track_id] = kf
            self.centroid_history[track_id] = deque(maxlen=self.smoothing_window)
        
        kf = self.kalman_filters[track_id]
        predicted_state = kf.predict()
        measurement = np.array(current_centroid, dtype=np.float32).reshape(2,1)
        corrected_state = kf.correct(measurement)
        smoothed_centroid = (int(corrected_state[0]), int(corrected_state[1]))
        self.centroid_history[track_id].append(smoothed_centroid)
        return smoothed_centroid

    def transform_point(self, point: Tuple[int, int]) -> Union[Tuple[float, float], None]:
        pt_np = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_np, self.matrix)
        if transformed is not None:
            return (float(transformed[0][0][0]), float(transformed[0][0][1]))
        return None

    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return 0 <= x < self.output_width and 0 <= y < self.output_height

    def get_color(self, idx: int) -> Tuple[int, int, int]:
        colors = [
            (37,255,225),(0,255,0),(0,55,255),(255,0,0),
            (255,127,0),(255,255,0),(255,0,255),(128,0,128)
        ]
        return colors[idx % len(colors)]

    def apply_calibration(self, speed: float) -> float:
        for min_s, max_s, factor in self.calibration_rules:
            if min_s <= speed < max_s:
                return speed * factor
        if speed >= self.calibration_rules[-1][1]:
             return speed * self.calibration_rules[-1][2]
        return speed

    def get_robust_speed_stat(self, values: List[float]) -> float:
        if not values: return 0.0
        if len(values) < 3: return statistics.mean(values) if values else 0.0
        sorted_values = sorted(values)
        trim_count = int(len(sorted_values) * 0.1)
        trimmed_values = sorted_values[trim_count : len(sorted_values) - trim_count]
        if not trimmed_values: return statistics.median(sorted_values)
        return statistics.median(trimmed_values)

    def calculate_speed(self, track_id: int, current_smoothed_centroid: Tuple[int, int]) -> float:
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0
        try:
            prev_smoothed_centroid = self.track_history[track_id][-2]
            prev_pos_trans = self.transform_point(prev_smoothed_centroid)
            curr_pos_trans = self.transform_point(current_smoothed_centroid)

            if prev_pos_trans is None or curr_pos_trans is None or \
               not self.is_point_in_roi(curr_pos_trans) or not self.is_point_in_roi(prev_pos_trans):
                return 0.0

            dx = curr_pos_trans[0] - prev_pos_trans[0]
            dy = curr_pos_trans[1] - prev_pos_trans[1]
            distance_meters = np.sqrt(dx**2 + dy**2) / self.pixels_per_meter

            if self.time_per_frame <= 0: return 0.0
            current_raw_speed_kmh = (distance_meters / self.time_per_frame) * 3.6

            if track_id not in self.speed_histories:
                self.speed_histories[track_id] = deque(maxlen=self.speed_history_window)
            self.speed_histories[track_id].append(current_raw_speed_kmh)

            if len(self.speed_histories[track_id]) >= 3:
                stable_raw_speed_kmh = self.get_robust_speed_stat(list(self.speed_histories[track_id]))
            else:
                stable_raw_speed_kmh = current_raw_speed_kmh
            stable_raw_speed_kmh = min(stable_raw_speed_kmh, self.outlier_threshold)

            alpha = 0.4
            if track_id not in self.track_speeds or not self.track_speeds[track_id]:
                self.track_speeds[track_id] = stable_raw_speed_kmh
            else:
                self.track_speeds[track_id] = alpha * stable_raw_speed_kmh + \
                                              (1 - alpha) * self.track_speeds[track_id]
            final_smoothed_speed = self.track_speeds[track_id]
            calibrated_speed = self.apply_calibration(final_smoothed_speed)
            if self.debug:
                logging.debug(f"ID {track_id}: Raw Speed={current_raw_speed_kmh:.1f}, "
                              f"StableRaw={stable_raw_speed_kmh:.1f}, EMA={final_smoothed_speed:.1f}, "
                              f"Calibrated={calibrated_speed:.1f} km/h")
            return calibrated_speed
        except Exception as e:
            if self.debug: logging.error(f"Error in calculate_speed for ID {track_id}: {e}", exc_info=True)
            return 0.0

    def expand_bbox(self, x1, y1, x2, y2, frame_width, frame_height, expansion_factor):
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return x1,y1,x2,y2
        center_x, center_y = x1 + w / 2, y1 + h / 2
        new_w, new_h = w * expansion_factor, h * expansion_factor
        new_x1 = max(0, center_x - new_w / 2)
        new_y1 = max(0, center_y - new_h / 2)
        new_x2 = min(frame_width, center_x + new_w / 2)
        new_y2 = min(frame_height, center_y + new_h / 2)
        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

    def publish_rabbitmq_message(self, message: dict):
        if not self.use_rabbitmq:
            logging.info(f"RabbitMQ disabled. Dummy message: {json.dumps(message)}")
            return
        def publish_task():
            try:
                if not self.rabbitmq_connection or self.rabbitmq_connection.is_closed:
                    logging.warning("RabbitMQ connection lost. Attempting to reconnect...")
                    if not self._init_rabbitmq():
                        logging.error("RabbitMQ reconnection failed. Message not sent.")
                        return
                self.rabbitmq_channel.basic_publish(
                    exchange="", routing_key=settings.RABBITMQ_QUEUE_NAME,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
                )
                if self.debug: logging.debug(f"Published to RabbitMQ: {message.get('text', '')[:50]}...")
            except pika.exceptions.AMQPConnectionError as e:
                logging.error(f"RabbitMQ AMQP Connection Error: {e}. Message may not have been sent.")
            except Exception as e:
                logging.error(f"Failed to publish to RabbitMQ (Thread): {e}", exc_info=True)
        threading.Thread(target=publish_task, daemon=True).start()

    def image_to_base64_with_mime(self, image_array):
        try:
            # Menggunakan self.jpeg_quality dari settings
            success, buffer = cv2.imencode(".jpg", image_array, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not success:
                logging.error("Failed to encode image to JPEG for base64.")
                return None
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logging.error(f"Error converting image to base64: {e}")
            return None

    def capture_and_send_violation(self, frame, tid, violation_type, speed_value=None, detection_bbox=None):
        if detection_bbox is None:
            logging.warning(f"Violation for ID {tid} ({violation_type}) - Bbox not provided. Cannot capture image.")
            return

        x1_orig, y1_orig, x2_orig, y2_orig = detection_bbox
        # Menggunakan self.violation_image_bbox_expansion_factor dari settings
        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = self.expand_bbox(
            x1_orig, y1_orig, x2_orig, y2_orig, frame.shape[1], frame.shape[0],
            expansion_factor=self.violation_image_bbox_expansion_factor
        )

        if not (expanded_x2 > expanded_x1 and expanded_y2 > expanded_y1):
            logging.warning(f"Invalid expanded bbox for violation ID {tid}. Original: {detection_bbox}")
            return
        try:
            cropped_image = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            if cropped_image.size == 0:
                logging.warning(f"Violation ID {tid}: Cropped image is empty.")
                return

            violator_dir = self.speed_violators_dir if violation_type == "speed" else self.counterflow_violators_dir
            filename = f"{violator_dir}/{violation_type}_ID{tid}_F{self.frame_count_internal}.jpg"
            cv2.imwrite(filename, cropped_image)
            
            image_b64_mime = self.image_to_base64_with_mime(cropped_image)
            if image_b64_mime is None: return

            text_message = f"Pelanggaran {violation_type.capitalize()}! ID Kendaraan: {tid}"
            if speed_value is not None:
                text_message += f", Kecepatan: {speed_value:.1f} km/jam"
            
            payload = {
                "wa_number": self.wa_recipient, # Menggunakan self.wa_recipient
                "text": text_message,
                "image": image_b64_mime,
            }
            self.publish_rabbitmq_message(payload)
            logging.info(f"Violation message sent for ID {tid} ({violation_type}). Speed: {speed_value if speed_value else 'N/A'}")
        except Exception as e:
            logging.error(f"Error in capture_and_send_violation for ID {tid}: {e}", exc_info=True)

    def process_frame(self, frame: np.ndarray, inference_times_list: list) -> np.ndarray:
        self.frame_count_internal += 1
        frame_height, frame_width = frame.shape[:2]
        processed_frame_bgr = frame.copy()

        start_inference = time.time()
        # Menggunakan settings.USE_CUDA
        device_setting = 0 if settings.USE_CUDA and torch.cuda.is_available() else "cpu"
        # Menggunakan settings.YOLO_CLASSES_TO_DETECT dan self.min_detection_confidence
        results = self.model(
            frame, stream=True, device=device_setting, half=False, verbose=False,
            imgsz=640, # Mungkin ingin dipindahkan ke settings
            classes=settings.YOLO_CLASSES_TO_DETECT if settings.YOLO_CLASSES_TO_DETECT else None,
            conf=self.min_detection_confidence
        )
        inference_time = time.time() - start_inference
        inference_times_list.append(inference_time)

        detections_for_tracker = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections_for_tracker.append([x1, y1, x2, y2, conf])
        
        detections_np = np.array(detections_for_tracker) if detections_for_tracker else np.empty((0,5))
        online_targets = self.tracker.update(detections_np, [frame_height, frame_width], [frame_height, frame_width])
        active_track_ids = set()

        for t in online_targets:
            tlwh = t.tlwh; tid = t.track_id; active_track_ids.add(tid)
            x1_trk, y1_trk, w_trk, h_trk = map(int, tlwh)
            if w_trk <= 0 or h_trk <= 0: continue
            x2_trk, y2_trk = x1_trk + w_trk, y1_trk + h_trk
            current_detection_bbox = [x1_trk, y1_trk, x2_trk, y2_trk]
            current_raw_centroid = (int(x1_trk + w_trk/2), int(y1_trk + h_trk/2))
            smoothed_centroid = self.smooth_centroid(tid, current_raw_centroid)
            transformed_centroid = self.transform_point(smoothed_centroid)
            if transformed_centroid is None or not self.is_point_in_roi(transformed_centroid):
                continue

            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.max_history_length)
                self.track_timestamps[tid] = deque(maxlen=self.max_history_length)
            self.track_history[tid].append(smoothed_centroid)
            self.track_timestamps[tid].append(self.frame_count_internal * self.time_per_frame)

            final_speed_kmh = self.calculate_speed(tid, smoothed_centroid)
            if tid not in self.track_recent_speeds_for_violation:
                self.track_recent_speeds_for_violation[tid] = deque(maxlen=self.consistent_speed_buffer_len)
            self.track_recent_speeds_for_violation[tid].append(final_speed_kmh)

            is_consistently_speeding = False
            recent_speeds_buffer = self.track_recent_speeds_for_violation[tid]
            if len(recent_speeds_buffer) == self.consistent_speed_buffer_len:
                if all(s > self.max_speed_threshold for s in recent_speeds_buffer):
                    is_consistently_speeding = True
            
            color = self.get_color(tid)
            if is_consistently_speeding:
                color = (0, 0, 255)
                if tid not in self.speed_violators:
                    reported_speed = statistics.mean(recent_speeds_buffer)
                    self.capture_and_send_violation(processed_frame_bgr, tid, "speed", reported_speed, current_detection_bbox)
                    self.speed_violators.add(tid)
                    if self.debug: logging.info(f"CONSISTENT Speed Violation: ID {tid}, Speed: {reported_speed:.1f} km/h")
            
            is_counterflow = False
            if len(self.track_history[tid]) >= 5:
                last_positions = list(self.track_history[tid])[-5:]
                transformed_positions = [self.transform_point(p) for p in last_positions if p is not None]
                valid_transformed_positions = [p for p in transformed_positions if p is not None and self.is_point_in_roi(p)]
                if len(valid_transformed_positions) >= 2:
                    delta_ys = [valid_transformed_positions[i][1] - valid_transformed_positions[i-1][1] for i in range(1, len(valid_transformed_positions))]
                    delta_xs = [valid_transformed_positions[i][0] - valid_transformed_positions[i-1][0] for i in range(1, len(valid_transformed_positions))]
                    if delta_ys and delta_xs:
                        avg_delta_y = sum(delta_ys) / len(delta_ys)
                        avg_delta_x = sum(delta_xs) / len(delta_xs)
                        direction_displacement_threshold = 5.0
                        axis_dominance_ratio = 2.5
                        if self.allowed_direction == "top_to_bottom":
                            if avg_delta_y < -direction_displacement_threshold and abs(avg_delta_y) > abs(avg_delta_x) * axis_dominance_ratio:
                                is_counterflow = True
                        elif self.allowed_direction == "bottom_to_top":
                            if avg_delta_y > direction_displacement_threshold and abs(avg_delta_y) > abs(avg_delta_x) * axis_dominance_ratio:
                                is_counterflow = True
                        elif self.allowed_direction == "left_to_right":
                            if avg_delta_x < -direction_displacement_threshold and abs(avg_delta_x) > abs(avg_delta_y) * axis_dominance_ratio:
                                is_counterflow = True
                        elif self.allowed_direction == "right_to_left":
                            if avg_delta_x > direction_displacement_threshold and abs(avg_delta_x) > abs(avg_delta_y) * axis_dominance_ratio:
                                is_counterflow = True
            
            if is_counterflow:
                color = (0, 100, 255)
                if tid not in self.counterflow_violators:
                    self.capture_and_send_violation(processed_frame_bgr, tid, "counterflow", final_speed_kmh, current_detection_bbox)
                    self.counterflow_violators[tid] = self.frame_count_internal
                    if self.debug: logging.info(f"Counterflow Violation: ID {tid}")
                cv2.putText(processed_frame_bgr, "Melawan Arus!", (x1_trk, y1_trk - 30 if y1_trk > 30 else y2_trk + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.rectangle(processed_frame_bgr, (x1_trk,y1_trk),(x2_trk,y2_trk), color, 2)
            cv2.circle(processed_frame_bgr, smoothed_centroid, 4, color, -1)
            speed_text = f"ID:{tid} {final_speed_kmh:.0f}km/h"
            (w_text, h_text), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = x1_trk, y1_trk - 7 if y1_trk > h_text + 7 else y1_trk + h_trk + h_text + 3
            cv2.rectangle(processed_frame_bgr, (text_x - 2, text_y - h_text - 2), (text_x + w_text + 2, text_y + 2), color, -1)
            cv2.putText(processed_frame_bgr, speed_text, (text_x, text_y -2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        if self.frame_count_internal % (self.video_fps * 10) == 0:
            self._cleanup_inactive_tracks(active_track_ids)
        return processed_frame_bgr

    def _cleanup_inactive_tracks(self, active_track_ids: set):
        keys_to_delete = [k for k in self.track_history if k not in active_track_ids]
        for key in keys_to_delete:
            if self.debug: logging.debug(f"Cleaning up inactive track ID: {key}")
            self.track_history.pop(key, None)
            self.track_timestamps.pop(key, None)
            self.track_speeds.pop(key, None)
            self.speed_histories.pop(key, None)
            self.centroid_history.pop(key, None)
            self.kalman_filters.pop(key, None)
            self.track_recent_speeds_for_violation.pop(key, None)
            self.speed_violators.discard(key)
            self.counterflow_violators.pop(key, None)

    def save_transformed_frame(self, frame: np.ndarray, output_path: str):
        if self.matrix is None or self.matrix.size == 0:
            logging.warning("Perspective matrix not initialized. Cannot save transformed frame.")
            return
        if self.output_width <= 0 or self.output_height <= 0:
            logging.warning(f"Invalid output dimensions for transform: {self.output_width}x{self.output_height}")
            return
        try:
            transformed = cv2.warpPerspective(frame, self.matrix, (self.output_width, self.output_height))
            cv2.imwrite(output_path, transformed)
            if self.debug: logging.info(f"Transformed frame saved to {output_path}")
        except cv2.error as e: logging.error(f"OpenCV error saving transformed frame: {e}")
        except Exception as e: logging.error(f"Generic error saving transformed frame: {e}")

# FrameProcessor class
class FrameProcessor:
    def __init__(self, estimator, num_threads, queue_size):
        self.estimator = estimator
        self.num_threads = num_threads
        # Menggunakan queue_size dari settings.MAX_QUEUE_SIZE
        self.processing_queue = queue.Queue(maxsize=queue_size)
        self.results_queue = queue.Queue(maxsize=queue_size)
        self.threads = []
        self.running = False
        self.inference_times_fp = []

    def start(self):
        self.running = True
        for i in range(self.num_threads):
            thread = threading.Thread(target=self._process_frames_task, args=(i, self.inference_times_fp))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        logging.info(f"FrameProcessor started {self.num_threads} processing threads.")

    def stop(self):
        self.running = False
        for _ in range(self.num_threads):
            try: self.processing_queue.put(None, timeout=0.1)
            except queue.Full: pass
        for thread in self.threads:
            thread.join(timeout=2.0)
        logging.info("FrameProcessor threads stopped.")

    def add_frame(self, frame_tuple):
        if not self.running: return False
        try:
            self.processing_queue.put(frame_tuple, block=False)
            return True
        except queue.Full:
            logging.warning("FrameProcessor input queue FULL. Frame dropped.")
            return False

    def get_processed_output(self, timeout=0.1):
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _process_frames_task(self, thread_id, shared_inference_times_list):
        logging.info(f"FrameProcessor thread {thread_id} running.")
        while self.running:
            try:
                task_data = self.processing_queue.get(timeout=0.5)
                if task_data is None: break
                frame, timestamp = task_data
                processed_frame = self.estimator.process_frame(frame, shared_inference_times_list)
                try: self.results_queue.put((processed_frame, timestamp), block=False)
                except queue.Full: logging.warning(f"FP results queue FULL (Thread {thread_id}). Dropped.")
            except queue.Empty: continue
            except Exception as e:
                logging.error(f"Error in FrameProcessor thread {thread_id}: {e}", exc_info=True)
        logging.info(f"FrameProcessor thread {thread_id} ended.")

# =================== Main Function ===================
def main():
    # Menggunakan variabel global use_cuda dari settings, tapi akan di-override jika CUDA tidak tersedia
    # Sebaiknya didefinisikan di sini berdasarkan settings.USE_CUDA
    use_cuda_flag = settings.USE_CUDA

    # Menggunakan settings.YOLO_MODEL_PATH
    model = YOLO(settings.YOLO_MODEL_PATH)
    if use_cuda_flag and torch.cuda.is_available():
        model.to("cuda")
        logging.info("Using CUDA for inference.")
    else:
        use_cuda_flag = False # Override jika tidak tersedia atau tidak di-set di settings
        model.to("cpu")
        logging.info("CUDA not available or disabled. Using CPU for inference.")

    # Menggunakan settings.PRELOAD_MODEL
    if settings.PRELOAD_MODEL:
        logging.info("Warming up model...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8) # Ukuran dummy input bisa disesuaikan
        try:
            for _ in range(3):
                _ = model(dummy_input, verbose=False, device=(0 if use_cuda_flag else "cpu"))
            logging.info("Model warmup successful.")
        except Exception as e: logging.error(f"Model warmup failed: {e}")

    estimator = SpeedEstimator(
        model=model,
        debug=settings.DEBUG_MODE,
        real_width=settings.REAL_WIDTH_M,
        real_height=settings.REAL_HEIGHT_M,
        pixels_per_meter=settings.PIXELS_PER_METER,
        video_fps=settings.VIDEO_FPS,
        allowed_direction=settings.ALLOWED_DIRECTION,
        max_speed_threshold=settings.SPEED_THRESHOLD_KMH,
        use_rabbitmq=settings.USE_RABBITMQ,
        pts1=settings.PTS1_COORDINATES,
        outlier_threshold=settings.OUTLIER_SPEED_THRESHOLD_KMH,
        tracker_args_dict=settings.TRACKER_ARGS, # Dilewatkan sebagai dict
        calibration_rules_list=settings.CALIBRATION_RULES, # Dilewatkan sebagai list
        min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
        consistent_speed_buffer_seconds=settings.CONSISTENT_SPEED_BUFFER_SECONDS,
        min_consistent_speed_frames=settings.MIN_CONSISTENT_SPEED_FRAMES,
        wa_recipient=settings.WA_RECIPIENT_NUMBER,
        speed_violators_dir=settings.SPEED_VIOLATORS_DIR,
        counterflow_violators_dir=settings.COUNTERFLOW_VIOLATORS_DIR,
        jpeg_quality=settings.JPEG_QUALITY_FOR_BASE64,
        violation_image_bbox_expansion_factor=settings.VIOLATION_IMAGE_BBOX_EXPANSION_FACTOR
    )

    inference_times_main = []
    frames_processed_main = [0]

    stop_monitor_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(stop_monitor_event, inference_times_main, frames_processed_main, time.time())
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    logging.info("Resource monitoring thread started.")

    # Menggunakan settings.VIDEO_PATH
    is_rtsp_stream = settings.VIDEO_PATH.startswith("rtsp://")
    rtsp_mngr = None
    video_cap = None
    output_video_writer = None

    if is_rtsp_stream:
        # Menggunakan settings.CV2_BUFFER_SIZE dan settings.MAX_QUEUE_SIZE
        rtsp_mngr = RTSPStreamManager(settings.VIDEO_PATH, settings.CV2_BUFFER_SIZE, settings.MAX_QUEUE_SIZE)
        rtsp_mngr.start()
        logging.info(f"RTSP mode: {settings.VIDEO_PATH}. Continuous prediction, no video file output.")
    else:
        video_cap = cv2.VideoCapture(settings.VIDEO_PATH)
        if not video_cap.isOpened():
            logging.error(f"Cannot open video file: {settings.VIDEO_PATH}"); stop_monitor_event.set(); return
        logging.info(f"File mode: {settings.VIDEO_PATH}")
        try:
            f_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            f_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if f_width > 0 and f_height > 0:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # Menggunakan settings.OUTPUT_VIDEO_FILE dan settings.VIDEO_FPS
                output_video_writer = cv2.VideoWriter(settings.OUTPUT_VIDEO_FILE, fourcc, settings.VIDEO_FPS, (f_width, f_height))
                logging.info(f"Output video will be saved to: {settings.OUTPUT_VIDEO_FILE}")
                ret_init, frame_init = video_cap.read()
                if ret_init:
                    estimator.save_transformed_frame(frame_init, "transformed_view_file_setup.jpg")
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: logging.warning("Could not read initial frame for transformed view (file mode).")
            else: logging.error("Invalid frame dimensions from video file. Output disabled.")
        except Exception as e:
            logging.error(f"Error initializing video writer for file: {e}")
            output_video_writer = None

    main_loop_start_time = time.time()
    initial_rtsp_transform_saved = False
    # Menggunakan settings.SKIP_FRAME_INTERVAL
    frame_counter_for_skip = 0

    # Menggunakan settings.PROCESSING_DURATION_SECONDS
    loop_end_time = None
    if settings.PROCESSING_DURATION_SECONDS is not None and not is_rtsp_stream: # Hanya berlaku untuk file video
        loop_end_time = main_loop_start_time + settings.PROCESSING_DURATION_SECONDS

    try:
        while True:
            # Logika durasi untuk file video
            if loop_end_time and time.time() >= loop_end_time:
                logging.info(f"Reached processing duration of {settings.PROCESSING_DURATION_SECONDS} seconds for video file.")
                break

            current_frame = None
            if is_rtsp_stream:
                if rtsp_mngr: current_frame = rtsp_mngr.get_frame(timeout=0.05)
                if current_frame is None: time.sleep(0.01); continue
                if not initial_rtsp_transform_saved:
                    estimator.save_transformed_frame(current_frame, "transformed_view_rtsp_setup.jpg")
                    initial_rtsp_transform_saved = True
            else:
                if video_cap:
                    ret_file, current_frame = video_cap.read()
                    if not ret_file: logging.info("End of video file."); break
                else: logging.error("VideoCap not available in file mode loop."); break
            
            frame_counter_for_skip += 1
            if frame_counter_for_skip % settings.SKIP_FRAME_INTERVAL != 0:
                continue

            if current_frame is not None:
                processed_output_frame = estimator.process_frame(current_frame, inference_times_main)
                frames_processed_main[0] += 1

                if not is_rtsp_stream and output_video_writer:
                    output_video_writer.write(processed_output_frame)
                
                # cv2.imshow("Live", processed_output_frame) # Komentari untuk performa
                # if cv2.waitKey(1) & 0xFF == ord('q'): logging.info("Exit via 'q' key."); break

                # Menggunakan settings.VIDEO_FPS
                if frames_processed_main[0] % (settings.VIDEO_FPS * 10) == 0:
                    elapsed_main = time.time() - main_loop_start_time
                    current_processing_fps = frames_processed_main[0] / elapsed_main if elapsed_main > 0 else 0
                    mode_str = "RTSP" if is_rtsp_stream else "File"
                    logging.info(f"[{mode_str}] Frames: {frames_processed_main[0]}, Overall Proc. FPS: {current_processing_fps:.1f}")
                    if is_rtsp_stream and rtsp_mngr: logging.info(f"RTSP Stats: {rtsp_mngr.get_stats()}")

    except KeyboardInterrupt: logging.info("User interrupted (Ctrl+C).")
    except Exception as e: logging.error(f"MAIN LOOP ERROR: {e}", exc_info=True)
    finally:
        logging.info("Initiating shutdown...")
        stop_monitor_event.set()
        if monitor_thread.is_alive(): monitor_thread.join(timeout=3.0)

        if is_rtsp_stream and rtsp_mngr: rtsp_mngr.stop()
        if video_cap: video_cap.release()
        if output_video_writer:
            output_video_writer.release()
            logging.info(f"Video output saved to {settings.OUTPUT_VIDEO_FILE}")
        
        cv2.destroyAllWindows()
        if gpu_available:
            try: pynvml.nvmlShutdown()
            except pynvml.NVMLError as e: logging.error(f"NVML Shutdown error: {e}")
        
        total_time = time.time() - main_loop_start_time
        logging.info(f"Total frames processed: {frames_processed_main[0]} in {total_time:.2f}s.")
        # Menggunakan settings.RESOURCE_MONITORING_LOG_FILE
        logging.info(f"Monitoring data in: {settings.RESOURCE_MONITORING_LOG_FILE}")
        logging.info("Application terminated.")

if __name__ == "__main__":
    main()