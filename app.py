from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import threading
import time
import heapq
import io
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8l.pt')

# CCTV video paths
video_paths = [
    "https://livepantau.semarangkota.go.id/hls/414/701/2024/8795266c-ebc3-4a95-8827-b82360403f0a_701.m3u8",  # 0. kalibanteng
    "https://livepantau.semarangkota.go.id/hls/414/1101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_110139.ts",  # 1. kaligarang
    "https://livepantau.semarangkota.go.id/hls/414/4801/2024/8795266c-ebc3-4a95-8827-b82360403f0a_4801.m3u8",  # 2. madukuro
    "https://livepantau.semarangkota.go.id/hls/414/901/2024/8795266c-ebc3-4a95-8827-b82360403f0a_901.m3u8",  # 3. kariadi
    "https://livepantau.semarangkota.go.id/hls/414/101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_10119.ts",  # 4. tugu muda
    "https://livepantau.semarangkota.go.id/hls/414/5201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_520120.ts",  # 5. indraprasta
    "https://livepantau.semarangkota.go.id/hls/414/5301/2024/8795266c-ebc3-4a95-8827-b82360403f0a_5301.m3u8",  # 6. bergota
    "https://livepantau.semarangkota.go.id/hls/414/201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_201.m3u8",  # 7. simp kyai saleh
]

status_thresholds = {'lancar': 5, 'padat': 15, 'macet': 20}

# Graph representation
graph = nx.Graph()

# Define graph nodes with positions and labels
positions = {
    0: (0, 0), 1: (1, 2), 2: (2, 0), 3: (3, 2),
    4: (4, 0), 5: (5, 2), 6: (6, 0), 7: (7, 2)
}

node_labels = {
    0: "Kalibanteng", 1: "Kaligarang", 2: "Madukuro", 3: "Kariadi",
    4: "Tugu Muda", 5: "Indraprasta", 6: "Bergota", 7: "Simpang Kyai Saleh"
}

# Define edges and their initial weights
edges = [
    (0, 1),  # 0 terhubung ke 1
    (0, 2),  # 0 terhubung ke 2
    (2, 5),  # 2 terhubung ke 5
    (2, 1),  # 2 terhubung ke 1
    (2, 4),  # 2 terhubung ke 4
    (5, 4),  # 5 terhubung ke 4
    (1, 3),  # 1 terhubung ke 3
    (4, 3),  # 4 terhubung ke 3
    (4, 7),  # 4 terhubung ke 7
    (7, 6),  # 7 terhubung ke 6
    (3, 6)   # 3 terhubung ke 6
]

graph.add_nodes_from(positions.keys())
graph.add_edges_from(edges, weight=1)

# Traffic status tracking for edges
traffic_status = {edge: "lancar" for edge in graph.edges()}

# Store the latest frames
latest_frames = {i: None for i in range(len(video_paths))}
vehicle_counts = {i: 0 for i in range(len(video_paths))}
traffic_statuses = {i: "lancar" for i in range(len(video_paths))}

# Dijkstra algorithm implementation
def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph.nodes()}
    dist[start] = 0
    pq = [(0, start)]
    prev = {node: None for node in graph.nodes()}

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > dist[current_node]:
            continue
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return dist, prev

# Function to recommend best route
def recommend_route(start, end):
    dist, prev = dijkstra(graph, start)
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = prev[current_node]
    path.reverse()
    return path, dist[end]

# Function to update edge weights based on traffic conditions
def update_edge_weights(node_id, vehicle_count):
    travel_time = (
        1 if vehicle_count <= status_thresholds['lancar'] else
        3 if vehicle_count <= status_thresholds['padat'] else
        5
    )
    for neighbor in graph.neighbors(node_id):
        graph[node_id][neighbor]['weight'] = travel_time
        if (node_id, neighbor) in traffic_status:
            traffic_status[(node_id, neighbor)] = (
                "lancar" if vehicle_count <= status_thresholds['lancar'] else
                "padat" if vehicle_count <= status_thresholds['padat'] else
                "macet"
            )
        elif (neighbor, node_id) in traffic_status:
            traffic_status[(neighbor, node_id)] = (
                "lancar" if vehicle_count <= status_thresholds['lancar'] else
                "padat" if vehicle_count <= status_thresholds['padat'] else
                "macet"
            )

# Function to visualize the graph
def visualize_graph(start=0, end=7):
    plt.figure(figsize=(10, 6))
    edge_colors = []
    for edge in graph.edges(data=True):
        weight = edge[2]['weight']
        if weight == 1:
            edge_colors.append("green")
        elif weight == 3:
            edge_colors.append("yellow")
        else:
            edge_colors.append("red")

    # Highlight the best route in blue
    best_route, _ = recommend_route(start, end)
    best_edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
    
    nx.draw(graph, pos=positions, with_labels=True, node_size=700, node_color="lightblue", 
            labels=node_labels, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos=positions, edge_color=edge_colors, width=3)
    nx.draw_networkx_edges(graph, pos=positions, edgelist=best_edges, edge_color='blue', width=4)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the buffer to base64
    graph_image = base64.b64encode(buf.read()).decode('utf-8')
    return graph_image, best_route

def process_video(video_path, node_id):
    global latest_frames, vehicle_counts, traffic_statuses
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)  # Restart the video capture
            continue

        # YOLO Inference
        results = model.predict(frame, stream=True)
        vehicle_count = 0  # Reset vehicle count

        # Allowed vehicle classes
        allowed_classes = {'car', 'truck', 'bus', 'motorcycle'}

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                label = model.names[cls]

                if label not in allowed_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0]
                vehicle_count += 1

                color_map = {
                    'car': (0, 255, 0),
                    'truck': (0, 0, 255),
                    'bus': (255, 0, 0),
                    'motorcycle': (255, 255, 0)
                }
                color = color_map.get(label, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        update_edge_weights(node_id, vehicle_count)
        vehicle_counts[node_id] = vehicle_count

        traffic_status_text = (
            "Lancar" if vehicle_count <= status_thresholds['lancar'] else
            "Padat" if vehicle_count <= status_thresholds['padat'] else
            "Macet"
        )
        traffic_statuses[node_id] = traffic_status_text

        cv2.putText(frame, f"{node_labels[node_id]} | Status: {traffic_status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Jumlah Kendaraan: {vehicle_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Store the latest frame
        latest_frames[node_id] = frame.copy()
        
        time.sleep(0.1)

    cap.release()

# Function to encode a frame to base64
def encode_frame(frame):
    if frame is None:
        # Return a black frame if no frame is available
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", black_frame)
    else:
        # Resize frame for better performance
        resized_frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode(".jpg", resized_frame)
    
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html', node_labels=node_labels)

@app.route('/get_frames')
def get_frames():
    encoded_frames = {}
    for node_id, frame in latest_frames.items():
        encoded_frames[node_id] = {
            'image': encode_frame(frame),
            'vehicle_count': vehicle_counts[node_id],
            'status': traffic_statuses[node_id]
        }
    return jsonify(encoded_frames)

@app.route('/get_route', methods=['POST'])
def get_route():
    data = request.get_json()
    start_node = int(data['start'])
    end_node = int(data['end'])
    
    graph_image, route = visualize_graph(start_node, end_node)
    route_text = []
    
    for i in range(len(route) - 1):
        current = route[i]
        next_node = route[i + 1]
        status = traffic_status.get((current, next_node), traffic_status.get((next_node, current), "lancar"))
        route_text.append({
            'from': node_labels[current],
            'to': node_labels[next_node],
            'status': status
        })
    
    return jsonify({
        'graph_image': graph_image,
        'route': route_text,
        'route_nodes': [node_labels[node] for node in route]
    })

if __name__ == '__main__':
    # Start threads for each video stream
    for node_id, video_path in enumerate(video_paths):
        t = threading.Thread(target=process_video, args=(video_path, node_id), daemon=True)
        t.start()
    
    app.run(debug=True, threaded=True)