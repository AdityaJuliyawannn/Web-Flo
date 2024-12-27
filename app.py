import math
import threading
import heapq
import time
import matplotlib.pyplot as plt
import networkx as nx
from ultralytics import YOLO
import cv2
from flask import Flask, jsonify, request, Response, render_template

# Flask app initialization
app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8l.pt')

# CCTV video paths
video_paths = [
    "https://livepantau.semarangkota.go.id/hls/414/4801/2024/8795266c-ebc3-4a95-8827-b82360403f0a_4801.m3u8",
    "https://livepantau.semarangkota.go.id/hls/414/701/2024/8795266c-ebc3-4a95-8827-b82360403f0a_701.m3u8",
    "https://livepantau.semarangkota.go.id/hls/414/101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_10119.ts",
    "https://livepantau.semarangkota.go.id/hls/414/201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_201.m3u8",
    "https://livepantau.semarangkota.go.id/hls/414/901/2024/8795266c-ebc3-4a95-8827-b82360403f0a_901.m3u8",
    "https://livepantau.semarangkota.go.id/hls/414/5201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_520120.ts",
    "https://livepantau.semarangkota.go.id/hls/414/1101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_110139.ts",
    "https://livepantau.semarangkota.go.id/hls/414/5301/2024/8795266c-ebc3-4a95-8827-b82360403f0a_5301.m3u8"
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

edges = [
    (0, 1), (0, 2), (2, 5), (2, 1), (2, 4),
    (5, 4), (1, 3), (4, 3), (4, 7), (7, 6), (3, 6)
]

graph.add_nodes_from(positions.keys())
graph.add_edges_from(edges, weight=1)

traffic_status = {edge: "lancar" for edge in graph.edges}

# Dijkstra's algorithm
def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph.nodes}
    dist[start] = 0
    pq = [(0, start)]
    prev = {node: None for node in graph.nodes}

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

# Recommend the best route
def recommend_route(start, end):
    dist, prev = dijkstra(graph, start)
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = prev[current_node]
    path.reverse()
    return path

# Update edge weights based on traffic conditions
def update_edge_weights(node_id, vehicle_count):
    travel_time = (
        1 if vehicle_count <= status_thresholds['lancar'] else
        3 if vehicle_count <= status_thresholds['padat'] else
        5
    )
    for neighbor in graph.neighbors(node_id):
        graph[node_id][neighbor]['weight'] = travel_time
        traffic_status[(node_id, neighbor)] = (
            "lancar" if vehicle_count <= status_thresholds['lancar'] else
            "padat" if vehicle_count <= status_thresholds['padat'] else
            "macet"
        )
    visualize_graph()


# Visualize the graph
def visualize_graph():
    plt.clf()
    edge_colors = [
        "green" if graph[u][v]['weight'] == 1 else
        "yellow" if graph[u][v]['weight'] == 3 else
        "red"
        for u, v in graph.edges
    ]
    nx.draw(
        graph, pos=positions, with_labels=True, 
        node_size=1200, node_color="#1f78b4", font_size=10, font_color="white", 
        labels=node_labels
    )
    nx.draw_networkx_edges(
        graph, pos=positions, edge_color=edge_colors, width=2
    )
    plt.title("Peta Lalu Lintas", fontsize=15)
    plt.savefig('static/traffic_graph.png')
    plt.close()


# Process video and update traffic status
def process_video(video_path, node_id):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=True)
        vehicle_count = 0
        allowed_classes = {'car', 'truck', 'bus', 'motorcycle'}

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in allowed_classes:
                    vehicle_count += 1

        update_edge_weights(node_id, vehicle_count)
        visualize_graph()
        time.sleep(0.1)

    cap.release()


# Video stream function
def video_stream(video_url):
    cap = cv2.VideoCapture(video_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', cameras=range(len(video_paths)))

@app.route('/route', methods=['GET'])
def get_route():
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    path = recommend_route(start, end)
    readable_path = [node_labels[node] for node in path]
    return jsonify({"route": readable_path})

@app.route('/traffic', methods=['GET'])
def get_traffic():
    readable_status = {
        f"{node_labels[u]}-{node_labels[v]}": status
        for (u, v), status in traffic_status.items()
    }
    return jsonify(readable_status)

@app.route('/camera/<int:cam_id>')
def camera_feed(cam_id):
    # Stream the video from the selected CCTV link
    return Response(video_stream(video_paths[cam_id]), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)