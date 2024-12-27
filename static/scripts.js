document.addEventListener("DOMContentLoaded", function() {
    var videoPaths = [
        "https://livepantau.semarangkota.go.id/hls/414/701/2024/8795266c-ebc3-4a95-8827-b82360403f0a_701.m3u8",  // Kalibanteng
        "https://livepantau.semarangkota.go.id/hls/414/1101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_1101.m3u8",  // Kaligarang
        "https://livepantau.semarangkota.go.id/hls/414/4801/2024/8795266c-ebc3-4a95-8827-b82360403f0a_4801.m3u8",  // Madukuro
        "https://livepantau.semarangkota.go.id/hls/414/901/2024/8795266c-ebc3-4a95-8827-b82360403f0a_901.m3u8",   // Kariadi
        "https://livepantau.semarangkota.go.id/hls/414/101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_101.m3u8",   // Tugu Muda
        "https://livepantau.semarangkota.go.id/hls/414/5201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_5201.m3u8",  // Indraprasta
        "https://livepantau.semarangkota.go.id/hls/414/5301/2024/8795266c-ebc3-4a95-8827-b82360403f0a_5301.m3u8",  // Bergota
        "https://livepantau.semarangkota.go.id/hls/414/201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_201.m3u8"     // Simpang Kyai Saleh
    ];

    // Load the video feeds
    for (let i = 0; i < videoPaths.length; i++) {
        let videoElement = document.getElementById(`video-feed-${i}`);
        if (Hls.isSupported()) {
            let hls = new Hls();
            hls.loadSource(videoPaths[i]);
            hls.attachMedia(videoElement);
            hls.on(Hls.Events.MANIFEST_PARSED, function () {
                videoElement.play();
            });
        } else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
            videoElement.src = videoPaths[i];
            videoElement.addEventListener('loadedmetadata', function() {
                videoElement.play();
            });
        }
    }

    // Event listener for the submit button
    document.getElementById('submit-btn').addEventListener('click', function() {
        var startPoint = document.getElementById('start-point').value;
        var endPoint = document.getElementById('end-point').value;

        // Draw the graph after selecting start and end points
        var ctx = document.getElementById('graph').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Titik Awal', 'Titik Tujuan'],
                datasets: [{
                    label: 'Perjalanan',
                    data: [parseInt(startPoint), parseInt(endPoint)],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Show the result graph
        document.getElementById('result-graph').style.display = 'block';
    });
});