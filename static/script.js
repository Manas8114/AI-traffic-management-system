document.addEventListener('DOMContentLoaded', function() {
    // Fetch traffic light statuses
    fetch('/api/traffic_lights')
        .then(response => response.json())
        .then(data => {
            const trafficLightsContainer = document.getElementById('traffic-lights-container');
            trafficLightsContainer.innerHTML = '';
            for (const [intersection, status] of Object.entries(data)) {
                const lightStatusDiv = document.createElement('div');
                lightStatusDiv.className = 'light-status ' + status.toLowerCase();
                lightStatusDiv.innerHTML = `
                    <label>${intersection}:</label>
                    <input type="text" value="${status.toUpperCase()}" readonly>
                `;
                trafficLightsContainer.appendChild(lightStatusDiv);
            }
        })
        .catch(error => console.error('Error fetching traffic light statuses:', error));

    // Fetch camera feeds
    fetch('/api/cameras')
        .then(response => response.json())
        .then(data => {
            const cameraFeedsContainer = document.getElementById('camera-feeds-container');
            cameraFeedsContainer.innerHTML = '';
            for (const [cameraId, cameraData] of Object.entries(data)) {
                const cameraFeedDiv = document.createElement('div');
                cameraFeedDiv.className = 'camera-feed';
                cameraFeedDiv.innerHTML = `
                    <h3>Camera ${cameraId}</h3>
                    <img src="/api/cameras/${cameraId}/stream" alt="Camera ${cameraId} Feed">
                `;
                cameraFeedsContainer.appendChild(cameraFeedDiv);
            }
        })
        .catch(error => console.error('Error fetching camera feeds:', error));

    // Fetch incidents
    fetch('/api/incidents')
        .then(response => response.json())
        .then(data => {
            const incidentsContainer = document.getElementById('incidents-container');
            incidentsContainer.innerHTML = '';
            data.active.forEach(incident => {
                const incidentDiv = document.createElement('div');
                incidentDiv.className = 'incident-item active';
                incidentDiv.innerHTML = `
                    <h4>${incident.type}</h4>
                    <p>${incident.message}</p>
                    <p><strong>Timestamp:</strong> ${incident.timestamp}</p>
                `;
                incidentsContainer.appendChild(incidentDiv);
            });
            data.historical.forEach(incident => {
                const incidentDiv = document.createElement('div');
                incidentDiv.className = 'incident-item resolved';
                incidentDiv.innerHTML = `
                    <h4>${incident.type}</h4>
                    <p>${incident.message}</p>
                    <p><strong>Timestamp:</strong> ${incident.timestamp}</p>
                `;
                incidentsContainer.appendChild(incidentDiv);
            });
        })
        .catch(error => console.error('Error fetching incidents:', error));

    // Fetch predictions
    fetch('/api/predict?minutes_ahead=15')
        .then(response => response.json())
        .then(data => {
            const predictionsContainer = document.getElementById('predictions-container');
            predictionsContainer.innerHTML = '';
            const predictionDiv = document.createElement('div');
            predictionDiv.className = 'prediction-item';
            predictionDiv.innerHTML = `
                <h4>Predicted Density</h4>
                <p>${data.predicted_density}</p>
                <p><strong>Confidence:</strong> ${data.confidence}</p>
                <p><strong>Predicted Time:</strong> ${data.predicted_time}</p>
            `;
            predictionsContainer.appendChild(predictionDiv);
        })
        .catch(error => console.error('Error fetching predictions:', error));
});
