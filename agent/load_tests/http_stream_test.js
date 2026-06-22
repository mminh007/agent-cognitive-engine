// load_tests/http_stream_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

// Create custom metrics for Grafana monitoring
const ttfbTrend = new Trend('custom_ttfb', true);
const errorRate = new Rate('custom_errors');

export const options = {
    // Load test scenario: ramp up to 50 users, hold for 30s, then ramp down
    stages: [
        { duration: '10s', target: 50 }, // Ramp-up
        { duration: '30s', target: 50 }, // Sustain load
        { duration: '10s', target: 0 },  // Ramp-down
    ],
};

export default function () {
    const url = 'http://agent_api_server:8000/api/chat/stream';
    
    // Generate a unique session_id to avoid cache collisions
    // (useful when testing actual system load)
    const sessionId = `loadtest_session_${__VU}_${__ITER}`;
    
    const payload = JSON.stringify({
        user_id: "k6_loadtest_user",
        session_id: sessionId,
        prompt: "Explain and analyze the architecture of a Multi-Agent system built with LangGraph."
    });

    const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'fastapi_sse_chat' }
    };

    // Send request
    const res = http.post(url, payload, params);

    // Record Time to First Byte (TTFB)
    // Time until the first chunk of data is received
    ttfbTrend.add(res.timings.waiting);

    // Validate response integrity
    const isSuccessful = check(res, {
        'status is 200': (r) => r.status === 200,
        'is streaming response': (r) => r.body.includes('data:'),
    });

    if (!isSuccessful) {
        errorRate.add(1);
        console.error(
            `\n[k6 Error] Request failed for session ${sessionId}. Status: ${res.status}\n`
        );
    }

    // Simulate user think time before sending the next message
    sleep(1);
}