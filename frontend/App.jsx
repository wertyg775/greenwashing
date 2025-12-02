// frontend/App.jsx
const { useState } = React;

function App() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const analyze = async () => {
        if (!text.trim()) return;
        
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            alert('Error: Make sure backend is running on port 8000');
        }
        setLoading(false);
    };

    const getRiskColor = (level) => {
        if (level === 'HIGH') return '#ef4444';
        if (level === 'MEDIUM') return '#f59e0b';
        return '#10b981';
    };

    return (
        <div className="container">
            <h1>Greenwashing Detection System</h1>
            <p className="subtitle">Module 1: Specificity Detector (85.5% accuracy)</p>

            <div className="input-section">
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter product environmental claim..."
                    rows="4"
                />
                <button onClick={analyze} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Analyze Claim'}
                </button>
            </div>

            {result && (
                <div className="results">
                    <div className="score-card" style={{ borderColor: getRiskColor(result.risk_level) }}>
                        <h2>Risk Score: {result.risk_points}/40</h2>
                        <span className="badge" style={{ backgroundColor: getRiskColor(result.risk_level) }}>
                            {result.risk_level} RISK
                        </span>
                    </div>

                    <div className="details">
                        <div className="detail-row">
                            <span>Prediction:</span>
                            <strong>{result.prediction}</strong>
                        </div>
                        <div className="detail-row">
                            <span>Confidence:</span>
                            <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="progress-bar">
                            <div 
                                className="progress-fill" 
                                style={{ 
                                    width: `${(result.risk_points / 40) * 100}%`,
                                    backgroundColor: getRiskColor(result.risk_level)
                                }}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));