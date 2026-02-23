//! Kalman filter for price smoothing and trend detection.
//!
//! Uses plain f64 arithmetic (no nalgebra). State is `[price, trend]` with
//! a 2x2 covariance matrix stored as `[f64; 4]` in row-major order.

/// Kalman filter with 2D state: `[price, trend]`.
pub struct KalmanFilter {
    /// State vector: [filtered_price, trend].
    state: [f64; 2],
    /// 2x2 covariance matrix (row-major): [P00, P01, P10, P11].
    cov: [f64; 4],
    /// Process noise covariance (row-major 2x2).
    q: [f64; 4],
    /// Measurement noise variance.
    r: f64,
    /// Whether the filter has been initialized with a measurement.
    initialized: bool,
}

impl KalmanFilter {
    /// Create a new Kalman filter with given noise parameters.
    pub fn new(process_var: f64, measurement_var: f64) -> Self {
        Self {
            state: [0.0, 0.0],
            cov: [1.0, 0.0, 0.0, 1.0],
            q: [process_var, 0.0, 0.0, process_var * 0.1],
            r: measurement_var,
            initialized: false,
        }
    }

    /// Sensible defaults for crypto price data.
    pub fn crypto_default() -> Self {
        Self::new(1e-5, 1e-3)
    }

    /// Process a new price measurement and return `(filtered_price, trend)`.
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        if !self.initialized {
            self.state = [price, 0.0];
            self.cov = [self.r, 0.0, 0.0, self.r];
            self.initialized = true;
            return (price, 0.0);
        }

        // --- Predict step ---
        // State transition: x_pred = F * x, where F = [[1, 1], [0, 1]]
        let x_pred = [self.state[0] + self.state[1], self.state[1]];

        // Covariance prediction: P_pred = F * P * F' + Q
        // F = [[1, 1], [0, 1]], F' = [[1, 0], [1, 1]]
        let p = self.cov;
        let fp = [
            p[0] + p[2],
            p[1] + p[3], // F * P row 0
            p[2],
            p[3], // F * P row 1
        ];
        let p_pred = [
            fp[0] + fp[1] + self.q[0], // (F*P*F')[0,0] + Q[0,0]
            fp[1] + self.q[1],         // (F*P*F')[0,1] + Q[0,1]
            fp[2] + fp[3] + self.q[2], // (F*P*F')[1,0] + Q[1,0]
            fp[3] + self.q[3],         // (F*P*F')[1,1] + Q[1,1]
        ];

        // --- Update step ---
        // Observation model: H = [1, 0], so y = price - x_pred[0]
        let y = price - x_pred[0];

        // Innovation covariance: S = H * P_pred * H' + R = P_pred[0,0] + R
        let s = p_pred[0] + self.r;

        // Kalman gain: K = P_pred * H' / S = [P_pred[0,0] / S, P_pred[1,0] / S]
        let k = [p_pred[0] / s, p_pred[2] / s];

        // State update: x = x_pred + K * y
        self.state = [x_pred[0] + k[0] * y, x_pred[1] + k[1] * y];

        // Covariance update: P = (I - K * H) * P_pred
        // K*H = [[k0, 0], [k1, 0]]
        self.cov = [
            (1.0 - k[0]) * p_pred[0],
            (1.0 - k[0]) * p_pred[1],
            -k[1] * p_pred[0] + p_pred[2],
            -k[1] * p_pred[1] + p_pred[3],
        ];

        (self.state[0], self.state[1])
    }

    /// Current filtered price.
    pub fn price(&self) -> f64 {
        self.state[0]
    }

    /// Current trend estimate.
    pub fn trend(&self) -> f64 {
        self.state[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filtered_variance_less_than_raw() {
        let mut kf = KalmanFilter::crypto_default();
        let raw_prices: Vec<f64> = (0..100)
            .map(|i| 50_000.0 + (i as f64 * 0.3).sin() * 500.0 + (i as f64) * 10.0)
            .collect();

        let filtered: Vec<f64> = raw_prices.iter().map(|&p| kf.update(p).0).collect();

        // Compute variance of differences
        let raw_diffs: Vec<f64> = raw_prices.windows(2).map(|w| w[1] - w[0]).collect();
        let filtered_diffs: Vec<f64> = filtered.windows(2).map(|w| w[1] - w[0]).collect();

        let raw_var = variance(&raw_diffs);
        let filtered_var = variance(&filtered_diffs);

        assert!(
            filtered_var < raw_var,
            "Filtered variance ({filtered_var:.2}) should be less than raw ({raw_var:.2})"
        );
    }

    #[test]
    fn test_trend_detection_on_linear_data() {
        let mut kf = KalmanFilter::crypto_default();

        // Feed a perfect linear uptrend: price increases by 10 each step
        for i in 0..50 {
            let price = 50_000.0 + (i as f64) * 10.0;
            kf.update(price);
        }

        // After 50 steps of +10/step, trend should be positive and near 10
        let trend = kf.trend();
        assert!(
            trend > 5.0,
            "Trend should be positive for uptrend, got {trend}"
        );
    }

    #[test]
    fn test_initialization() {
        let mut kf = KalmanFilter::crypto_default();
        let (price, trend) = kf.update(42_000.0);
        assert_eq!(price, 42_000.0);
        assert_eq!(trend, 0.0);
    }

    fn variance(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }
}
