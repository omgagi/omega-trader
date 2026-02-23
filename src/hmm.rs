//! Hidden Markov Model for regime detection (3-state: Bull, Bear, Lateral).
//!
//! Uses 5 discrete observations (BigDown, SmallDown, Flat, SmallUp, BigUp)
//! and Baum-Welch for online training. No external stats library needed —
//! all probability math is inline.

use crate::signal::Regime;

const NUM_STATES: usize = 3;
const NUM_OBS: usize = 5;

/// Discretized return observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation {
    BigDown = 0,
    SmallDown = 1,
    Flat = 2,
    SmallUp = 3,
    BigUp = 4,
}

/// 3-state Hidden Markov Model for regime detection.
pub struct HiddenMarkovModel {
    /// Transition probabilities: `transition[i][j]` = P(state j | state i).
    transition: Vec<Vec<f64>>,
    /// Emission probabilities: `emission[i][j]` = P(obs j | state i).
    emission: Vec<Vec<f64>>,
    /// Current belief (posterior) over states.
    belief: [f64; NUM_STATES],
    /// Duration counter for current regime.
    regime_duration: u64,
    /// Previous regime for duration tracking.
    prev_regime: Option<Regime>,
}

#[allow(clippy::needless_range_loop)]
impl HiddenMarkovModel {
    /// Crypto-tuned defaults.
    pub fn crypto_default() -> Self {
        // Transition: high self-transition (0.85), distributed rest
        let transition = vec![
            vec![0.85, 0.10, 0.05], // Bull stays bull
            vec![0.10, 0.85, 0.05], // Bear stays bear
            vec![0.10, 0.10, 0.80], // Lateral is stickiest outward
        ];

        // Emission probabilities: P(observation | state)
        // Rows: Bull, Bear, Lateral; Cols: BigDown, SmallDown, Flat, SmallUp, BigUp
        let emission = vec![
            vec![0.02, 0.08, 0.20, 0.35, 0.35], // Bull: skews up
            vec![0.35, 0.35, 0.20, 0.08, 0.02], // Bear: skews down
            vec![0.05, 0.20, 0.50, 0.20, 0.05], // Lateral: centered
        ];

        Self {
            transition,
            emission,
            belief: [1.0 / 3.0; NUM_STATES],
            regime_duration: 0,
            prev_regime: None,
        }
    }

    /// Discretize a percentage return into an observation category.
    pub fn discretize_return(pct_return: f64) -> Observation {
        if pct_return < -0.02 {
            Observation::BigDown
        } else if pct_return < -0.005 {
            Observation::SmallDown
        } else if pct_return <= 0.005 {
            Observation::Flat
        } else if pct_return <= 0.02 {
            Observation::SmallUp
        } else {
            Observation::BigUp
        }
    }

    /// Update belief with a new observation, return (most_likely_regime, probabilities).
    pub fn update(&mut self, obs: Observation) -> (Regime, [f64; NUM_STATES]) {
        let obs_idx = obs as usize;

        // Forward step: belief' = normalize(emission[s][obs] * sum_s'(transition[s'][s] * belief[s']))
        let mut new_belief = [0.0; NUM_STATES];
        for j in 0..NUM_STATES {
            let mut sum = 0.0;
            for i in 0..NUM_STATES {
                sum += self.transition[i][j] * self.belief[i];
            }
            new_belief[j] = self.emission[j][obs_idx] * sum;
        }

        // Normalize
        let total: f64 = new_belief.iter().sum();
        if total > 1e-300 {
            for b in &mut new_belief {
                *b /= total;
            }
        } else {
            // Fallback to uniform if degenerate
            new_belief = [1.0 / 3.0; NUM_STATES];
        }

        self.belief = new_belief;

        // Determine regime
        let regime = if new_belief[0] >= new_belief[1] && new_belief[0] >= new_belief[2] {
            Regime::Bull
        } else if new_belief[1] >= new_belief[0] && new_belief[1] >= new_belief[2] {
            Regime::Bear
        } else {
            Regime::Lateral
        };

        // Track duration
        if self.prev_regime == Some(regime) {
            self.regime_duration += 1;
        } else {
            self.regime_duration = 1;
            self.prev_regime = Some(regime);
        }

        (regime, new_belief)
    }

    /// Train the model on a sequence of returns using Baum-Welch.
    pub fn train(&mut self, returns: &[f64], n_iterations: usize) {
        if returns.is_empty() {
            return;
        }

        let observations: Vec<usize> = returns
            .iter()
            .map(|&r| Self::discretize_return(r) as usize)
            .collect();
        let t = observations.len();

        for _ in 0..n_iterations {
            // --- Forward pass ---
            let mut alpha = vec![[0.0; NUM_STATES]; t];
            for s in 0..NUM_STATES {
                alpha[0][s] = (1.0 / NUM_STATES as f64) * self.emission[s][observations[0]];
            }
            normalize_slice(&mut alpha[0]);

            for step in 1..t {
                for j in 0..NUM_STATES {
                    let mut sum = 0.0;
                    for i in 0..NUM_STATES {
                        sum += alpha[step - 1][i] * self.transition[i][j];
                    }
                    alpha[step][j] = sum * self.emission[j][observations[step]];
                }
                normalize_slice(&mut alpha[step]);
            }

            // --- Backward pass ---
            let mut beta = vec![[0.0; NUM_STATES]; t];
            beta[t - 1] = [1.0; NUM_STATES];

            for step in (0..t - 1).rev() {
                for i in 0..NUM_STATES {
                    let mut sum = 0.0;
                    for j in 0..NUM_STATES {
                        sum += self.transition[i][j]
                            * self.emission[j][observations[step + 1]]
                            * beta[step + 1][j];
                    }
                    beta[step][i] = sum;
                }
                normalize_slice(&mut beta[step]);
            }

            // --- M-step: re-estimate transition and emission ---
            let mut new_transition = vec![vec![0.0; NUM_STATES]; NUM_STATES];
            let mut new_emission = vec![vec![0.0; NUM_OBS]; NUM_STATES];
            let mut gamma_sum = [0.0; NUM_STATES];

            for step in 0..t {
                let mut gamma = [0.0; NUM_STATES];
                for s in 0..NUM_STATES {
                    gamma[s] = alpha[step][s] * beta[step][s];
                }
                normalize_slice(&mut gamma);

                for s in 0..NUM_STATES {
                    gamma_sum[s] += gamma[s];
                    new_emission[s][observations[step]] += gamma[s];
                }

                if step < t - 1 {
                    let mut xi = [[0.0; NUM_STATES]; NUM_STATES];
                    for i in 0..NUM_STATES {
                        for j in 0..NUM_STATES {
                            xi[i][j] = alpha[step][i]
                                * self.transition[i][j]
                                * self.emission[j][observations[step + 1]]
                                * beta[step + 1][j];
                        }
                    }
                    let xi_total: f64 = xi.iter().flat_map(|row| row.iter()).sum();
                    if xi_total > 1e-300 {
                        for i in 0..NUM_STATES {
                            for j in 0..NUM_STATES {
                                new_transition[i][j] += xi[i][j] / xi_total;
                            }
                        }
                    }
                }
            }

            // Normalize and apply
            for i in 0..NUM_STATES {
                let row_sum: f64 = new_transition[i].iter().sum();
                if row_sum > 1e-300 {
                    for j in 0..NUM_STATES {
                        self.transition[i][j] = new_transition[i][j] / row_sum;
                    }
                }

                if gamma_sum[i] > 1e-300 {
                    for o in 0..NUM_OBS {
                        self.emission[i][o] = new_emission[i][o] / gamma_sum[i];
                    }
                }
            }
        }
    }

    /// Current belief probabilities.
    pub fn probabilities(&self) -> [f64; NUM_STATES] {
        self.belief
    }

    /// How long the current regime has persisted (in ticks).
    pub fn regime_duration(&self) -> u64 {
        self.regime_duration
    }
}

/// Normalize a slice to sum to 1.
fn normalize_slice(slice: &mut [f64; NUM_STATES]) {
    let total: f64 = slice.iter().sum();
    if total > 1e-300 {
        for v in slice.iter_mut() {
            *v /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bull_detection_after_positive_returns() {
        let mut hmm = HiddenMarkovModel::crypto_default();

        // Feed many positive returns
        for _ in 0..20 {
            hmm.update(Observation::BigUp);
        }

        let probs = hmm.probabilities();
        assert!(
            probs[0] > probs[1] && probs[0] > probs[2],
            "Bull should dominate after BigUp sequence: {probs:?}"
        );
    }

    #[test]
    fn test_bear_detection_after_negative_returns() {
        let mut hmm = HiddenMarkovModel::crypto_default();

        for _ in 0..20 {
            hmm.update(Observation::BigDown);
        }

        let probs = hmm.probabilities();
        assert!(
            probs[1] > probs[0] && probs[1] > probs[2],
            "Bear should dominate after BigDown sequence: {probs:?}"
        );
    }

    #[test]
    fn test_lateral_detection_after_flat_returns() {
        let mut hmm = HiddenMarkovModel::crypto_default();

        for _ in 0..30 {
            hmm.update(Observation::Flat);
        }

        let probs = hmm.probabilities();
        assert!(
            probs[2] > probs[0] && probs[2] > probs[1],
            "Lateral should dominate after Flat sequence: {probs:?}"
        );
    }

    #[test]
    fn test_baum_welch_rows_sum_to_one() {
        let mut hmm = HiddenMarkovModel::crypto_default();

        // Generate some training data
        let returns: Vec<f64> = (0..100).map(|i| ((i as f64) * 0.1).sin() * 0.03).collect();
        hmm.train(&returns, 5);

        // Check transition rows sum to 1
        for (i, row) in hmm.transition.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Transition row {i} sums to {sum}, expected 1.0"
            );
        }

        // Check emission rows sum to 1
        for (i, row) in hmm.emission.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Emission row {i} sums to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_discretize_return() {
        assert_eq!(
            HiddenMarkovModel::discretize_return(-0.05),
            Observation::BigDown
        );
        assert_eq!(
            HiddenMarkovModel::discretize_return(-0.01),
            Observation::SmallDown
        );
        assert_eq!(HiddenMarkovModel::discretize_return(0.0), Observation::Flat);
        assert_eq!(
            HiddenMarkovModel::discretize_return(0.01),
            Observation::SmallUp
        );
        assert_eq!(
            HiddenMarkovModel::discretize_return(0.05),
            Observation::BigUp
        );
    }

    #[test]
    fn test_regime_duration_tracking() {
        let mut hmm = HiddenMarkovModel::crypto_default();

        // Feed consistent bull observations
        for _ in 0..10 {
            hmm.update(Observation::BigUp);
        }
        assert!(
            hmm.regime_duration() >= 5,
            "Duration should track consecutive regime"
        );
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let mut hmm = HiddenMarkovModel::crypto_default();
        hmm.update(Observation::SmallUp);

        let probs = hmm.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {sum}"
        );
    }
}
