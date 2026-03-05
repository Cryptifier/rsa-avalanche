from dataclasses import dataclass
import math
from typing import List, Tuple

@dataclass(order=True)
class Hypothesis:
    score: float          # log-probability (bigger is better)
    bits: Tuple[int, ...] # immutable so we can safely store/compare

def beam_search_bits(p_ones: List[float], beam_width: int = 32) -> List[Hypothesis]:
    """
    Beam search for the most likely bitstring given per-position probabilities p_i = P(bit_i = 1).
    Returns the final beam (sorted best-first).
    """
    # Small epsilon to avoid log(0) if p_i is exactly 0 or 1 from estimation artifacts.
    eps = 1e-15

    beam = [Hypothesis(score=0.0, bits=())]

    for i, p in enumerate(p_ones):
        p = min(max(p, eps), 1.0 - eps)

        next_beam = []
        for h in beam:
            # Extend with 0
            s0 = h.score + math.log(1.0 - p)
            next_beam.append(Hypothesis(score=s0, bits=h.bits + (0,)))

            # Extend with 1
            s1 = h.score + math.log(p)
            next_beam.append(Hypothesis(score=s1, bits=h.bits + (1,)))

        # Keep top beam_width by score
        next_beam.sort(reverse=True)  # Hypothesis is orderable by score because of @dataclass(order=True)
        beam = next_beam[:beam_width]

    return beam

if __name__ == "__main__":
    # Example: 16-bit "oracle probabilities"
    p = [0.52, 0.49, 0.501, 0.7, 0.3, 0.55, 0.53, 0.48, 0.9, 0.1, 0.6, 0.4, 0.51, 0.51, 0.49, 0.52]
    beam = beam_search_bits(p, beam_width=8)
    best = beam[0]
    print("Best bits:", "".join(map(str, best.bits)))
    print("Log-score:", best.score)
    print("Top 3:")
    for h in beam[:3]:
        print(" ", "".join(map(str, h.bits)), h.score)
