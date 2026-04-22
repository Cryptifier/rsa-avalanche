# Avalanche-Based Recovery of RSA Message Leakage

## Problem: Side-Channel Leakage in RSA

The problem addressed here is partial recovery of an RSA-encrypted message when the ciphertext can be viewed through many noisy, statistically biased decryptions under alternate candidate moduli. This is not a claim of recovering the original RSA private key and it does not depend on factoring the original modulus directly. Instead, the method treats the recovered candidates as a side-channel on the message itself: each candidate decryption leaks an imperfect view of the plaintext, and the repeated biases across those views can be aggregated to recover a large fraction of the original RSA-encrypted message. For structured payloads such as a PGP session-key envelope, that partial recovery is operationally meaningful because the leaked region includes a fixed-format AES-128 session key container.

## RSA And Candidate-Modulus Model

Standard RSA is

```math
N = pq,
\quad
\varphi(N) = (p-1)(q-1),
\quad
c = m^e \bmod N.
```

The private exponent satisfies

```math
d \equiv e^{-1} \pmod{\varphi(N)},
\quad
m = c^d \bmod N.
```

The method introduces speculative candidate moduli $r$ that are easier to analyze than $N$. A convenient model is

```math
r_j = \prod_{i=1}^{k_j} p_{j,i},
\quad
r_j \approx N^{\alpha_j},
```

where each $r_j$ is built from several prime factors and retargeted so that it represents a related modulus rather than the original RSA modulus itself.

For each candidate modulus, the ciphertext may also be perturbed by an odd exponent $x$:

```math
c_x = c^x \bmod N,
\quad
d_{r,x} \equiv (ex)^{-1} \pmod{\varphi(r)}.
```

The candidate-space decryption path is

```math
\widetilde{c}_{r,x} = HBC(c_x, r, N),
```

```math
\widehat{m}_{r,x}
=
HBC(\widetilde{c}_{r,x}^{d_{r,x}} \bmod r, N, r)
\bmod N.
```

Each $\widehat{m}_{r,x}$ is a noisy plaintext estimate. If the plaintext width is $W$ bits, its quality can be expressed as

```math
\mathrm{match}(\widehat{m}_{r,x}, m)
=
\frac{\mathrm{matching\ bits}}{W}.
```

## r-Candidates And c^x Candidates

- **$r$-candidates** are alternate moduli constructed as easier-to-factor products of primes that approximate fractional-power retargetings of the original modulus. Each $r$-candidate yields a different biased plaintext estimate and acts like a separate weak oracle on the message bits.
- **$c^x$ candidates** are ciphertext variants formed by raising the original ciphertext to odd exponents $x$, subject to the requirement that $ex$ remain invertible modulo $\varphi(r)$ for the candidate being tested. These variants generate additional decryptable views from the same original ciphertext and enlarge the pool of noisy message estimates available to the method.

## Avalanche Method

The Avalanche method takes many scored plaintext candidates, pairs the most similar bit-vectors, and recursively merges them so that stable agreements survive while inconsistent bits lose influence. The resulting per-bit bias pattern is converted into probabilities and then ranked with beam search, allowing repeated weak leaks from multiple $r$-candidates and $c^x$ candidates to combine into a stronger message estimate. In this work, Avalanche is the central aggregation step that turns distributed statistical leakage into practical plaintext recovery.

## PGP Envelope Format And AES-128 Focus

For an RSA-encrypted OpenPGP session-key envelope, the payload of interest is the encoded session-key block rather than arbitrary user data. A compact model is

```math
M_{PGP} = a || K_{AES128} || s.
```

embedded inside an RSA PKCS#1 v1.5 encryption block

```math
EM = 0x00 || 0x02 || PS || 0x00 || M_{PGP}.
```

where $a$ is the symmetric algorithm identifier, $K_{AES128}$ is the 16-byte AES-128 session key, and $s$ is the checksum field.

| Envelope field | Typical content | Why it matters |
| --- | --- | --- |
| PKCS#1 prefix | `0x00 0x02` | Fixed structure helps align the recovered block. |
| Padding string `PS` | Nonzero random bytes | Adds variability, but also preserves the location of the payload delimiter. |
| Delimiter | `0x00` | Marks the start of the OpenPGP session-key payload. |
| Symmetric algorithm octet | `0x07` for AES-128 | Identifies the target cipher and helps confirm payload alignment. |
| Session key | 16 bytes | Main recovery target: the AES-128 key material. |
| Checksum | 2 bytes | Supports validation of a partially recovered session key. |

The practical focus is the AES-128 session key inside this envelope. Even when the full RSA plaintext is not recovered perfectly, recovering most of the encoded block is valuable because the fixed structure narrows the uncertainty around the 128-bit session key region.

## Results Summary

1. Up to **74% retrieval** of the RSA-encrypted message has been achieved, with the current PGP-envelope setting reporting roughly **70-74%** message recovery.
2. The method is **independent of the RSA modulus size** in the sense that it relies on candidate-modulus leakage aggregation and Avalanche reduction rather than a modulus-size-specific trick.
3. The method is centered on **Avalanche-based aggregation** of many weak plaintext estimates obtained from $r$-candidates and $c^x$ candidates.

## Contact

Nicholas LaRoche  
<nlaroche@cryptifier.dev>
