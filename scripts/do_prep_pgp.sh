#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

KEY_DIR="${REPO_ROOT}/config/keys"
PREP_PGP_MODULUS_BITS=${PREP_PGP_MODULUS_BITS:-2048}
PREP_PGP_KEY_BASENAME=${PREP_PGP_KEY_BASENAME:-prep_pgp}
PREP_PGP_OUTPUT_DIR=${PREP_PGP_OUTPUT_DIR:-config/pgp}
PREP_PGP_CIPHER_ALGO=${PREP_PGP_CIPHER_ALGO:-}
if [[ "${PREP_PGP_OUTPUT_DIR}" = /* ]]; then
  PGP_DIR="${PREP_PGP_OUTPUT_DIR}"
else
  PGP_DIR="${REPO_ROOT}/${PREP_PGP_OUTPUT_DIR}"
fi
PRIVATE_KEY="${KEY_DIR}/${PREP_PGP_KEY_BASENAME}_private_key.yaml"
PUBLIC_KEY="${KEY_DIR}/${PREP_PGP_KEY_BASENAME}_public_key.yaml"
ARMORED_PUBLIC_KEY="${KEY_DIR}/${PREP_PGP_KEY_BASENAME}_public.asc"
IDENTITY_FILE="${KEY_DIR}/${PREP_PGP_KEY_BASENAME}_identity.env"
CONVERTER="${SCRIPT_DIR}/convert_rsa_yaml_to_pgp.py"

mkdir -p "${KEY_DIR}" "${PGP_DIR}"

if [[ ! -f "${PUBLIC_KEY}" && ! -f "${PRIVATE_KEY}" ]]; then
  cargo run --quiet --bin kgen -- \
    --size-mode modulus \
    --modulus-bits "${PREP_PGP_MODULUS_BITS}" \
    --output "${PRIVATE_KEY}" \
    --public-output "${PUBLIC_KEY}"
elif [[ ! -f "${PUBLIC_KEY}" && -f "${PRIVATE_KEY}" ]]; then
  cargo run --quiet --bin kgen -- \
    --input-private-key "${PRIVATE_KEY}" \
    --public-output "${PUBLIC_KEY}" \
    --force
fi

if [[ ! -f "${PUBLIC_KEY}" ]]; then
  echo "Missing RSA public key ${PUBLIC_KEY}" >&2
  exit 1
fi

if [[ ! -f "${IDENTITY_FILE}" ]]; then
  random_local="prep-$(openssl rand -hex 6)"
  prep_name="Prep Dev"
  prep_email="${random_local}@dev.local"
  printf 'PREP_PGP_NAME=%q\nPREP_PGP_EMAIL=%q\n' \
    "${prep_name}" \
    "${prep_email}" \
    > "${IDENTITY_FILE}"
fi

# shellcheck disable=SC1090
source "${IDENTITY_FILE}"

if [[ ! -f "${ARMORED_PUBLIC_KEY}" ]]; then
  if [[ ! -f "${PRIVATE_KEY}" ]]; then
    echo "Cannot create ${ARMORED_PUBLIC_KEY} without ${PRIVATE_KEY}" >&2
    exit 1
  fi

  python3 "${CONVERTER}" \
    --private-key "${PRIVATE_KEY}" \
    --public-key "${PUBLIC_KEY}" \
    --name "${PREP_PGP_NAME}" \
    --email "${PREP_PGP_EMAIL}" \
    --output "${ARMORED_PUBLIC_KEY}"
fi

tmpdir=$(mktemp -d)
trap 'rm -rf -- "${tmpdir}"' EXIT

gnupghome="${tmpdir}/gnupg"
mkdir -m 700 "${gnupghome}"

if ! gpg --homedir "${gnupghome}" --batch --import "${ARMORED_PUBLIC_KEY}" >/dev/null 2>&1; then
  if ! gpg --homedir "${gnupghome}" --batch --list-keys "${PREP_PGP_EMAIL}" >/dev/null 2>&1; then
    echo "Failed to import ${ARMORED_PUBLIC_KEY} into temporary GPG home ${gnupghome}" >&2
    exit 1
  fi
fi

python3 - <<'PY' "${REPO_ROOT}" "${PGP_DIR}"
import random
import re
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
rng = random.SystemRandom()


def clean_paragraph(text: str) -> str:
    paragraph = text.strip()
    paragraph = re.sub(r"\s+", " ", paragraph)
    return paragraph


paragraphs = []
for markdown_path in sorted(repo_root.rglob("*.md")):
    if ".git" in markdown_path.parts:
        continue
    try:
        source_text = markdown_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source_text = markdown_path.read_text(encoding="utf-8", errors="ignore")

    for raw_paragraph in re.split(r"\n\s*\n", source_text):
        paragraph = clean_paragraph(raw_paragraph)
        if len(paragraph.split()) >= 20:
            paragraphs.append((markdown_path.relative_to(repo_root).as_posix(), paragraph))

if len(paragraphs) < 25:
    raise SystemExit("not enough markdown paragraph content to build random blobs")

targets = [
    (90, 150),
    (180, 280),
    (320, 460),
    (520, 700),
    (850, 1150),
]

for index, (minimum_words, maximum_words) in enumerate(targets, start=1):
    target_words = rng.randint(minimum_words, maximum_words)
    chosen = []
    used_sources = set()
    remaining = target_words

    while remaining > 0:
        source_path, paragraph = rng.choice(paragraphs)
        paragraph_words = len(paragraph.split())

        if len(used_sources) < 4 and source_path in used_sources:
            continue

        chosen.append(f"[source: {source_path}]\n{paragraph}")
        used_sources.add(source_path)
        remaining -= paragraph_words

    output_path = output_dir / f"prep_blob_{index}.txt"
    output_path.write_text("\n\n".join(chosen) + "\n", encoding="utf-8")
PY

for index in 1 2 3 4 5; do
  plaintext_path="${PGP_DIR}/prep_blob_${index}.txt"
  output_path="${PGP_DIR}/prep_blob_${index}.asc"

  gpg_args=(
    --homedir "${gnupghome}"
    --batch
    --yes
    --trust-model always
    --armor
    --recipient "${PREP_PGP_EMAIL}"
    --output "${output_path}"
  )
  if [[ -n "${PREP_PGP_CIPHER_ALGO}" ]]; then
    gpg_args+=(--cipher-algo "${PREP_PGP_CIPHER_ALGO}")
  fi
  gpg "${gpg_args[@]}" --encrypt "${plaintext_path}"
done

printf 'Prepared RSA keypair %s and %s\n' "${PRIVATE_KEY}" "${PUBLIC_KEY}"
printf 'Prepared armored OpenPGP public key %s for %s\n' "${ARMORED_PUBLIC_KEY}" "${PREP_PGP_EMAIL}"
printf 'Wrote plaintext comparison blobs to %s/prep_blob_{1..5}.txt\n' "${PGP_DIR}"
printf 'Wrote encrypted armored messages to %s/prep_blob_{1..5}.asc\n' "${PGP_DIR}"
