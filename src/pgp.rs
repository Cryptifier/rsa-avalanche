/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{error::Error, fs, path::Path};

use num_bigint::BigUint;
use serde::Serialize;

const CRC24_INIT: u32 = 0xB704CE;
const CRC24_POLY: u32 = 0x1864CFB;

/// Parsed RSA public key extracted from an OpenPGP public-key packet.
#[derive(Debug, Clone)]
pub struct ImportedRsaPublicKey {
    /// RSA modulus `n` extracted from the packet MPI values.
    pub modulus: BigUint,
    /// RSA public exponent `e` extracted from the packet MPI values.
    pub public_exponent: BigUint,
    /// Fully parsed OpenPGP file representation suitable for YAML serialization.
    pub parsed_file: PgpFileYaml,
}

/// Serializable YAML representation of an OpenPGP file.
#[derive(Debug, Clone, Serialize)]
pub struct PgpFileYaml {
    /// Stable file-format identifier for unpacked OpenPGP files.
    pub format: String,
    /// Human-readable container family.
    pub container: String,
    /// Whether the input file was ASCII armored or raw packet bytes.
    pub source_kind: String,
    /// Original on-disk file contents encoded as lowercase hex bytes.
    pub source_bytes_hex: String,
    /// Decoded OpenPGP packet stream encoded as lowercase hex bytes.
    pub decoded_bytes_hex: String,
    /// ASCII armor details when the source file used radix-64 armor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub armor: Option<PgpArmorYaml>,
    /// Number of parsed packets in the decoded packet stream.
    pub packet_count: usize,
    /// Parsed OpenPGP packets, including raw bytes and selected decoded fields.
    pub packets: Vec<PgpPacketYaml>,
}

/// Serializable metadata for one ASCII-armored OpenPGP block.
#[derive(Debug, Clone, Serialize)]
pub struct PgpArmorYaml {
    /// Text between the `BEGIN`/`END` armor markers.
    pub block_type: String,
    /// Full `-----BEGIN ...-----` line from the source text.
    pub begin_line: String,
    /// Full `-----END ...-----` line from the source text.
    pub end_line: String,
    /// Armor header lines preserved in source order.
    pub headers: Vec<PgpArmorHeaderYaml>,
    /// Optional radix-64 CRC24 checksum line and verification result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<PgpArmorChecksumYaml>,
}

/// One ASCII armor header line.
#[derive(Debug, Clone, Serialize)]
pub struct PgpArmorHeaderYaml {
    /// Header name before the `:`.
    pub name: String,
    /// Header value after the `:`.
    pub value: String,
}

/// Parsed ASCII armor checksum metadata.
#[derive(Debug, Clone, Serialize)]
pub struct PgpArmorChecksumYaml {
    /// Base64-encoded CRC24 text from the armor checksum line.
    pub encoded: String,
    /// Decoded CRC24 value rendered as lowercase hex bytes.
    pub crc24_hex: String,
    /// Whether the checksum matched the decoded radix-64 payload.
    pub matches_payload: bool,
}

/// Serializable representation of one OpenPGP packet.
#[derive(Debug, Clone, Serialize)]
pub struct PgpPacketYaml {
    /// One-based packet index within the decoded packet stream.
    pub packet_index: usize,
    /// Zero-based byte offset of the packet within the decoded packet stream.
    pub packet_offset: usize,
    /// Numeric OpenPGP packet tag.
    pub packet_tag: u8,
    /// Human-readable packet tag label.
    pub packet_name: String,
    /// Packet header encoding family.
    pub header_format: String,
    /// Exact packet header bytes encoded as lowercase hex.
    pub header_bytes_hex: String,
    /// Exact packet body bytes encoded as lowercase hex.
    pub body_bytes_hex: String,
    /// Exact full packet bytes encoded as lowercase hex.
    pub packet_bytes_hex: String,
    /// Packet body length after reassembling any partial chunks.
    pub body_length: usize,
    /// Reassembled partial chunk lengths when the packet used partial-body encoding.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub partial_body_lengths: Vec<usize>,
    /// Known packet-body parsing error while retaining the raw packet bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,
    /// Decoded public-key packet metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key: Option<PgpPublicKeyPacketYaml>,
    /// Decoded public-key-encrypted session-key metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key_encrypted_session_key: Option<PgpPkeskPacketYaml>,
    /// Decoded literal-data packet metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub literal_data: Option<PgpLiteralDataPacketYaml>,
    /// Decoded user ID packet metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<PgpUserIdPacketYaml>,
    /// Decoded compressed-data packet metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compressed_data: Option<PgpCompressedPacketYaml>,
    /// Decoded encrypted-data packet metadata when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_data: Option<PgpEncryptedDataPacketYaml>,
}

/// Parsed OpenPGP public-key packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpPublicKeyPacketYaml {
    /// Whether the packet was a primary key or a subkey packet.
    pub role: String,
    /// OpenPGP public-key packet version.
    pub version: u8,
    /// Packet creation timestamp.
    pub created_unix: u32,
    /// Optional validity period in days for legacy version 2/3 packets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validity_days: Option<u16>,
    /// Numeric public-key algorithm identifier.
    pub public_key_algorithm_id: u8,
    /// Human-readable public-key algorithm label.
    pub public_key_algorithm: String,
    /// RSA modulus `n` rendered in decimal when the packet carries an RSA key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modulus: Option<String>,
    /// RSA public exponent `e` rendered in decimal when the packet carries an RSA key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_exponent: Option<String>,
    /// RSA modulus width in bits when the packet carries an RSA key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modulus_bits: Option<u64>,
}

/// Parsed OpenPGP public-key-encrypted session-key packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpPkeskPacketYaml {
    /// Packet version.
    pub version: u8,
    /// Target recipient key ID as hex bytes.
    pub key_id_hex: String,
    /// Numeric public-key algorithm identifier.
    pub public_key_algorithm_id: u8,
    /// Human-readable public-key algorithm label.
    pub public_key_algorithm: String,
    /// RSA-encrypted session-key MPI when the packet uses RSA.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rsa_encrypted_session_key: Option<PgpMpiYaml>,
    /// Remaining body bytes not decoded into structured fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remaining_bytes_hex: Option<String>,
}

/// Parsed OpenPGP literal-data packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpLiteralDataPacketYaml {
    /// Literal-data format byte rendered as a printable label when possible.
    pub data_format: String,
    /// Original filename interpreted as UTF-8 when possible.
    pub filename: String,
    /// Original filename bytes encoded as lowercase hex.
    pub filename_hex: String,
    /// Literal-data timestamp.
    pub created_unix: u32,
    /// Literal payload bytes encoded as lowercase hex.
    pub payload_hex: String,
}

/// Parsed OpenPGP user ID packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpUserIdPacketYaml {
    /// User ID interpreted as UTF-8 when possible.
    pub value: String,
    /// Raw user ID bytes encoded as lowercase hex.
    pub value_hex: String,
}

/// Parsed OpenPGP compressed-data packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpCompressedPacketYaml {
    /// Numeric compression algorithm identifier.
    pub compression_algorithm_id: u8,
    /// Human-readable compression algorithm label.
    pub compression_algorithm: String,
    /// Compressed payload bytes encoded as lowercase hex.
    pub compressed_payload_hex: String,
}

/// Parsed OpenPGP encrypted-data packet fields.
#[derive(Debug, Clone, Serialize)]
pub struct PgpEncryptedDataPacketYaml {
    /// Human-readable encrypted packet kind.
    pub kind: String,
    /// Optional version byte when the packet format carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u8>,
    /// Remaining encrypted payload bytes encoded as lowercase hex.
    pub payload_hex: String,
}

/// Parsed OpenPGP multi-precision integer.
#[derive(Debug, Clone, Serialize)]
pub struct PgpMpiYaml {
    /// Declared MPI bit length.
    pub bit_length: u16,
    /// Raw MPI bytes encoded as lowercase hex.
    pub bytes_hex: String,
    /// MPI value rendered in decimal.
    pub value_decimal: String,
}

#[derive(Debug, Clone)]
struct ParsedPacket {
    packet_index: usize,
    packet_offset: usize,
    tag: u8,
    header_format: &'static str,
    header_bytes: Vec<u8>,
    body_bytes: Vec<u8>,
    packet_bytes: Vec<u8>,
    partial_body_lengths: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ArmorParse {
    block_type: String,
    begin_line: String,
    end_line: String,
    headers: Vec<PgpArmorHeaderYaml>,
    checksum: Option<PgpArmorChecksumYaml>,
    decoded_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ParsedMpi {
    bit_length: u16,
    bytes: Vec<u8>,
    value: BigUint,
}

/// Reads an OpenPGP file from disk and converts it into the `pgp-file-v1` YAML shape.
///
/// # Parameters
/// - `path`: Filesystem path to an ASCII-armored or binary OpenPGP file.
///
/// # Returns
/// - `Result<PgpFileYaml, Box<dyn Error>>`: Parsed OpenPGP file metadata and packet contents.
///
/// # Expected Output
/// - Reads the input file from disk and returns a serializable YAML document; no stdout/stderr output.
pub fn parse_openpgp_file_path(path: &Path) -> Result<PgpFileYaml, Box<dyn Error>> {
    let source_bytes = fs::read(path)?;
    parse_openpgp_bytes(&source_bytes)
}

/// Imports the first RSA public key found in an OpenPGP public-key file.
///
/// # Parameters
/// - `path`: Filesystem path to an ASCII-armored or binary OpenPGP public-key file.
///
/// # Returns
/// - `Result<ImportedRsaPublicKey, Box<dyn Error>>`: Extracted RSA modulus/exponent plus the parsed file view.
///
/// # Expected Output
/// - Reads the input file from disk and returns the extracted RSA public key; no stdout/stderr output.
pub fn import_rsa_public_key_from_pgp_path(
    path: &Path,
) -> Result<ImportedRsaPublicKey, Box<dyn Error>> {
    let parsed_file = parse_openpgp_file_path(path)?;
    for packet in &parsed_file.packets {
        if let Some(public_key) = &packet.public_key {
            let modulus = match public_key.modulus.as_deref() {
                Some(value) => value.parse::<BigUint>()?,
                None => continue,
            };
            let public_exponent = match public_key.public_exponent.as_deref() {
                Some(value) => value.parse::<BigUint>()?,
                None => continue,
            };
            return Ok(ImportedRsaPublicKey {
                modulus,
                public_exponent,
                parsed_file,
            });
        }
    }

    Err(format!("no RSA public-key packet found in {}", path.display()).into())
}

/// Parses an OpenPGP byte slice into YAML-friendly packet metadata.
///
/// # Parameters
/// - `source_bytes`: Exact file contents from disk.
///
/// # Returns
/// - `Result<PgpFileYaml, Box<dyn Error>>`: Parsed OpenPGP file metadata and packet contents.
///
/// # Expected Output
/// - Returns a serializable YAML document without writing files or printing output.
pub fn parse_openpgp_bytes(source_bytes: &[u8]) -> Result<PgpFileYaml, Box<dyn Error>> {
    let source_kind = if looks_like_ascii_armor(source_bytes) {
        "ascii_armor".to_string()
    } else {
        "binary".to_string()
    };
    let (decoded_bytes, armor) = if source_kind == "ascii_armor" {
        let text = std::str::from_utf8(source_bytes)
            .map_err(|err| format!("OpenPGP armor is not valid UTF-8: {err}"))?;
        let armor = parse_ascii_armor_block(text)?;
        (armor.decoded_bytes.clone(), Some(armor))
    } else {
        (source_bytes.to_vec(), None)
    };

    let parsed_packets = parse_packets(&decoded_bytes)?;
    let packets = parsed_packets
        .iter()
        .map(build_packet_yaml)
        .collect::<Vec<_>>();

    Ok(PgpFileYaml {
        format: "pgp-file-v1".to_string(),
        container: "OpenPGP".to_string(),
        source_kind,
        source_bytes_hex: hex::encode(source_bytes),
        decoded_bytes_hex: hex::encode(&decoded_bytes),
        armor: armor.map(|parsed| PgpArmorYaml {
            block_type: parsed.block_type,
            begin_line: parsed.begin_line,
            end_line: parsed.end_line,
            headers: parsed.headers,
            checksum: parsed.checksum,
        }),
        packet_count: packets.len(),
        packets,
    })
}

/// Detects whether the byte slice appears to be an ASCII-armored OpenPGP block.
///
/// # Parameters
/// - `source_bytes`: Exact file contents from disk.
///
/// # Returns
/// - `bool`: `true` when the bytes begin with an OpenPGP `BEGIN` armor marker.
///
/// # Expected Output
/// - Returns a boolean classification without side effects.
fn looks_like_ascii_armor(source_bytes: &[u8]) -> bool {
    let trimmed = String::from_utf8_lossy(source_bytes);
    trimmed.trim_start().starts_with("-----BEGIN PGP ")
}

/// Parses one ASCII-armored OpenPGP block and decodes its radix-64 payload.
///
/// # Parameters
/// - `text`: UTF-8 text containing one armored OpenPGP block.
///
/// # Returns
/// - `Result<ArmorParse, Box<dyn Error>>`: Parsed armor metadata plus decoded binary payload.
///
/// # Expected Output
/// - Returns parsed armor data without stdout/stderr output.
fn parse_ascii_armor_block(text: &str) -> Result<ArmorParse, Box<dyn Error>> {
    let mut lines = text.lines().peekable();
    let begin_line = lines
        .find(|line| !line.trim().is_empty())
        .ok_or("missing OpenPGP armor begin line")?
        .trim()
        .to_string();
    let block_type = begin_line
        .strip_prefix("-----BEGIN ")
        .and_then(|value| value.strip_suffix("-----"))
        .ok_or("invalid OpenPGP armor begin line")?
        .to_string();

    let mut headers = Vec::new();
    while let Some(line) = lines.peek().copied() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            lines.next();
            break;
        }
        if !trimmed.contains(':') {
            break;
        }
        let line = lines.next().unwrap_or_default();
        let mut parts = line.splitn(2, ':');
        let name = parts.next().unwrap_or_default().trim().to_string();
        let value = parts.next().unwrap_or_default().trim().to_string();
        headers.push(PgpArmorHeaderYaml { name, value });
    }

    let mut payload_base64 = String::new();
    let mut checksum_encoded = None;
    let mut end_line = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("-----END ") {
            end_line = Some(trimmed.to_string());
            break;
        }
        if let Some(checksum) = trimmed.strip_prefix('=') {
            checksum_encoded = Some(checksum.to_string());
            continue;
        }
        payload_base64.push_str(trimmed);
    }

    let end_line = end_line.ok_or("missing OpenPGP armor end line")?;
    let expected_end = format!("-----END {block_type}-----");
    if end_line != expected_end {
        return Err(
            format!("OpenPGP armor end line {end_line} does not match {expected_end}").into(),
        );
    }

    let decoded_bytes = decode_base64(&payload_base64)?;
    let checksum = checksum_encoded
        .map(|encoded| {
            let checksum_bytes = decode_base64(&encoded)?;
            if checksum_bytes.len() != 3 {
                return Err("OpenPGP armor checksum must decode to exactly 3 bytes".into());
            }
            let computed = crc24(&decoded_bytes).to_be_bytes();
            let matches_payload = checksum_bytes == computed[1..4];
            if !matches_payload {
                return Err("OpenPGP armor checksum does not match the decoded payload".into());
            }
            Ok::<PgpArmorChecksumYaml, Box<dyn Error>>(PgpArmorChecksumYaml {
                encoded,
                crc24_hex: hex::encode(checksum_bytes.clone()),
                matches_payload,
            })
        })
        .transpose()?;

    Ok(ArmorParse {
        block_type,
        begin_line,
        end_line,
        headers,
        checksum,
        decoded_bytes,
    })
}

/// Decodes a standard base64 string used by OpenPGP radix-64 armor.
///
/// # Parameters
/// - `input`: Base64 text with optional ASCII whitespace and padding.
///
/// # Returns
/// - `Result<Vec<u8>, Box<dyn Error>>`: Decoded binary bytes.
///
/// # Expected Output
/// - Returns decoded bytes without stdout/stderr output.
fn decode_base64(input: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut sanitized = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if sanitized.is_empty() {
        return Ok(Vec::new());
    }
    while sanitized.len() % 4 != 0 {
        sanitized.push(b'=');
    }

    let mut output = Vec::with_capacity((sanitized.len() / 4) * 3);
    for chunk in sanitized.chunks(4) {
        if chunk.len() != 4 {
            return Err("invalid base64 quartet length".into());
        }
        let values = chunk
            .iter()
            .map(|byte| base64_value(*byte))
            .collect::<Option<Vec<_>>>()
            .ok_or("invalid base64 character in OpenPGP armor")?;
        let triple = [
            (values[0] << 2) | (values[1] >> 4),
            ((values[1] & 0x0f) << 4) | (values[2] >> 2),
            ((values[2] & 0x03) << 6) | values[3],
        ];
        output.push(triple[0]);
        if chunk[2] != b'=' {
            output.push(triple[1]);
        }
        if chunk[3] != b'=' {
            output.push(triple[2]);
        }
    }

    Ok(output)
}

/// Converts one base64 byte into its 6-bit value.
///
/// # Parameters
/// - `byte`: ASCII base64 byte.
///
/// # Returns
/// - `Option<u8>`: Six-bit base64 value or `Some(0)` for padding.
///
/// # Expected Output
/// - Returns a decoded sextet without side effects.
fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

/// Computes the OpenPGP CRC24 checksum for decoded radix-64 payload bytes.
///
/// # Parameters
/// - `data`: Decoded radix-64 payload bytes.
///
/// # Returns
/// - `u32`: CRC24 checksum in the low 24 bits.
///
/// # Expected Output
/// - Returns the checksum value without side effects.
fn crc24(data: &[u8]) -> u32 {
    let mut crc = CRC24_INIT;
    for &byte in data {
        crc ^= u32::from(byte) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if (crc & 0x1000000) != 0 {
                crc ^= CRC24_POLY;
            }
        }
    }
    crc & 0x00FF_FFFF
}

/// Parses all packets in an OpenPGP packet stream.
///
/// # Parameters
/// - `bytes`: Decoded OpenPGP packet bytes.
///
/// # Returns
/// - `Result<Vec<ParsedPacket>, Box<dyn Error>>`: Parsed packet stream in source order.
///
/// # Expected Output
/// - Returns packet metadata without stdout/stderr output.
fn parse_packets(bytes: &[u8]) -> Result<Vec<ParsedPacket>, Box<dyn Error>> {
    let mut packets = Vec::new();
    let mut offset = 0usize;
    let mut packet_index = 1usize;
    while offset < bytes.len() {
        let packet = parse_one_packet(bytes, offset, packet_index)?;
        offset = packet.packet_offset + packet.packet_bytes.len();
        packet_index += 1;
        packets.push(packet);
    }
    Ok(packets)
}

/// Parses one packet starting at the given byte offset.
///
/// # Parameters
/// - `bytes`: Decoded OpenPGP packet bytes.
/// - `offset`: Packet start offset within `bytes`.
/// - `packet_index`: One-based packet index.
///
/// # Returns
/// - `Result<ParsedPacket, Box<dyn Error>>`: Parsed packet with raw bytes and reassembled body.
///
/// # Expected Output
/// - Returns one parsed packet without stdout/stderr output.
fn parse_one_packet(
    bytes: &[u8],
    offset: usize,
    packet_index: usize,
) -> Result<ParsedPacket, Box<dyn Error>> {
    let first = *bytes
        .get(offset)
        .ok_or("missing packet header byte at requested offset")?;
    if (first & 0x80) == 0 {
        return Err(
            format!("OpenPGP packet at offset {offset} is missing the high-bit marker").into(),
        );
    }

    if (first & 0x40) != 0 {
        parse_new_packet(bytes, offset, packet_index)
    } else {
        parse_old_packet(bytes, offset, packet_index)
    }
}

fn parse_new_packet(
    bytes: &[u8],
    offset: usize,
    packet_index: usize,
) -> Result<ParsedPacket, Box<dyn Error>> {
    let tag = bytes[offset] & 0x3F;
    let mut cursor = offset + 1;
    let (initial_length, initial_length_bytes, initial_partial) = read_new_length(bytes, cursor)?;
    let header_end = cursor + initial_length_bytes;
    cursor = header_end;

    let mut partial_body_lengths = Vec::new();
    let mut body_bytes = Vec::new();
    if initial_partial {
        let mut current_length = initial_length;
        loop {
            let end = cursor
                .checked_add(current_length)
                .ok_or("OpenPGP partial packet length overflow")?;
            if end > bytes.len() {
                return Err(format!(
                    "OpenPGP partial packet at offset {offset} extends past the input"
                )
                .into());
            }
            body_bytes.extend_from_slice(&bytes[cursor..end]);
            partial_body_lengths.push(current_length);
            cursor = end;

            let (next_length, next_length_bytes, next_partial) = read_new_length(bytes, cursor)?;
            cursor += next_length_bytes;
            if next_partial {
                current_length = next_length;
                continue;
            }
            let final_end = cursor
                .checked_add(next_length)
                .ok_or("OpenPGP final packet length overflow")?;
            if final_end > bytes.len() {
                return Err(
                    format!("OpenPGP packet at offset {offset} extends past the input").into(),
                );
            }
            body_bytes.extend_from_slice(&bytes[cursor..final_end]);
            cursor = final_end;
            break;
        }
    } else {
        let end = cursor
            .checked_add(initial_length)
            .ok_or("OpenPGP packet length overflow")?;
        if end > bytes.len() {
            return Err(format!("OpenPGP packet at offset {offset} extends past the input").into());
        }
        body_bytes.extend_from_slice(&bytes[cursor..end]);
        cursor = end;
    }

    Ok(ParsedPacket {
        packet_index,
        packet_offset: offset,
        tag,
        header_format: "new",
        header_bytes: bytes[offset..header_end].to_vec(),
        body_bytes,
        packet_bytes: bytes[offset..cursor].to_vec(),
        partial_body_lengths,
    })
}

fn parse_old_packet(
    bytes: &[u8],
    offset: usize,
    packet_index: usize,
) -> Result<ParsedPacket, Box<dyn Error>> {
    let first = bytes[offset];
    let tag = (first >> 2) & 0x0F;
    let length_type = first & 0x03;
    let mut cursor = offset + 1;

    let body_end = match length_type {
        0 => {
            let len = usize::from(*bytes.get(cursor).ok_or("missing old-format length byte")?);
            cursor += 1;
            cursor + len
        }
        1 => {
            let len = usize::from(read_be_u16(bytes, cursor)?);
            cursor += 2;
            cursor + len
        }
        2 => {
            let len = usize::try_from(read_be_u32(bytes, cursor)?)
                .map_err(|_| "old-format packet length exceeds platform usize")?;
            cursor += 4;
            cursor + len
        }
        3 => bytes.len(),
        _ => unreachable!(),
    };

    if body_end > bytes.len() {
        return Err(format!("OpenPGP packet at offset {offset} extends past the input").into());
    }

    Ok(ParsedPacket {
        packet_index,
        packet_offset: offset,
        tag,
        header_format: "old",
        header_bytes: bytes[offset..cursor].to_vec(),
        body_bytes: bytes[cursor..body_end].to_vec(),
        packet_bytes: bytes[offset..body_end].to_vec(),
        partial_body_lengths: Vec::new(),
    })
}

fn read_new_length(bytes: &[u8], offset: usize) -> Result<(usize, usize, bool), Box<dyn Error>> {
    let first = *bytes
        .get(offset)
        .ok_or("missing new-format packet length byte")?;
    match first {
        0..=191 => Ok((usize::from(first), 1, false)),
        192..=223 => {
            let second = *bytes
                .get(offset + 1)
                .ok_or("missing second new-format packet length byte")?;
            let length = ((usize::from(first) - 192) << 8) + usize::from(second) + 192;
            Ok((length, 2, false))
        }
        224..=254 => Ok((1usize << usize::from(first & 0x1F), 1, true)),
        255 => {
            let length = usize::try_from(read_be_u32(bytes, offset + 1)?)
                .map_err(|_| "new-format packet length exceeds platform usize")?;
            Ok((length, 5, false))
        }
    }
}

fn read_be_u16(bytes: &[u8], offset: usize) -> Result<u16, Box<dyn Error>> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or("missing 2-byte big-endian integer")?;
    Ok(u16::from_be_bytes([slice[0], slice[1]]))
}

fn read_be_u32(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn Error>> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or("missing 4-byte big-endian integer")?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn build_packet_yaml(packet: &ParsedPacket) -> PgpPacketYaml {
    let mut yaml = PgpPacketYaml {
        packet_index: packet.packet_index,
        packet_offset: packet.packet_offset,
        packet_tag: packet.tag,
        packet_name: packet_tag_name(packet.tag).to_string(),
        header_format: packet.header_format.to_string(),
        header_bytes_hex: hex::encode(&packet.header_bytes),
        body_bytes_hex: hex::encode(&packet.body_bytes),
        packet_bytes_hex: hex::encode(&packet.packet_bytes),
        body_length: packet.body_bytes.len(),
        partial_body_lengths: packet.partial_body_lengths.clone(),
        parse_error: None,
        public_key: None,
        public_key_encrypted_session_key: None,
        literal_data: None,
        user_id: None,
        compressed_data: None,
        encrypted_data: None,
    };

    let parse_result = match packet.tag {
        1 => parse_pkesk_packet(&packet.body_bytes).map(|parsed| {
            yaml.public_key_encrypted_session_key = Some(parsed);
        }),
        6 => parse_public_key_packet(&packet.body_bytes, false).map(|parsed| {
            yaml.public_key = Some(parsed);
        }),
        8 => parse_compressed_packet(&packet.body_bytes).map(|parsed| {
            yaml.compressed_data = Some(parsed);
        }),
        9 => parse_encrypted_packet(&packet.body_bytes, "symmetrically_encrypted_data").map(
            |parsed| {
                yaml.encrypted_data = Some(parsed);
            },
        ),
        11 => parse_literal_data_packet(&packet.body_bytes).map(|parsed| {
            yaml.literal_data = Some(parsed);
        }),
        13 => parse_user_id_packet(&packet.body_bytes).map(|parsed| {
            yaml.user_id = Some(parsed);
        }),
        14 => parse_public_key_packet(&packet.body_bytes, true).map(|parsed| {
            yaml.public_key = Some(parsed);
        }),
        18 => parse_seipd_packet(&packet.body_bytes).map(|parsed| {
            yaml.encrypted_data = Some(parsed);
        }),
        20 => parse_aead_packet(&packet.body_bytes).map(|parsed| {
            yaml.encrypted_data = Some(parsed);
        }),
        _ => Ok(()),
    };
    if let Err(err) = parse_result {
        yaml.parse_error = Some(err);
    }

    yaml
}

fn parse_public_key_packet(body: &[u8], is_subkey: bool) -> Result<PgpPublicKeyPacketYaml, String> {
    if body.len() < 6 {
        return Err("public-key packet body is too short".to_string());
    }
    let version = body[0];
    let mut cursor = 1usize;
    let created_unix = read_be_u32(body, cursor).map_err(|err| err.to_string())?;
    cursor += 4;

    let validity_days = if matches!(version, 2 | 3) {
        let days = read_be_u16(body, cursor).map_err(|err| err.to_string())?;
        cursor += 2;
        Some(days)
    } else {
        None
    };

    let public_key_algorithm_id = *body
        .get(cursor)
        .ok_or_else(|| "public-key packet is missing the algorithm byte".to_string())?;
    cursor += 1;

    let mut modulus = None;
    let mut public_exponent = None;
    let mut modulus_bits = None;
    if matches!(public_key_algorithm_id, 1 | 2 | 3) {
        let (n, used_n) = parse_mpi(body, cursor).map_err(|err| err.to_string())?;
        cursor += used_n;
        let (e, _) = parse_mpi(body, cursor).map_err(|err| err.to_string())?;
        modulus_bits = Some(u64::from(n.bit_length));
        modulus = Some(n.value.to_string());
        public_exponent = Some(e.value.to_string());
    }

    Ok(PgpPublicKeyPacketYaml {
        role: if is_subkey {
            "public_subkey".to_string()
        } else {
            "public_key".to_string()
        },
        version,
        created_unix,
        validity_days,
        public_key_algorithm_id,
        public_key_algorithm: public_key_algorithm_name(public_key_algorithm_id).to_string(),
        modulus,
        public_exponent,
        modulus_bits,
    })
}

fn parse_pkesk_packet(body: &[u8]) -> Result<PgpPkeskPacketYaml, String> {
    if body.len() < 10 {
        return Err("public-key-encrypted session-key packet body is too short".to_string());
    }
    let version = body[0];
    let key_id_hex = hex::encode(&body[1..9]);
    let public_key_algorithm_id = body[9];
    let mut rsa_encrypted_session_key = None;
    let mut remaining_bytes_hex = None;

    if matches!(public_key_algorithm_id, 1 | 2 | 3) {
        if body.len() > 10 {
            let (mpi, consumed) = parse_mpi(body, 10).map_err(|err| err.to_string())?;
            rsa_encrypted_session_key = Some(PgpMpiYaml {
                bit_length: mpi.bit_length,
                bytes_hex: hex::encode(&mpi.bytes),
                value_decimal: mpi.value.to_string(),
            });
            if 10 + consumed < body.len() {
                remaining_bytes_hex = Some(hex::encode(&body[10 + consumed..]));
            }
        }
    } else if body.len() > 10 {
        remaining_bytes_hex = Some(hex::encode(&body[10..]));
    }

    Ok(PgpPkeskPacketYaml {
        version,
        key_id_hex,
        public_key_algorithm_id,
        public_key_algorithm: public_key_algorithm_name(public_key_algorithm_id).to_string(),
        rsa_encrypted_session_key,
        remaining_bytes_hex,
    })
}

fn parse_literal_data_packet(body: &[u8]) -> Result<PgpLiteralDataPacketYaml, String> {
    if body.len() < 6 {
        return Err("literal-data packet body is too short".to_string());
    }
    let data_format = literal_data_format_name(body[0]).to_string();
    let filename_len = usize::from(body[1]);
    let filename_end = 2usize
        .checked_add(filename_len)
        .ok_or_else(|| "literal-data filename length overflow".to_string())?;
    let timestamp_end = filename_end
        .checked_add(4)
        .ok_or_else(|| "literal-data timestamp length overflow".to_string())?;
    if timestamp_end > body.len() {
        return Err("literal-data packet body is truncated".to_string());
    }
    let filename_bytes = &body[2..filename_end];
    let created_unix = read_be_u32(body, filename_end).map_err(|err| err.to_string())?;
    Ok(PgpLiteralDataPacketYaml {
        data_format,
        filename: String::from_utf8_lossy(filename_bytes).to_string(),
        filename_hex: hex::encode(filename_bytes),
        created_unix,
        payload_hex: hex::encode(&body[timestamp_end..]),
    })
}

fn parse_user_id_packet(body: &[u8]) -> Result<PgpUserIdPacketYaml, String> {
    Ok(PgpUserIdPacketYaml {
        value: String::from_utf8_lossy(body).to_string(),
        value_hex: hex::encode(body),
    })
}

fn parse_compressed_packet(body: &[u8]) -> Result<PgpCompressedPacketYaml, String> {
    let algorithm_id = *body
        .first()
        .ok_or_else(|| "compressed-data packet body is too short".to_string())?;
    Ok(PgpCompressedPacketYaml {
        compression_algorithm_id: algorithm_id,
        compression_algorithm: compression_algorithm_name(algorithm_id).to_string(),
        compressed_payload_hex: hex::encode(&body[1..]),
    })
}

fn parse_encrypted_packet(
    body: &[u8],
    kind: &'static str,
) -> Result<PgpEncryptedDataPacketYaml, String> {
    Ok(PgpEncryptedDataPacketYaml {
        kind: kind.to_string(),
        version: None,
        payload_hex: hex::encode(body),
    })
}

fn parse_seipd_packet(body: &[u8]) -> Result<PgpEncryptedDataPacketYaml, String> {
    let version = *body
        .first()
        .ok_or_else(|| "SEIPD packet body is too short".to_string())?;
    Ok(PgpEncryptedDataPacketYaml {
        kind: "symmetrically_encrypted_integrity_protected_data".to_string(),
        version: Some(version),
        payload_hex: hex::encode(&body[1..]),
    })
}

fn parse_aead_packet(body: &[u8]) -> Result<PgpEncryptedDataPacketYaml, String> {
    let version = *body
        .first()
        .ok_or_else(|| "AEAD packet body is too short".to_string())?;
    Ok(PgpEncryptedDataPacketYaml {
        kind: "aead_encrypted_data".to_string(),
        version: Some(version),
        payload_hex: hex::encode(&body[1..]),
    })
}

fn parse_mpi(bytes: &[u8], offset: usize) -> Result<(ParsedMpi, usize), Box<dyn Error>> {
    let bit_length = read_be_u16(bytes, offset)?;
    let byte_len = usize::from(bit_length).div_ceil(8);
    let start = offset + 2;
    let end = start
        .checked_add(byte_len)
        .ok_or("OpenPGP MPI length overflow")?;
    let mpi_bytes = bytes.get(start..end).ok_or("truncated OpenPGP MPI")?;
    Ok((
        ParsedMpi {
            bit_length,
            bytes: mpi_bytes.to_vec(),
            value: BigUint::from_bytes_be(mpi_bytes),
        },
        2 + byte_len,
    ))
}

fn packet_tag_name(tag: u8) -> &'static str {
    match tag {
        1 => "Public-Key Encrypted Session Key",
        2 => "Signature",
        3 => "Symmetric-Key Encrypted Session Key",
        4 => "One-Pass Signature",
        5 => "Secret Key",
        6 => "Public Key",
        7 => "Secret Subkey",
        8 => "Compressed Data",
        9 => "Symmetrically Encrypted Data",
        10 => "Marker",
        11 => "Literal Data",
        12 => "Trust",
        13 => "User ID",
        14 => "Public Subkey",
        17 => "User Attribute",
        18 => "Symmetrically Encrypted Integrity Protected Data",
        19 => "Modification Detection Code",
        20 => "AEAD Encrypted Data",
        _ => "Unknown",
    }
}

fn public_key_algorithm_name(algorithm_id: u8) -> &'static str {
    match algorithm_id {
        1 => "RSA Encrypt or Sign",
        2 => "RSA Encrypt-Only",
        3 => "RSA Sign-Only",
        16 => "Elgamal Encrypt-Only",
        17 => "DSA",
        18 => "ECDH",
        19 => "ECDSA",
        21 => "Diffie-Hellman",
        22 => "EdDSA",
        _ => "Unknown",
    }
}

fn compression_algorithm_name(algorithm_id: u8) -> &'static str {
    match algorithm_id {
        0 => "Uncompressed",
        1 => "ZIP",
        2 => "ZLIB",
        3 => "BZip2",
        _ => "Unknown",
    }
}

fn literal_data_format_name(format_byte: u8) -> &'static str {
    match format_byte {
        b'b' => "binary",
        b't' => "text",
        b'u' => "utf8",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PUBLIC_KEY_ASC: &str = "-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: Test Suite\nComment: Minimal RSA sample\n\nxg0EEjRWeAEADAyhAAUR\n=wIfx\n-----END PGP PUBLIC KEY BLOCK-----\n";

    const TEST_MESSAGE_ASC: &str = "-----BEGIN PGP MESSAGE-----\nVersion: Test Suite\n\nwQ8DAQIDBAUGBwgBABUSNFbSCQHerb7vyv66vg==\n=qExN\n-----END PGP MESSAGE-----\n";

    #[test]
    fn test_parse_openpgp_public_key_armor_extracts_rsa_values() {
        let parsed = parse_openpgp_bytes(TEST_PUBLIC_KEY_ASC.as_bytes()).expect("parse public key");

        assert_eq!(parsed.format, "pgp-file-v1");
        assert_eq!(parsed.source_kind, "ascii_armor");
        assert_eq!(parsed.packet_count, 1);
        assert_eq!(
            parsed.armor.as_ref().expect("armor").block_type,
            "PGP PUBLIC KEY BLOCK"
        );
        assert!(
            parsed
                .armor
                .as_ref()
                .expect("armor")
                .checksum
                .as_ref()
                .expect("checksum")
                .matches_payload
        );

        let packet = parsed.packets.first().expect("packet");
        let public_key = packet.public_key.as_ref().expect("public key");
        assert_eq!(public_key.public_key_algorithm_id, 1);
        assert_eq!(public_key.modulus.as_deref(), Some("3233"));
        assert_eq!(public_key.public_exponent.as_deref(), Some("17"));
        assert_eq!(public_key.modulus_bits, Some(12));
    }

    #[test]
    fn test_import_rsa_public_key_from_pgp_path_reads_first_rsa_key() {
        let path = std::env::temp_dir().join(format!(
            "pgp_public_key_{}_{}.asc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::write(&path, TEST_PUBLIC_KEY_ASC).expect("write sample public key");

        let imported = import_rsa_public_key_from_pgp_path(&path).expect("import public key");
        assert_eq!(imported.modulus.to_string(), "3233");
        assert_eq!(imported.public_exponent.to_string(), "17");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_parse_openpgp_message_preserves_headers_and_packets() {
        let parsed = parse_openpgp_bytes(TEST_MESSAGE_ASC.as_bytes()).expect("parse message");

        assert_eq!(parsed.packet_count, 2);
        assert_eq!(parsed.armor.as_ref().expect("armor").headers.len(), 1);
        assert_eq!(parsed.packets[0].packet_tag, 1);
        assert_eq!(parsed.packets[1].packet_tag, 18);

        let pkesk = parsed.packets[0]
            .public_key_encrypted_session_key
            .as_ref()
            .expect("pkesk");
        assert_eq!(pkesk.key_id_hex, "0102030405060708");
        assert_eq!(
            pkesk
                .rsa_encrypted_session_key
                .as_ref()
                .expect("rsa mpi")
                .value_decimal,
            "1193046"
        );

        let encrypted = parsed.packets[1]
            .encrypted_data
            .as_ref()
            .expect("encrypted data");
        assert_eq!(encrypted.version, Some(1));
        assert_eq!(encrypted.payload_hex, "deadbeefcafebabe");
    }

    #[test]
    fn test_parse_openpgp_rejects_bad_crc24() {
        let invalid = TEST_PUBLIC_KEY_ASC.replace("=wIfx", "=AAAA");
        let err = parse_openpgp_bytes(invalid.as_bytes()).expect_err("crc mismatch should fail");
        assert!(
            err.to_string().contains("checksum")
                || err.to_string().contains("CRC")
                || err.to_string().contains("armor")
        );
    }
}
