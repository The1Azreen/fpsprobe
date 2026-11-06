//! Writes HDR10 content light levels (MaxCLL / MaxFALL) into a finished MP4 or MOV.
//!
//! The levels are only known once the whole video has been measured, which is
//! while it is being rendered, and FFmpeg takes no such metadata for raw video
//! input. So the `clli` box is added to the video's sample entry afterwards.
//! FFmpeg writes the index (`moov`) after the media data, so growing it moves no
//! sample and leaves every chunk offset as it was.

use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Size of the fixed part of a visual sample entry, after its box header.
const VISUAL_SAMPLE_ENTRY: usize = 78;

/// A box inside a buffer: where it starts, its header length and its total size.
#[derive(Clone, Copy, Debug)]
struct Atom {
    offset: usize,
    header: usize,
    size: usize,
    kind: [u8; 4],
}

impl Atom {
    fn end(&self) -> usize {
        self.offset + self.size
    }

    fn body(&self) -> usize {
        self.offset + self.header
    }
}

/// Boxes laid out one after another in `buf[start..end]`.
fn atoms(buf: &[u8], start: usize, end: usize) -> Option<Vec<Atom>> {
    let mut out = Vec::new();
    let mut at = start;
    while at + 8 <= end {
        let size32 = u32::from_be_bytes(buf[at..at + 4].try_into().ok()?) as usize;
        let kind: [u8; 4] = buf[at + 4..at + 8].try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, end - at),
            1 => (16, u64::from_be_bytes(buf.get(at + 8..at + 16)?.try_into().ok()?) as usize),
            n => (8, n),
        };
        if size < header || at + size > end {
            return None;
        }
        out.push(Atom { offset: at, header, size, kind });
        at += size;
    }
    Some(out)
}

fn child(buf: &[u8], parent: &Atom, kind: &[u8; 4]) -> Option<Atom> {
    atoms(buf, parent.body(), parent.end())?.into_iter().find(|a| &a.kind == kind)
}

/// Adds `delta` to a box's size field.
fn grow(buf: &mut [u8], atom: &Atom, delta: usize) {
    if atom.header == 16 {
        let f = &mut buf[atom.offset + 8..atom.offset + 16];
        let v = u64::from_be_bytes(f.try_into().unwrap()) + delta as u64;
        f.copy_from_slice(&v.to_be_bytes());
    } else {
        let f = &mut buf[atom.offset..atom.offset + 4];
        let v = u32::from_be_bytes(f.try_into().unwrap()) + delta as u32;
        f.copy_from_slice(&v.to_be_bytes());
    }
}

/// Sets MaxCLL / MaxFALL on the HEVC track of the `moov` box in `moov`
/// (the whole box, header included). False when there is no HEVC track.
fn patch_moov(moov: &mut Vec<u8>, max_cll: u16, max_fall: u16) -> bool {
    let Some(root) = atoms(moov, 0, moov.len()).and_then(|a| a.first().copied()) else {
        return false;
    };
    let payload = [max_cll.to_be_bytes(), max_fall.to_be_bytes()].concat();
    for trak in atoms(moov, root.body(), root.end()).unwrap_or_default().into_iter().filter(|a| &a.kind == b"trak") {
        let mut path = vec![root, trak];
        for kind in [b"mdia", b"minf", b"stbl", b"stsd"] {
            match child(moov, path.last().unwrap(), kind) {
                Some(a) => path.push(a),
                None => break,
            }
        }
        let stsd = *path.last().unwrap();
        if &stsd.kind != b"stsd" {
            continue;
        }
        // stsd: version/flags and an entry count, then the sample entries.
        let Some(entry) = atoms(moov, stsd.body() + 8, stsd.end()).and_then(|a| a.first().copied()) else {
            continue;
        };
        if !matches!(&entry.kind, b"hvc1" | b"hev1") || entry.size < entry.header + VISUAL_SAMPLE_ENTRY {
            continue;
        }
        let inner = entry.body() + VISUAL_SAMPLE_ENTRY;
        let existing = atoms(moov, inner, entry.end()).and_then(|a| a.into_iter().find(|a| &a.kind == b"clli"));
        match existing {
            Some(clli) if clli.size == 12 => moov[clli.body()..clli.end()].copy_from_slice(&payload),
            _ => {
                let mut clli = 12u32.to_be_bytes().to_vec();
                clli.extend_from_slice(b"clli");
                clli.extend_from_slice(&payload);
                moov.splice(entry.end()..entry.end(), clli);
                for a in path.iter().chain([&entry]) {
                    grow(moov, a, 12);
                }
            }
        }
        return true;
    }
    false
}

/// Writes MaxCLL / MaxFALL (nits) into the HEVC track of an MP4 or MOV file.
/// Returns false, leaving the file untouched, when it has no HEVC track or its
/// index isn't at the end (where FFmpeg puts it).
pub fn write_content_light_level(path: &Path, max_cll: f32, max_fall: f32) -> Result<bool> {
    let level = |v: f32| v.round().clamp(0.0, u16::MAX as f32) as u16;
    let mut file = OpenOptions::new().read(true).write(true).open(path).context("opening the rendered video")?;
    let len = file.metadata()?.len();

    // Top-level boxes: only their headers are read.
    let mut at = 0u64;
    let mut moov = None;
    while at + 8 <= len {
        let mut head = [0u8; 16];
        file.seek(SeekFrom::Start(at))?;
        file.read_exact(&mut head[..8])?;
        let size = match u32::from_be_bytes(head[..4].try_into().unwrap()) {
            0 => len - at,
            1 => {
                file.read_exact(&mut head[8..])?;
                u64::from_be_bytes(head[8..].try_into().unwrap())
            }
            n => n as u64,
        };
        if size < 8 {
            return Ok(false);
        }
        if &head[4..8] == b"moov" {
            moov = Some((at, size));
        }
        at += size;
    }
    let Some((offset, size)) = moov.filter(|&(o, s)| o + s == len) else {
        return Ok(false);
    };

    let mut buf = vec![0u8; size as usize];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut buf)?;
    if !patch_moov(&mut buf, level(max_cll), level(max_fall)) {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&buf)?;
    file.flush()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn moov_with(entry_children: &[u8]) -> Vec<u8> {
        let entry = atom(b"hvc1", &[&[0u8; VISUAL_SAMPLE_ENTRY][..], entry_children].concat());
        let stsd = atom(b"stsd", &[&[0, 0, 0, 0, 0, 0, 0, 1][..], &entry].concat());
        let stbl = atom(b"stbl", &stsd);
        let minf = atom(b"minf", &stbl);
        let mdia = atom(b"mdia", &minf);
        let trak = atom(b"trak", &mdia);
        atom(b"moov", &[atom(b"mvhd", &[0; 100]), trak].concat())
    }

    /// The clli payload in a patched moov, and whether every size still adds up.
    fn read_clli(moov: &[u8]) -> Option<(u16, u16)> {
        let at = moov.windows(4).position(|w| w == b"clli")?;
        assert_eq!(&moov[at - 4..at], &12u32.to_be_bytes());
        assert_eq!(u32::from_be_bytes(moov[..4].try_into().unwrap()) as usize, moov.len());
        Some((
            u16::from_be_bytes(moov[at + 4..at + 6].try_into().unwrap()),
            u16::from_be_bytes(moov[at + 6..at + 8].try_into().unwrap()),
        ))
    }

    #[test]
    fn adds_clli_and_grows_parents() {
        let mut moov = moov_with(&atom(b"colr", b"nclx\0\x09\0\x10\0\x09\0"));
        let before = moov.len();
        assert!(patch_moov(&mut moov, 2567, 1449));
        assert_eq!(moov.len(), before + 12);
        assert_eq!(read_clli(&moov), Some((2567, 1449)));
        // Every box along the path parses with its new size.
        let root = atoms(&moov, 0, moov.len()).unwrap()[0];
        let trak = child(&moov, &root, b"trak").unwrap();
        assert_eq!(trak.end(), moov.len());
    }

    #[test]
    fn replaces_existing_clli() {
        let mut moov = moov_with(&atom(b"clli", &[0, 1, 0, 2]));
        let before = moov.len();
        assert!(patch_moov(&mut moov, 1000, 400));
        assert_eq!(moov.len(), before);
        assert_eq!(read_clli(&moov), Some((1000, 400)));
    }

    #[test]
    fn ignores_other_codecs() {
        let entry = atom(b"avc1", &[0u8; VISUAL_SAMPLE_ENTRY]);
        let stsd = atom(b"stsd", &[&[0, 0, 0, 0, 0, 0, 0, 1][..], &entry].concat());
        let mut moov = atom(b"moov", &atom(b"trak", &atom(b"mdia", &atom(b"minf", &atom(b"stbl", &stsd)))));
        assert!(!patch_moov(&mut moov, 1000, 400));
    }
}
