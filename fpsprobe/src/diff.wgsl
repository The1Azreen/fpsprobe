// Per-frame difference statistics between two 8-bit luma frames.
//
// Frames are uploaded as packed u32 words (4 pixels each). One workgroup covers
// one 64x64 pixel tile: 16 threads across (16 words = 64 px), 16 threads down
// with 4 rows each. For every tile we write the sum of absolute differences;
// for the whole frame we count pixels whose difference exceeds `compat_thr`
// (trdrop's original "pixel difference" rule).

struct Params {
    width_words: u32,
    height: u32,
    tiles_x: u32,
    compat_thr: u32,
    base: u32,      // offset of this frame's slot in `stats`
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prev: array<u32>;
@group(0) @binding(2) var<storage, read> cur: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> wg_sad: atomic<u32>;
var<workgroup> wg_compat: atomic<u32>;

fn absdiff(a: u32, b: u32) -> u32 {
    return max(a, b) - min(a, b);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        atomicStore(&wg_sad, 0u);
        atomicStore(&wg_compat, 0u);
    }
    workgroupBarrier();

    var sad = 0u;
    var compat = 0u;
    let wx = wid.x * 16u + lid.x;
    if (wx < p.width_words) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let y = wid.y * 64u + lid.y * 4u + k;
            if (y < p.height) {
                let idx = y * p.width_words + wx;
                let a = prev[idx];
                let b = cur[idx];
                for (var s = 0u; s < 32u; s = s + 8u) {
                    let d = absdiff((a >> s) & 0xffu, (b >> s) & 0xffu);
                    sad = sad + d;
                    compat = compat + select(0u, 1u, d > p.compat_thr);
                }
            }
        }
    }
    if (sad > 0u) { atomicAdd(&wg_sad, sad); }
    if (compat > 0u) { atomicAdd(&wg_compat, compat); }
    workgroupBarrier();

    if (li == 0u) {
        let tile = wid.y * p.tiles_x + wid.x;
        atomicStore(&stats[p.base + 1u + tile], atomicLoad(&wg_sad));
        let c = atomicLoad(&wg_compat);
        if (c > 0u) { atomicAdd(&stats[p.base], c); }
    }
}
