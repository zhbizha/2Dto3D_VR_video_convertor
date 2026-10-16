namespace VRify.Core;

/// <summary>
/// Temporal smoothing of depth maps, guided by RGB similarity between frames.
///
/// Rationale: monocular depth models jitter slightly frame-to-frame — at object
/// boundaries this jitter shows as "swimming" edges in stereo. A naive EMA causes
/// ghost trails on moving objects instead. So the blend weight is decided PER PIXEL
/// by color change:
///   - color unchanged (static scene) → trust history strongly → kills jitter;
///   - color changed (motion edge)   → trust current frame → no ghosting.
///
/// Uses two ping-ponged buffers (no per-frame allocation) and runs row-parallel.
/// Not thread-safe against concurrent calls — call from a single stage thread.
/// </summary>
public sealed class DepthSmoother
{
    private readonly float _alphaStatic;   // weight of NEW frame where scene is static
    private readonly float _alphaMotion;   // weight of NEW frame where color changed
    private float[]? _prev;
    private float[]? _bufA;
    private float[]? _bufB;
    private byte[]? _prevRgb;

    public DepthSmoother(float temporalSmoothing = 0.75f)
    {
        float s = Math.Clamp(temporalSmoothing, 0f, 0.95f);
        _alphaStatic = 1f - s;             // heavy history in static areas
        _alphaMotion = MathF.Min(0.9f, _alphaStatic + 0.6f); // mostly fresh in motion
    }

    /// <summary>depth: this frame's map; rgb: this frame (motion guide); width: row length.</summary>
    public float[] Smooth(float[] depth, byte[] rgb, int width)
    {
        int n = depth.Length;
        if (_prev == null || _prev.Length != n || _prevRgb == null || _prevRgb.Length != rgb.Length
            || _alphaStatic >= 0.999f)
        {
            // First frame: initialize history with a copy, return input unchanged.
            _bufA ??= new float[n];
            Array.Copy(depth, _bufA, n);
            _prev = _bufA;
            _prevRgb ??= new byte[rgb.Length];
            Buffer.BlockCopy(rgb, 0, _prevRgb, 0, rgb.Length);
            return depth;
        }

        // Ping-pong: write into the buffer that is NOT the one being read.
        if (_bufA == null || _bufA.Length != n) _bufA = new float[n];
        if (_bufB == null || _bufB.Length != n) _bufB = new float[n];
        float[] dst = ReferenceEquals(_prev, _bufA) ? _bufB : _bufA;

        var prev = _prev;
        var prevRgb = _prevRgb;
        int h = n / width;

        // Color-change thresholds on max channel delta (0..255).
        const float chgStatic = 6f;
        const float chgMotion = 24f;
        float aStat = _alphaStatic, aMot = _alphaMotion;

        Parallel.For(0, h, y =>
        {
            int i0 = y * width, p0 = i0 * 3;
            for (int i = i0, p = p0; i < i0 + width; i++, p += 3)
            {
                float dr = MathF.Abs(rgb[p] - prevRgb[p]);
                float dg = MathF.Abs(rgb[p + 1] - prevRgb[p + 1]);
                float db = MathF.Abs(rgb[p + 2] - prevRgb[p + 2]);
                float chg = MathF.Max(dr, MathF.Max(dg, db));

                // staticness: 1 = static, 0 = moving
                float staticness = chg <= chgStatic ? 1f
                                 : chg >= chgMotion ? 0f
                                 : 1f - (chg - chgStatic) / (chgMotion - chgStatic);

                float alpha = aMot + (aStat - aMot) * staticness;
                dst[i] = prev[i] + (depth[i] - prev[i]) * alpha;
            }
        });

        // Snapshot this frame's colors for next comparison.
        Buffer.BlockCopy(rgb, 0, prevRgb, 0, rgb.Length);
        _prev = dst;
        return dst;
    }
}
