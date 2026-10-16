namespace VRify.Core;

/// <summary>
/// Converts a 2D RGB frame + depth map into a side-by-side stereo pair.
///
/// Right view is synthesized by FORWARD warping: every source pixel is splatted to
/// x − disparity(depth) with a z-test, so occlusions resolve correctly (near geometry
/// properly overwrites far geometry instead of sampling the wrong object). Disocclusion
/// holes are filled with a pyramid push-pull inpaint.
/// </summary>
public sealed class StereoSynthesizer
{
    private readonly float _strength;     // max disparity in percent of width

    public StereoSynthesizer(float strengthPercent = 2.5f, float temporalSmoothing = 0f)
    {
        _strength = Math.Clamp(strengthPercent, 0f, 10f);
    }

    /// <summary>Capture the hole mask (before inpainting) for debugging. Off by default.</summary>
    public bool CaptureHoles { get; set; }

    /// <summary>
    /// Fill disocclusion holes with mirrored BACKGROUND pixels (default) instead of the
    /// legacy thin-row interpolation + push-pull blur. Exposed so the probe harness can
    /// A/B the two fills.
    /// </summary>
    public bool BackgroundFill { get; set; } = true;

    /// <summary>
    /// Variant of the background fill that mirrors in the destination domain and removes the
    /// doubled local background gradient, so the fill continues the background's slope across
    /// the hole instead of bouncing off the boundary (a bounce flips the slope and leaves a
    /// ridge at the hole plus a step at the silhouette).
    /// </summary>
    public bool TrendFill { get; set; }

    /// <summary>Hole mask from the last Synthesize call (null unless CaptureHoles was set).</summary>
    public bool[]? LastHoleMask { get; private set; }

    /// <summary>
    /// Produces a packed SBS frame (left | right) into <paramref name="sbs"/>.
    /// rgb: source frame; depth: normalized 0..1 with 1 = closest; both W*H.
    /// </summary>
    public void Synthesize(byte[] rgb, float[] depth, int width, int height, byte[] sbs)
    {
        int W = width, H = height;
        int n = W * H;
        int frameBytes = n * 3;
        float maxShift = _strength / 100f * W;
        // z-tolerance: pixels of the SAME surface blend (sub-pixel smoothness), but keep
        // it tight — a wide tolerance smears the depth-transition band into edge mush.
        float zTol = Math.Max(0.5f, maxShift * 0.03f);

        EnsureBuffers(n);
        Array.Clear(_acc, 0, frameBytes);
        Array.Clear(_wsum, 0, n);
        Array.Fill(_z, -1f);

        // --- Forward warp with bilinear splat + z-test accumulation ---
        // Right eye = camera moved right → scene shifts LEFT by disparity (near shifts more).
        // Rows are independent (horizontal-only disparity) → parallel over rows.
        int wLen = W;
        Parallel.For(0, H, y =>
        {
            int srcRow = y * wLen * 3;
            int maskRow = y * wLen;
            for (int xs = 0; xs < wLen; xs++)
            {
                int si3 = srcRow + xs * 3;
                float r = rgb[si3], g = rgb[si3 + 1], b = rgb[si3 + 2];
                float disp = depth[maskRow + xs] * maxShift;
                float xf = xs - disp;
                if (xf < -1f || xf > wLen) continue;

                int x0 = (int)xf;
                float t = xf - x0;
                int x1 = x0 + 1;

                if (x0 >= 0 && x0 < wLen)
                    Splat(maskRow + x0, disp, (1f - t), r, g, b, _acc, _wsum, _z, zTol);
                if (x1 >= 0 && x1 < wLen && t > 0f)
                    Splat(maskRow + x1, disp, t, r, g, b, _acc, _wsum, _z, zTol);
            }
        });

        // --- Resolve accumulation → byte image, mark unwritten as holes ---
        var right = new byte[frameBytes];
        var holeMask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            float w = _wsum[i];
            if (w <= 1e-6f) { holeMask[i] = true; continue; }
            int o = i * 3;
            float inv = 1f / w;
            right[o] = (byte)Math.Clamp(_acc[o] * inv, 0f, 255f);
            right[o + 1] = (byte)Math.Clamp(_acc[o + 1] * inv, 0f, 255f);
            right[o + 2] = (byte)Math.Clamp(_acc[o + 2] * inv, 0f, 255f);
        }

        LastHoleMask = CaptureHoles ? (bool[])holeMask.Clone() : null;

        if (holeMask.Contains(true))
        {
            // Thin vertical "comb" holes (1-4 px wide, from depth-edge rasterization)
            // → fill horizontally along the row: bilinear between nearest known left/right
            // neighbors. Physically correct for horizontal-only disparity and removes
            // the sawtooth fringes on silhouettes.
            //
            // Disocclusion holes must be filled with BACKGROUND, never with a blur that
            // averages the foreground in — that reads as a halo / rubber band on
            // silhouettes. FillHolesFromBackground mirrors the real background pixels
            // across the run's background boundary; anything it cannot reach (whole-row
            // holes, borders) still goes to the pyramid push-pull.
            if (BackgroundFill) FillHolesFromBackground(right, holeMask, rgb, _z, W, H, maxShift, TrendFill);
            else FillThinHolesHorizontal(right, holeMask, W, H, maxRun: 5);

            if (holeMask.Contains(true)) PushPullInpaint(right, holeMask, W, H);
            // NOTE: do NOT blur pixels adjacent to holes — those are the object
            // silhouette pixels themselves; blurring them every frame made all
            // object edges "swim"/melt. Fill pixels only (push-pull above).
        }

        // --- Pack SBS row-by-row (destination is twice as wide) ---
        int srcStride = W * 3;
        int dstStride = W * 6;
        for (int y = 0; y < H; y++)
        {
            Buffer.BlockCopy(rgb, y * srcStride, sbs, y * dstStride, srcStride);
            Buffer.BlockCopy(right, y * srcStride, sbs, y * dstStride + srcStride, srcStride);
        }
    }

    /// <summary>
    /// Fills every hole run that has a known pixel on at least one side of its row.
    ///
    /// In a horizontal-disparity warp a disocclusion always sits between a near region and
    /// a far region, and the far one is the revealed background. We therefore take the
    /// BACKGROUND side (the smaller disparity), map the run into that surface's source
    /// domain, and mirror the real background pixels about the run boundary. That continues
    /// the actual background texture; the old push-pull averaged the foreground into the
    /// hole and produced the blurred band that reads as a halo / rubber edge.
    ///
    /// Filled pixels are cleared from <paramref name="holes"/> so only genuinely
    /// unreachable holes remain for the pyramid inpaint.
    ///
    /// When <paramref name="trend"/> is set the mirror is taken in the DESTINATION domain
    /// (continuous with the background at the boundary by construction) and the doubled
    /// local background gradient is subtracted, so the fill carries the background's slope
    /// ACROSS the hole instead of bouncing off the boundary. A plain mirror reflects the
    /// slope, which lands a ridge at the boundary and a step at the silhouette whenever the
    /// background is not flat; the correction removes both while still mirroring the real
    /// background texture.
    /// </summary>
    private static unsafe void FillHolesFromBackground(
        byte[] img, bool[] holes, byte[] src, float[] z, int W, int H, float maxShift, bool trend)
    {
        // A disocclusion is at most one disparity jump wide, so a scan bounded by
        // ceil(maxShift)+2 always reaches the far side. The bound keeps the GPU port of
        // this routine (BgFill in GpuShaders.cs) and this CPU path behaviourally identical;
        // anything not reached (whole-row holes, borders) stays for the push-pull.
        int maxScan = (int)MathF.Ceiling(maxShift) + 2;
        fixed (byte* pImg = img, pSrc = src)
        fixed (bool* h = holes)
        fixed (float* pz = z)
        {
            for (int y = 0; y < H; y++)
            {
                int row = y * W;
                byte* srcRow = pSrc + row * 3;
                int x = 0;
                while (x < W)
                {
                    if (!h[row + x]) { x++; continue; }
                    int a = x;
                    while (x < W && h[row + x]) x++;
                    int b = x;                       // exclusive

                    // Nearest known disparity on each side (z[] holds the winning disparity).
                    float dl = float.NaN, dr = float.NaN;
                    for (int k = a - 1; k >= 0 && a - 1 - k <= maxScan; k--)
                        if (!h[row + k]) { dl = pz[row + k]; break; }
                    for (int k = b; k < W && k - b <= maxScan; k++)
                        if (!h[row + k]) { dr = pz[row + k]; break; }

                    float anchor, dispFar;
                    bool leftBg;
                    if (!float.IsNaN(dl) && (float.IsNaN(dr) || dl <= dr)) { anchor = a - 1; dispFar = dl; leftBg = true; }
                    else if (!float.IsNaN(dr)) { anchor = b; dispFar = dr; leftBg = false; }
                    else continue;                   // whole row is a hole → leave for push-pull

                    int an = (int)anchor;

                    // Trend variant: per-channel gradient of the background measured outward
                    // from the boundary over the width of the run. Subtracting twice this per
                    // unit of distance from the boundary turns the mirror's reflected slope back
                    // into a continuation of the background's own slope.
                    float g0 = 0f, g1 = 0f, g2 = 0f;
                    if (trend)
                    {
                        int span = Math.Max(1, b - a);
                        int outward = leftBg ? Math.Max(0, an - span) : Math.Min(W - 1, an + span);
                        int denom = Math.Abs(outward - an);
                        if (denom == 0) denom = 1;
                        byte* pa = pImg + (row + an) * 3;
                        byte* pb = pImg + (row + outward) * 3;
                        g0 = (pb[0] - pa[0]) / (float)denom;
                        g1 = (pb[1] - pa[1]) / (float)denom;
                        g2 = (pb[2] - pa[2]) / (float)denom;
                    }

                    for (int xd = a; xd < b; xd++)
                    {
                        byte* d = pImg + (row + xd) * 3;

                        if (trend)
                        {
                            // Destination-domain mirror; walk outward if that lands on a hole.
                            int mx = 2 * an - xd;
                            int step = leftBg ? -1 : 1;
                            for (int g = 0; g < maxScan && mx >= 0 && mx < W && h[row + mx]; g++) mx += step;
                            if (mx < 0 || mx >= W || h[row + mx]) continue;   // → push-pull
                            int dist = Math.Abs(xd - an);
                            byte* m = pImg + (row + mx) * 3;
                            d[0] = (byte)Math.Clamp(m[0] - 2f * g0 * dist, 0f, 255f);
                            d[1] = (byte)Math.Clamp(m[1] - 2f * g1 * dist, 0f, 255f);
                            d[2] = (byte)Math.Clamp(m[2] - 2f * g2 * dist, 0f, 255f);
                            h[row + xd] = false;
                            continue;
                        }

                        // Mirror the background about the boundary, in the source domain.
                        float xs = 2f * anchor + dispFar - xd;
                        if (xs < 0f) xs = -xs;
                        if (xs > W - 1) xs = 2f * (W - 1) - xs;
                        xs = Math.Clamp(xs, 0f, W - 1f);

                        int s0 = (int)xs;
                        int s1 = Math.Min(W - 1, s0 + 1);
                        float t = xs - s0;
                        byte* q0 = srcRow + s0 * 3;
                        byte* q1 = srcRow + s1 * 3;
                        d[0] = (byte)Math.Clamp(q0[0] + (q1[0] - q0[0]) * t, 0f, 255f);
                        d[1] = (byte)Math.Clamp(q0[1] + (q1[1] - q0[1]) * t, 0f, 255f);
                        d[2] = (byte)Math.Clamp(q0[2] + (q1[2] - q0[2]) * t, 0f, 255f);
                        h[row + xd] = false;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Fills thin hole runs (≤ maxRun px) per row by interpolating between the nearest
    /// known pixels on the left and right. Clears them from the mask so the pyramid
    /// inpaint only handles wide holes.
    /// </summary>
    private static unsafe void FillThinHolesHorizontal(byte[] img, bool[] holes, int W, int H, int maxRun)
    {
        fixed (byte* p = img)
        fixed (bool* h = holes)
        {
            for (int y = 0; y < H; y++)
            {
                int row = y * W;
                int x = 0;
                while (x < W)
                {
                    if (!h[row + x]) { x++; continue; }
                    int runStart = x;
                    while (x < W && h[row + x]) x++;
                    int runEnd = x;              // exclusive
                    int runLen = runEnd - runStart;
                    if (runLen > maxRun) continue;

                    bool hasLeft = runStart > 0;
                    bool hasRight = runEnd < W;
                    if (!hasLeft && !hasRight) continue;

                    byte* l = hasLeft ? p + (row + runStart - 1) * 3 : null;
                    byte* r = hasRight ? p + (row + runEnd) * 3 : null;
                    int span = (hasLeft ? 1 : 0) + (hasRight ? 1 : 0) + runLen - 1;
                    if (span <= 0) span = 1;

                    for (int i = 0; i < runLen; i++)
                    {
                        int distFromLeft = hasLeft ? i + 1 : 0;
                        float t = hasLeft && hasRight ? (float)distFromLeft / (span + 1)
                                : hasLeft ? 1f : 0f;
                        byte* dst = p + (row + runStart + i) * 3;
                        for (int c = 0; c < 3; c++)
                        {
                            float v;
                            if (hasLeft && hasRight)
                                v = l[c] + (r[c] - l[c]) * t;
                            else if (hasLeft)
                                v = l[c];
                            else
                                v = r[c];
                            dst[c] = (byte)Math.Clamp(v, 0f, 255f);
                        }
                        h[row + runStart + i] = false;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Bilinear splat of one source pixel contribution into an accumulation buffer.
    /// Z-test with tolerance: much-nearer pixels replace the buffer; comparable-depth
    /// pixels blend (gives smooth sub-pixel resampling instead of nearest-neighbor noise).
    /// Safe for parallel row-writes: each call touches only its own pixel index.
    /// </summary>
    private static void Splat(int di, float disp, float weight, float r, float g, float b,
        float[] acc, float[] wsum, float[] z, float zTol)
    {
        float zPrev = z[di];
        if (zPrev < 0f || disp > zPrev + zTol)
        {
            // New occupant is clearly nearer (or pixel empty) → replace
            z[di] = disp;
            wsum[di] = weight;
            int o = di * 3;
            acc[o] = r * weight;
            acc[o + 1] = g * weight;
            acc[o + 2] = b * weight;
        }
        else if (disp >= zPrev - zTol)
        {
            // Same surface → blend
            wsum[di] += weight;
            int o = di * 3;
            acc[o] += r * weight;
            acc[o + 1] += g * weight;
            acc[o + 2] += b * weight;
        }
        // else: occluded by clearly nearer geometry → drop
    }

    private float[] _acc = Array.Empty<float>();
    private float[] _wsum = Array.Empty<float>();
    private float[] _z = Array.Empty<float>();

    private void EnsureBuffers(int n)
    {
        if (_acc.Length != n * 3) _acc = new float[n * 3];
        if (_wsum.Length != n) _wsum = new float[n];
        if (_z.Length != n) _z = new float[n];
    }

    /// <summary>
    /// Pyramid push-pull inpaint: downscale (averaging known pixels only), then upscale
    /// back, blending known pixels with the smooth estimate. Soft, artifact-free fill
    /// for disocclusion strips.
    /// </summary>
    private static unsafe void PushPullInpaint(byte[] img, bool[] holes, int W, int H)
    {
        int levels = 1;
        while ((W >> levels) > 4 && (H >> levels) > 4) levels++;

        var rgbP = new float[levels][];
        var wP = new float[levels][];
        int w0 = W, h0 = H;
        rgbP[0] = new float[w0 * h0 * 3];
        wP[0] = new float[w0 * h0];
        fixed (byte* pImg = img)
        fixed (float* pR0 = rgbP[0], pW0 = wP[0])
        {
            for (int i = 0; i < w0 * h0; i++)
            {
                float known = holes[i] ? 0f : 1f;
                pW0[i] = known;
                pR0[i * 3] = pImg[i * 3] * known;
                pR0[i * 3 + 1] = pImg[i * 3 + 1] * known;
                pR0[i * 3 + 2] = pImg[i * 3 + 2] * known;
            }
        }

        for (int l = 1; l < levels; l++)
        {
            int pw = Math.Max(1, w0 >> l), ph = Math.Max(1, h0 >> l);
            int cw = Math.Max(1, w0 >> (l - 1)), ch = Math.Max(1, h0 >> (l - 1));
            rgbP[l] = new float[pw * ph * 3];
            wP[l] = new float[pw * ph];
            var pr = rgbP[l - 1]; var pwPrev = wP[l - 1];
            var cr = rgbP[l];     var cwArr = wP[l];
            for (int y = 0; y < ph; y++)
            {
                int sy = Math.Min(ch - 1, y * 2);
                int sy2 = Math.Min(ch - 1, y * 2 + 1);
                for (int x = 0; x < pw; x++)
                {
                    int sx = Math.Min(cw - 1, x * 2);
                    int sx2 = Math.Min(cw - 1, x * 2 + 1);
                    int i00 = sy * cw + sx, i01 = sy * cw + sx2;
                    int i10 = sy2 * cw + sx, i11 = sy2 * cw + sx2;
                    float wsum = pwPrev[i00] + pwPrev[i01] + pwPrev[i10] + pwPrev[i11];
                    int o = y * pw + x;
                    cwArr[o] = wsum > 0f ? 1f : 0f;
                    if (wsum <= 0f) continue;
                    float inv = 1f / wsum;
                    for (int c = 0; c < 3; c++)
                        cr[o * 3 + c] = (pr[i00 * 3 + c] + pr[i01 * 3 + c] +
                                        pr[i10 * 3 + c] + pr[i11 * 3 + c]) * inv;
                }
            }
        }

        for (int l = levels - 1; l >= 1; l--)
        {
            int cw = Math.Max(1, w0 >> l), ch = Math.Max(1, h0 >> l);
            int pw = Math.Max(1, w0 >> (l - 1)), ph = Math.Max(1, h0 >> (l - 1));
            var cr = rgbP[l]; var cwr = wP[l];
            var pr = rgbP[l - 1]; var pwr = wP[l - 1];
            for (int y = 0; y < ph; y++)
            {
                float fy = Math.Clamp((y + 0.5f) * ch / ph - 0.5f, 0, ch - 1f);
                int y0 = (int)fy, y1 = Math.Min(y0 + 1, ch - 1);
                float ty = fy - y0;
                for (int x = 0; x < pw; x++)
                {
                    float fx = Math.Clamp((x + 0.5f) * cw / pw - 0.5f, 0, cw - 1f);
                    int x0 = (int)fx, x1 = Math.Min(x0 + 1, cw - 1);
                    float tx = fx - x0;
                    int i00 = y0 * cw + x0, i01 = y0 * cw + x1;
                    int i10 = y1 * cw + x0, i11 = y1 * cw + x1;
                    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty),
                          w10 = (1 - tx) * ty,       w11 = tx * ty;
                    int o = y * pw + x;
                    float known = pwr[o];
                    for (int c = 0; c < 3; c++)
                    {
                        float est = cr[i00 * 3 + c] * w00 + cr[i01 * 3 + c] * w01 +
                                   cr[i10 * 3 + c] * w10 + cr[i11 * 3 + c] * w11;
                        pr[o * 3 + c] = known > 0f ? pr[o * 3 + c] : est;
                    }
                }
            }
        }

        fixed (byte* pImg = img)
        fixed (bool* pH = holes)
        fixed (float* pR0 = rgbP[0])
        {
            for (int i = 0; i < W * H; i++)
            {
                if (!pH[i]) continue;
                pImg[i * 3] = (byte)Math.Clamp(pR0[i * 3], 0f, 255f);
                pImg[i * 3 + 1] = (byte)Math.Clamp(pR0[i * 3 + 1], 0f, 255f);
                pImg[i * 3 + 2] = (byte)Math.Clamp(pR0[i * 3 + 2], 0f, 255f);
            }
        }
    }

    /// <summary>Light 3×3 blur on pixels adjacent to holes to hide seams.</summary>
    private static unsafe void SoftenHoleEdges(byte[] img, bool[] holes, int W, int H)
    {
        var outp = (byte[])img.Clone();
        fixed (byte* pSrc = img, pDst = outp)
        fixed (bool* h = holes)
        {
            for (int y = 1; y < H - 1; y++)
            {
                for (int x = 1; x < W - 1; x++)
                {
                    int i = y * W + x;
                    if (h[i] || h[i - 1] || h[i + 1] || h[i - W] || h[i + W]) continue;
                    for (int c = 0; c < 3; c++)
                    {
                        int sum = 0;
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                                sum += pSrc[((y + dy) * W + (x + dx)) * 3 + c];
                        pDst[i * 3 + c] = (byte)(sum / 9);
                    }
                }
            }
        }
        Buffer.BlockCopy(outp, 0, img, 0, img.Length);
    }
}
