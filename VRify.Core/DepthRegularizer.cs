namespace VRify.Core;

/// <summary>
/// Edge-preserving regularization of the depth map: guided filter (He, Sun, Tang — 2013)
/// with the RGB frame as guide.
///
/// Inside objects (textured flat surfaces) depth is flattened → removes the "wavy face"
/// texture-as-geometry artifact; at silhouettes the depth edge stays sharp because the
/// guide has an edge there. O(N) via integral images. Buffers are cached per resolution
/// (called twice per frame at different sizes — no per-frame reallocation) and all
/// per-pixel passes run in parallel.
/// </summary>
public sealed class DepthRegularizer
{
    private readonly float _eps;
    private sealed class Buffers
    {
        public int W, H;
        public float[] Luma = Array.Empty<float>();
        public float[] A = Array.Empty<float>();
        public float[] B = Array.Empty<float>();
        public float[] Ii1 = Array.Empty<float>();  // ∫ I
        public float[] Ii2 = Array.Empty<float>();  // ∫ I²
        public float[] Ii3 = Array.Empty<float>();  // ∫ P  (reused: ∫ b)
        public float[] Ii4 = Array.Empty<float>();  // ∫ I·P (reused: ∫ a)
    }

    private readonly Dictionary<(int w, int h), Buffers> _cache = new();

    public int Radius { get; set; }

    public DepthRegularizer(int radius = 12, float eps = 0.002f)
    {
        Radius = Math.Max(1, radius);
        _eps = eps;
    }

    public void Apply(float[] depth, byte[] guideRgb, int w, int h, int stride)
    {
        var b = Get(w, h);
        int n = w * h;
        int r = Radius;

        // Luma guide 0..1
        Parallel.For(0, h, y =>
        {
            int row = y * w, sRow = y * stride;
            for (int x = 0; x < w; x++)
            {
                int p = sRow + x * 3;
                b.Luma[row + x] = (0.299f * guideRgb[p] + 0.587f * guideRgb[p + 1] +
                                   0.114f * guideRgb[p + 2]) / 255f;
            }
        });

        BuildIntegral(b.Luma, w, h, b.Ii1);
        BuildIntegral2(b.Luma, w, h, b.Ii2);
        BuildIntegral(depth, w, h, b.Ii3);
        BuildProductIntegral(b.Luma, depth, w, h, b.Ii4);

        // Pass 1: a = cov(I,P) / (var(I) + eps)   — rows independent (integral is read-only)
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                float mi = BoxMean(b.Ii1, w, h, r, x, y);
                float mi2 = BoxMean(b.Ii2, w, h, r, x, y);
                float mp = BoxMean(b.Ii3, w, h, r, x, y);
                float mip = BoxMean(b.Ii4, w, h, r, x, y);
                b.A[i] = (mip - mi * mp) / (mi2 - mi * mi + _eps);
            }
        });

        // b = mean(P) − a·mean(I)
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                float mi = BoxMean(b.Ii1, w, h, r, x, y);
                float mp = BoxMean(b.Ii3, w, h, r, x, y);
                b.B[i] = mp - b.A[i] * mi;
            }
        });

        // Pass 2: q = mean(a)·I + mean(b)   (∫a → Ii4, ∫b → Ii3, P/I·P no longer needed)
        BuildIntegral(b.A, w, h, b.Ii4);
        BuildIntegral(b.B, w, h, b.Ii3);

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                float ma = BoxMean(b.Ii4, w, h, r, x, y);
                float mb = BoxMean(b.Ii3, w, h, r, x, y);
                depth[i] = Math.Clamp(ma * b.Luma[i] + mb, 0f, 1f);
            }
        });
    }

    private Buffers Get(int w, int h)
    {
        if (_cache.TryGetValue((w, h), out var b)) return b;
        b = new Buffers
        {
            W = w, H = h,
            Luma = new float[w * h],
            A = new float[w * h],
            B = new float[w * h],
            Ii1 = new float[(w + 1) * (h + 1)],
            Ii2 = new float[(w + 1) * (h + 1)],
            Ii3 = new float[(w + 1) * (h + 1)],
            Ii4 = new float[(w + 1) * (h + 1)],
        };
        // Bound the cache (pipeline uses exactly 2 resolutions)
        if (_cache.Count > 4) _cache.Clear();
        _cache[(w, h)] = b;
        return b;
    }

    // Integral images: row-sequential (cheap, memory-bound); per-pixel BoxMean passes above
    // are the hot part and they are parallel now.
    private static void BuildIntegral(float[] src, int w, int h, float[] ii)
    {
        int iw = w + 1;
        Array.Clear(ii, 0, iw * (h + 1));
        for (int y = 0; y < h; y++)
        {
            float row = 0f;
            int srcRow = y * w, dstRow = (y + 1) * iw, prevRow = y * iw;
            for (int x = 0; x < w; x++)
            {
                row += src[srcRow + x];
                ii[dstRow + x + 1] = ii[prevRow + x + 1] + row;
            }
        }
    }

    private static void BuildIntegral2(float[] src, int w, int h, float[] ii)
    {
        int iw = w + 1;
        Array.Clear(ii, 0, iw * (h + 1));
        for (int y = 0; y < h; y++)
        {
            float row = 0f;
            int srcRow = y * w, dstRow = (y + 1) * iw, prevRow = y * iw;
            for (int x = 0; x < w; x++)
            {
                float v = src[srcRow + x];
                row += v * v;
                ii[dstRow + x + 1] = ii[prevRow + x + 1] + row;
            }
        }
    }

    private static void BuildProductIntegral(float[] a, float[] b, int w, int h, float[] ii)
    {
        int iw = w + 1;
        Array.Clear(ii, 0, iw * (h + 1));
        for (int y = 0; y < h; y++)
        {
            float row = 0f;
            int aRow = y * w, dstRow = (y + 1) * iw, prevRow = y * iw;
            for (int x = 0; x < w; x++)
            {
                row += a[aRow + x] * b[aRow + x];
                ii[dstRow + x + 1] = ii[prevRow + x + 1] + row;
            }
        }
    }

    /// <summary>Mean over the radius-box centered at (x,y), clamped at borders. No divisions per call beyond the final one.</summary>
    private static float BoxMean(float[] ii, int w, int h, int r, int x, int y)
    {
        int iw = w + 1;
        int x0 = x - r; if (x0 < 0) x0 = 0;
        int x1 = x + r; if (x1 >= w) x1 = w - 1;
        int y0 = y - r; if (y0 < 0) y0 = 0;
        int y1 = y + r; if (y1 >= h) y1 = h - 1;
        float s = ii[(y1 + 1) * iw + (x1 + 1)]
                - ii[y0 * iw + (x1 + 1)]
                - ii[(y1 + 1) * iw + x0]
                + ii[y0 * iw + x0];
        return s / ((x1 - x0 + 1) * (y1 - y0 + 1));
    }
}
