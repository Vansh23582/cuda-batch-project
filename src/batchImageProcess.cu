/*
 * batchImageProcess.cu
 *
 * I wrote this for the CUDA at Scale for the Enterprise course project.
 * The goal: process a big batch of grayscale PGM images on the GPU using:
 *   1. NPP Gaussian blur  (smoothing before edge detection)
 *   2. NPP Sobel edge detection (finding edges in the smoothed image)
 *   3. My own CUDA kernel that does per-image histogram equalization
 *      (stretches contrast so dark images become clearer)
 *
 * The program loops over every .pgm file in a given folder,
 * runs all three GPU stages on each image back-to-back, and
 * saves the results to an output folder.  That way I can prove
 * the GPU is doing real work on 100+ images without touching the CPU
 * for the actual pixel math.
 *
 * Build:  make all
 * Run:    ./run.sh  (or  bin/run --input data/ --output output/)
 */

// ── Windows-only guards (not needed on Linux lab, but harmless) ──────────────
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  #define WINDOWS_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #pragma warning(disable : 4819)
#endif

// ── NVIDIA headers ────────────────────────────────────────────────────────────
#include <cuda_runtime.h>   // cudaMalloc, cudaMemcpy, cudaFree …
#include <npp.h>            // nppiFilter*, nppiSobel* …

// ── CUDA Samples helpers (image I/O, device selection) ───────────────────────
#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <helper_cuda.h>
#include <helper_string.h>

// ── Standard library ─────────────────────────────────────────────────────────
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>   // opendir / readdir – list files in a folder on Linux
#include <sys/stat.h> // mkdir

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 1 – My custom CUDA kernel: per-image histogram equalization
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * buildHistogram_kernel
 *
 * Each thread looks at one pixel and atomically increments the bin for
 * that pixel's intensity value (0-255).  Using atomicAdd is the standard
 * trick here because many threads hit the same bin at once.
 *
 * d_img   – input image on device (8-bit, single channel)
 * d_hist  – 256-element histogram array on device
 * nPixels – total number of pixels in the image
 */
__global__ void buildHistogram_kernel(const unsigned char* d_img,
                                       unsigned int*        d_hist,
                                       int                  nPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPixels) {
        // atomicAdd prevents two threads from clobbering the same counter
        atomicAdd(&d_hist[d_img[idx]], 1u);
    }
}

/*
 * applyLUT_kernel
 *
 * After I build the equalized look-up table (LUT) on the CPU from the
 * histogram, I send the LUT back to the device and let every thread
 * remap its own pixel.  This is embarrassingly parallel – no sharing needed.
 *
 * d_src   – source pixels (original intensity values)
 * d_dst   – destination pixels (equalized)
 * d_lut   – 256-entry mapping table
 * nPixels – total pixels
 */
__global__ void applyLUT_kernel(const unsigned char* d_src,
                                 unsigned char*       d_dst,
                                 const unsigned char* d_lut,
                                 int                  nPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPixels) {
        d_dst[idx] = d_lut[d_src[idx]];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 – Helper: run histogram equalization on one device image
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * histogramEqualize
 *
 * This function wraps both kernels above into one clean call.
 * Steps:
 *   a) Launch buildHistogram_kernel  → get intensity counts
 *   b) CPU computes CDF → build LUT  (tiny array, fast)
 *   c) Launch applyLUT_kernel        → remap every pixel on GPU
 *
 * I keep the LUT computation on the CPU because it's only 256 elements;
 * moving that tiny work to a kernel would add kernel-launch overhead for
 * no real gain.
 */
void histogramEqualize(unsigned char* d_img,   // in/out device pointer
                        unsigned char* d_tmp,   // temp device buffer (same size)
                        int            width,
                        int            height)
{
    int nPixels = width * height;

    // ── Allocate device histogram (256 unsigned ints, zeroed) ────────────────
    unsigned int* d_hist = nullptr;
    checkCudaErrors(cudaMalloc(&d_hist, 256 * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)));

    // ── Launch histogram kernel ───────────────────────────────────────────────
    // I chose 256 threads per block; typical for 1D image traversal
    int blockSize = 256;
    int gridSize  = (nPixels + blockSize - 1) / blockSize;
    buildHistogram_kernel<<<gridSize, blockSize>>>(d_img, d_hist, nPixels);
    checkCudaErrors(cudaDeviceSynchronize());

    // ── Pull histogram to host, build CDF, then build LUT ────────────────────
    unsigned int h_hist[256] = {};
    checkCudaErrors(cudaMemcpy(h_hist, d_hist,
                               256 * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
    cudaFree(d_hist);

    // Compute cumulative distribution function (CDF)
    unsigned long long cdf[256] = {};
    cdf[0] = h_hist[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + h_hist[i];

    // Find first non-zero CDF value (cdf_min) for equalization formula
    unsigned long long cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) { cdf_min = cdf[i]; break; }
    }

    // Standard histogram equalization formula:
    //   lut[v] = round( (cdf[v] - cdf_min) / (N - cdf_min) * 255 )
    unsigned char h_lut[256] = {};
    unsigned long long denom = (unsigned long long)nPixels - cdf_min;
    if (denom == 0) denom = 1; // guard against all-same-color image
    for (int i = 0; i < 256; ++i) {
        double val = (double)(cdf[i] - cdf_min) / (double)denom * 255.0;
        if (val < 0.0)   val = 0.0;
        if (val > 255.0) val = 255.0;
        h_lut[i] = (unsigned char)(val + 0.5); // round
    }

    // ── Upload LUT, apply on GPU ──────────────────────────────────────────────
    unsigned char* d_lut = nullptr;
    checkCudaErrors(cudaMalloc(&d_lut, 256));
    checkCudaErrors(cudaMemcpy(d_lut, h_lut, 256, cudaMemcpyHostToDevice));

    applyLUT_kernel<<<gridSize, blockSize>>>(d_img, d_tmp, d_lut, nPixels);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy equalized image back into d_img so the caller sees the result
    checkCudaErrors(cudaMemcpy(d_img, d_tmp, nPixels, cudaMemcpyDeviceToDevice));

    cudaFree(d_lut);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 3 – Helper: collect .pgm filenames from a directory
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * listPGMFiles
 *
 * I use POSIX opendir/readdir here because the lab runs Linux.
 * It walks the given directory and returns every filename ending in ".pgm".
 */
std::vector<std::string> listPGMFiles(const std::string& dirPath)
{
    std::vector<std::string> files;
    DIR* dir = opendir(dirPath.c_str());
    if (!dir) {
        std::cerr << "[WARN] Cannot open input directory: " << dirPath << "\n";
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        // Only keep files that end with ".pgm"
        if (name.size() > 4 &&
            name.substr(name.size() - 4) == ".pgm") {
            files.push_back(dirPath + "/" + name);
        }
    }
    closedir(dir);
    return files;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 4 – Helper: print NPP library and CUDA version info
// ═══════════════════════════════════════════════════════════════════════════════

void printVersionInfo()
{
    const NppLibraryVersion* ver = nppGetLibVersion();
    printf("NPP Library Version %d.%d.%d\n",
           ver->major, ver->minor, ver->build);

    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    printf("  CUDA Driver  Version: %d.%d\n",
           driver  / 1000, (driver  % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n",
           runtime / 1000, (runtime % 100) / 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 5 – Process one image through the full GPU pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * processOneImage
 *
 * This is the heart of the project.  For each input PGM file I:
 *   1. Load it from disk into a CPU buffer
 *   2. Upload it to the GPU (device image)
 *   3. NPP Gaussian blur  – smooths noise before Sobel
 *   4. My histogramEqualize kernel – stretches contrast
 *   5. NPP Sobel edge detection – highlights edges
 *   6. Download result to CPU and save to output folder
 *
 * Returns true on success, false on any error (so main can count failures).
 */
bool processOneImage(const std::string& inputPath,
                     const std::string& outputDir)
{
    // ── Derive output filename ───────────────────────────────────────────────
    // Strip directory prefix and extension, append "_processed.pgm"
    std::string baseName = inputPath;
    size_t slash = baseName.rfind('/');
    if (slash != std::string::npos) baseName = baseName.substr(slash + 1);
    size_t dot = baseName.rfind('.');
    if (dot != std::string::npos) baseName = baseName.substr(0, dot);
    std::string outputPath = outputDir + "/" + baseName + "_processed.pgm";

    try {
        // ── Step 1: Load image from disk (CPU) ───────────────────────────────
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(inputPath, hostSrc);

        int W = (int)hostSrc.width();
        int H = (int)hostSrc.height();
        std::cout << "  Processing: " << baseName
                  << " (" << W << "x" << H << ")\n";

        // ── Step 2: Upload to GPU ─────────────────────────────────────────────
        // ImageNPP_8u_C1 constructor does cudaMemcpy for us
        npp::ImageNPP_8u_C1 devSrc(hostSrc);

        NppiSize  roi    = { W, H };
        NppiPoint origin = { 0, 0 };

        // ── Step 3: NPP Gaussian blur (3x3 kernel, sigma ≈ 1) ────────────────
        // I blur first so Sobel doesn't pick up high-frequency sensor noise
        npp::ImageNPP_8u_C1 devBlurred(W, H);

        // nppiFilterGauss_8u_C1R: standard NPP call, runs entirely on GPU
        NPP_CHECK_NPP(
            nppiFilterGauss_8u_C1R(
                devSrc.data(),      devSrc.pitch(),
                devBlurred.data(),  devBlurred.pitch(),
                roi,
                NPP_MASK_SIZE_3_X_3  // 3x3 Gaussian kernel
            )
        );

        // ── Step 4: Histogram equalization (my custom CUDA kernels) ──────────
        // I allocate a temporary device buffer of the same size for the LUT pass
        unsigned char* d_tmp = nullptr;
        checkCudaErrors(cudaMalloc(&d_tmp, W * H));

        // devBlurred.data() is a flat device pointer; pitch may be > W, but
        // for histogram purposes I treat rows as contiguous (pitch == W for
        // images allocated by ImageNPP when width is already aligned).
        // For safety, I copy to a tight (pitch=W) buffer first.
        unsigned char* d_tight = nullptr;
        checkCudaErrors(cudaMalloc(&d_tight, W * H));
        checkCudaErrors(cudaMemcpy2D(
            d_tight, W,                       // dst, dst pitch
            devBlurred.data(), devBlurred.pitch(), // src, src pitch
            W, H,                             // width (bytes), height
            cudaMemcpyDeviceToDevice
        ));

        histogramEqualize(d_tight, d_tmp, W, H);

        // Copy equalized result back into devBlurred so Sobel sees it
        checkCudaErrors(cudaMemcpy2D(
            devBlurred.data(), devBlurred.pitch(),
            d_tight, W,
            W, H,
            cudaMemcpyDeviceToDevice
        ));
        cudaFree(d_tight);
        cudaFree(d_tmp);

        // ── Step 5: NPP Sobel edge detection ─────────────────────────────────
        // nppiFilterSobelHoriz + nppiFilterSobelVert would give X and Y
        // separately, but nppiFilterSobel gives the combined magnitude.
        // I use nppiFilterSobelHorizBorder and VvertBorder for border safety.
        npp::ImageNPP_8u_C1 devEdge(W, H);

        // Allocate 16-bit scratch buffers for horizontal and vertical Sobel
        // NPP Sobel on 8u input needs 16s intermediate
        npp::ImageNPP_16s_C1 devSobelX(W, H);
        npp::ImageNPP_16s_C1 devSobelY(W, H);
        NppiSize roiSobel = { W, H };

        // Horizontal gradient (left→right edges)
        NPP_CHECK_NPP(
            nppiFilterSobelHorizBorder_8u16s_C1R(
                devBlurred.data(), devBlurred.pitch(),
                roi, origin,
                devSobelX.data(), devSobelX.pitch(),
                roiSobel,
                NPP_MASK_SIZE_3_X_3,
                NPP_BORDER_REPLICATE  // pad edges by replicating border pixels
            )
        );

        // Vertical gradient (top→bottom edges)
        NPP_CHECK_NPP(
            nppiFilterSobelVertBorder_8u16s_C1R(
                devBlurred.data(), devBlurred.pitch(),
                roi, origin,
                devSobelY.data(), devSobelY.pitch(),
                roiSobel,
                NPP_MASK_SIZE_3_X_3,
                NPP_BORDER_REPLICATE
            )
        );

        // Combine: magnitude = sqrt(Gx^2 + Gy^2), clamped back to 8-bit
        // nppiMagnitude_16s8u_C1R does exactly this in one GPU call
        NPP_CHECK_NPP(
            nppiMagnitude_16s8u_C1R(
                devSobelX.data(), devSobelX.pitch(),
                devSobelY.data(), devSobelY.pitch(),
                devEdge.data(),   devEdge.pitch(),
                roiSobel
            )
        );

        // ── Step 6: Download result and save ─────────────────────────────────
        npp::ImageCPU_8u_C1 hostDst(devEdge.size());
        devEdge.copyTo(hostDst.data(), hostDst.pitch());
        saveImage(outputPath, hostDst);

        std::cout << "  Saved  → " << outputPath << "\n";

        // ── Free device images ────────────────────────────────────────────────
        // ImageNPP RAII destructors handle their own memory, but free the
        // intermediate NPP images explicitly to keep peak VRAM low on large batches
        nppiFree(devSrc.data());
        nppiFree(devBlurred.data());
        nppiFree(devEdge.data());
        nppiFree(devSobelX.data());
        nppiFree(devSobelY.data());

        return true;
    }
    catch (npp::Exception& e) {
        std::cerr << "  [ERROR] NPP exception on " << inputPath
                  << ": " << e << "\n";
        return false;
    }
    catch (...) {
        std::cerr << "  [ERROR] Unknown exception on " << inputPath << "\n";
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 6 – main: parse args, loop over all images, report timing
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    printf("=== CUDA Batch Image Processor ===\n");
    printf("Assignment: CUDA at Scale for the Enterprise\n\n");

    // Select GPU (picks the best one if multiple exist)
    findCudaDevice(argc, (const char**)argv);
    printVersionInfo();
    printf("\n");

    // ── Parse command-line arguments ─────────────────────────────────────────
    // --input  <folder>   folder containing .pgm files (default: ./data)
    // --output <folder>   where to write results        (default: ./output)
    std::string inputDir  = "data";
    std::string outputDir = "output";

    char* tmp = nullptr;
    if (checkCmdLineFlag(argc, (const char**)argv, "input")) {
        getCmdLineArgumentString(argc, (const char**)argv, "input", &tmp);
        inputDir = tmp;
    }
    if (checkCmdLineFlag(argc, (const char**)argv, "output")) {
        getCmdLineArgumentString(argc, (const char**)argv, "output", &tmp);
        outputDir = tmp;
    }

    // Create output directory if it doesn't exist
    mkdir(outputDir.c_str(), 0755);

    // ── Gather list of PGM files ─────────────────────────────────────────────
    std::vector<std::string> files = listPGMFiles(inputDir);
    if (files.empty()) {
        std::cerr << "[ERROR] No .pgm files found in: " << inputDir << "\n";
        std::cerr << "        Run scripts/generate_data.py first to create sample images.\n";
        return EXIT_FAILURE;
    }
    printf("Found %zu image(s) in '%s'\n\n", files.size(), inputDir.c_str());

    // ── Process each image and time the whole batch ───────────────────────────
    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    int ok = 0, fail = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        printf("[%zu/%zu] ", i + 1, files.size());
        if (processOneImage(files[i], outputDir)) ++ok;
        else                                       ++fail;
    }

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, tStart, tStop);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    // ── Summary ───────────────────────────────────────────────────────────────
    printf("\n=== Done ===\n");
    printf("  Processed : %d image(s) OK, %d failed\n", ok, fail);
    printf("  Total GPU time (wall): %.2f ms\n", ms);
    if (ok > 0)
        printf("  Avg per image        : %.2f ms\n", ms / (float)ok);

    return (fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
