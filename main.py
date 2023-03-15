from sewar.full_ref import ssim, scc, rase, sam, msssim
from image_similarity_measures.quality_metrics import fsim, issm
import cv2
import time
from orb_similarity import orb_similarity

SCANS_FILENAMES = ["Scans/Scan 3.jpg", "Scans/Scan 0.jpg", "Scans/Scan 0 var1.jpg", "Scans/Scan 0 var2.jpg"]
PICTURE_FILENAME = "Scans/Scan 3.jpg"

# Read images
picture = cv2.imread(PICTURE_FILENAME)

scans = []
for scan_filename in SCANS_FILENAMES:
    scans.append(cv2.imread(scan_filename))

# Resize scans to match picture
for i in range(len(scans)):
    scans[i] = cv2.resize(scans[i], (picture.shape[1], picture.shape[0]))

# Calculate metrics for each scan
results = {
    "ssim": [],
    "msssim": [],
    "fsim": [],
    "issm": [],
    "scc": [],
    "rase": [],
    "sam": [],
    "keypoints": [],
}

timer = {
    "ssim": [],
    "msssim": [],
    "fsim": [],
    "issm": [],
    "scc": [],
    "rase": [],
    "sam": [],
    "keypoints": [],
}

for i in range(len(scans)):
    # ORB similarity (keypoints)
    print(f"Calculating ORB similarity for scan {i}...")
    start_time = time.time()
    results["keypoints"].append(orb_similarity(picture, scans[i]))
    timer["keypoints"].append(time.time() - start_time)

    # Feature similarity index (FSIM)
    print(f"Calculating FSIM for scan {i}. This may take up to 5 minutes...")
    start_time = time.time()
    results["fsim"].append(fsim(picture, scans[i]))
    timer["fsim"].append(time.time() - start_time)

    # Convert to grayscale for the following metrics
    print(f"Converting to grayscale for scan {i}...")
    picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    scan_gray = cv2.cvtColor(scans[i], cv2.COLOR_BGR2GRAY)

    # Information theoretic-based statistic similarity (ISSM)
    print(f"Calculating ISSM for scan {i}...")
    start_time = time.time()
    results["issm"].append(issm(picture_gray, scan_gray))
    timer["issm"].append(time.time() - start_time)

    # Structure similarity index (SSIM)
    print(f"Calculating SSIM for scan {i}...")
    start_time = time.time()
    results["ssim"].append(ssim(picture_gray, scan_gray))
    timer["ssim"].append(time.time() - start_time)

    # Multi-scale structural similarity index (MSSSIM)
    print(f"Calculating MSSSIM for scan {i}...")
    start_time = time.time()
    results["msssim"].append(msssim(picture_gray, scan_gray))
    timer["msssim"].append(time.time() - start_time)

    # Spatial correlation coefficient (SCC)
    print(f"Calculating SCC for scan {i}...")
    start_time = time.time()
    results["scc"].append(scc(picture_gray, scan_gray))
    timer["scc"].append(time.time() - start_time)

    # Relative average spectral error (RASE)
    print(f"Calculating RASE for scan {i}...")
    start_time = time.time()
    results["rase"].append(rase(picture_gray, scan_gray))
    timer["rase"].append(time.time() - start_time)

    # Spectral angle mapper (SAM)
    print(f"Calculating SAM for scan {i}...")
    start_time = time.time()
    results["sam"].append(sam(picture_gray, scan_gray))
    timer["sam"].append(time.time() - start_time)


# Print results
print(f"{'Metric':<10}{'Picture':<42}{'Scan 0':<42}{'Scan 0 var1':<42}{'Scan 0 var2':<42}")
for metric in results.keys():
    print(f"{metric:<10}{str(results[metric][0]):<42}{str(results[metric][1]):<42}{str(results[metric][2]):<42}{str(results[metric][3]):<42}")

print()

# Print timers
print(f"{'Metric':<10}{'Picture':<42}{'Scan 0':<42}{'Scan 0 var1':<42}{'Scan 0 var2':<42}")
for metric in timer.keys():
    print(f"{metric:<10}{str(timer[metric][0]):<42}{str(timer[metric][1]):<42}{str(timer[metric][2]):<42}{str(timer[metric][3]):<42}")