
import cv2

def orb_similarity(image1, image2):
    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect and compute ORB features and descriptors.
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match features.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Only extract matches with a low distance value.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    
    # Return the ratio of similar_matches and total_matches.
    return len(similar_regions) / len(matches)