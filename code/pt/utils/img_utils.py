import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import List
import scipy.ndimage.filters as flt
import warnings

def crop(img):
    """
    Crop ROI from image.
    """

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(3,3),1)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w]

def invert_image(image: np.array) -> np.array:
    """
    Invert image
    image: The screening mammography
    returns: Inverted image
    """
    return cv2.bitwise_not(image)

def get_contours(image, thresh_low: int = 5, thresh_high: int = 255) -> List:
    """
    Get the list of contours of an image with opencv

    image: The screening mammography
    thresh_low: Lower bound for the threshold of the image
    thresh_high : Upper bound for the threshold of the image

    returns: The list of contours of the image
    """
    # Perform thresholding to create a binary image
    _, binary = cv2.threshold(image, thresh_low, thresh_high, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def to_gray(image, method_to_gray: str = "default") -> np.array:
    """
    Convert mammography sceening to gray

    image: The image to convert
    method_to_gray: Which method to use to convert to gray (None correponds to the open cv method
    TODO explore other ways to make it gray, e.g PCA cf. Impact of Image Enhancement Module for Analysis
    of Mammogram Images for Diagnostics of Breast Cancer)

    returns: The image in grayscale
    """
    if method_to_gray == "default":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def draw_contours(contours: List, image: np.array, biggest: bool = True) -> tuple:
    """
    Draw contour on images

    contours: The list of the contour of the image
    image: The screening mammography
    biggest: If True, draws the biggest contour, else draw other contours

    returns: image with associate contours, binary mask of the contour
    """

    biggest_contour = max(contours, key=cv2.contourArea)
    # Create a mask image with the same size as the original image
    mask = np.zeros_like(image)
    # Convert the mask image to grayscale
    if len(image.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    if biggest:
        # Draw the biggest contour on the mask image
        # cv2.drawContours(mask_gray, [biggest_contour], -1, (255, 255, 255), -1)
        cv2.drawContours(mask_gray, [biggest_contour], -1, 255, -1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(image, [biggest_contour], -1, (255, 0, 0), 3)

        # Set the pixels inside the removed contours to red
    # image[mask_gray == 255] = 255
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for contour in contours:
            if cv2.contourArea(contour) != cv2.contourArea(biggest_contour):
                cv2.drawContours(mask_gray, [contour], -1, (255, 255, 255), -1)
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
    return image, mask_gray

def gamma_correct(image: np.array, gamma: float = 2.2) -> np.array:
    """
    Gamma correct image

    image: The screening mammography
    gamma: The factor to use to gamma correct image

    returns: Gamma corrected image
    """
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, lut)

def remove_annotation(
        image: np.array, thresh_high: int = 255, thresh_low: int = 5
    ) -> np.array:
        """
        Remove annotation in image

        image: The screening mammography

        returns: Image without annotation
        """
        shape = image.shape
        # Get contours
        contours = get_contours(
            image, thresh_low=thresh_low, thresh_high=thresh_high
        )
        biggest_contour = max(contours, key=cv2.contourArea)
        # Apply mask on biggest contour
        mask = draw_contours(contours, image, biggest=True)[1]
        # # Remove small spaces at the top and bottom of image
        # _, y, _, h = cv2.boundingRect(biggest_contour)
        # image = image[y + 50 : y + h, :]
        # # Resize image to correct shape
        # image = cv2.resize(image, shape)
        return cv2.bitwise_and(image, mask), mask

def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1]/2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1]/2):])
    #print(left_nonzero, right_nonzero)
    if(left_nonzero < right_nonzero):
        image = cv2.flip(image, 1)

    return image

def mirrored_image(
        image: np.array,
        method: str = "prewitt",
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        
        
        # Get edges
        edges = get_edges(image=image, method=method)

        #cv2.imshow('fteste', edges)
        

        edges = remove_useless_edges(
            edges=edges,
            kernel_size_closing=kernel_size_closing,
            thresh_mask_edges=thresh_mask_edges,
            kernel_erosion_shape=kernel_erosion_shape,
        )

        #cv2.imshow('fteste2', edges)
        #cv2.waitKey()

        hull = get_convex_hull(edges=edges)

        if hull is None:
            return image
        mask = np.zeros_like(image)

        # Fill the convex hull with 1's in the mask
        cv2.fillConvexPoly(mask, hull, 1)

        # Apply the binary mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(hull)
        
        merged = cv2.hconcat([cv2.flip(image[y:y+h, x:x+w],1), image[y:y+h, x:x+w]]) 
        #merged = cv2.hconcat([cv2.flip(image,1), image]) 
        return merged

def remove_pectoral_muscle(
        image: np.array,
        method: str = "prewitt",
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        """
        Method to remove pectoral muscle (implementation adapted from
        Removal of pectoral muscle based on topographic map and shape-shifting silhouette)

        image: Screening mammography
        method: The method for edge detection (here prewitt method is used)
        kernel_size_closing: Kernel used for closing method (cf. opencv)
        thresh_mask_edges: Adaptative relative thresholding for mask
        kernel_erosion_shape: Kernel of the erosion

        returns: Image without pectoral muscle
        """
        # Get edges
        edges = get_edges(image=image, method=method)
        edges = remove_useless_edges(
            edges=edges,
            kernel_size_closing=kernel_size_closing,
            thresh_mask_edges=thresh_mask_edges,
            kernel_erosion_shape=kernel_erosion_shape,
        )

        hull = get_convex_hull(edges=edges)

        if hull is None:
            return image
        mask = np.zeros_like(image)

        # Fill the convex hull with 1's in the mask
        cv2.fillConvexPoly(mask, hull, 1)

        # Apply the binary mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)
        return image

def get_convex_hull(edges: np.array) -> np.array:
    """
    Get convex hull from edges of image

    edges: Detected edges from image

    returns: Minimum convex hull (cf. opencv)
    """
    # If the image is fully black
    if edges is None:
        return None
    # Find the non-zero pixels in the image
    points = np.argwhere(edges > 0)

    points = np.array([[elem[1], elem[0]] for elem in points])

    # Calculate the convex hull of the points
    hull = cv2.convexHull(points)

    return hull


def get_edges(image: np.array, method: str = "prewitt") -> np.array:
    """
    Detect edges from an image given a method

    image: Screening mammography
    method: The method for edge detection (here prewitt method is used)

    returns: Edges of the image
    """
    edges = []
    image = image / 255
    if method == "prewitt":
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        img_prewittx = cv2.filter2D(image, -1, kernelx)
        img_prewitty = cv2.filter2D(image, -1, kernely)

        # Calculate the gradient magnitude
        edges = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty))

        # Normalize the gradient magnitude image
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return edges

def remove_useless_edges(
    edges: np.array,
    kernel_size_closing: tuple = (5, 5),
    thresh_mask_edges: float = 0.95,
    kernel_erosion_shape: tuple = (1, 2),
) -> np.array:
    """
    Remove useless edges by adaptive thresholding and small erosion

    edges: Detected edges from image
    kernel_size_closing: Kernel used for closing method (cf. opencv)
    thresh_mask_edges: Adaptative relative thresholding for mask
    kernel_erosion_shape: Kernel of the erosion

    returns: Filtered edges
    """
    # Define the kernel for the closing operation
    kernel = np.ones(kernel_size_closing, np.uint8)

    # Apply the closing operation to the image
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    intensities = closed.flatten()

    # Create a new array of non-zero intensities
    intensities = intensities[intensities > 2]

    # Sort the array of pixel intensities
    intensities.sort()

    # Find the index of the thresh_mask_edges quantile
    index = int(len(intensities) * thresh_mask_edges)

    if index == 0:
        return None
    else:
        # Retrieve the 50th quantile value from the sorted array
        quantile = intensities[index]

    _, edges_thresh = cv2.threshold(closed, quantile, 255, cv2.THRESH_BINARY)

    # Define the kernel for the erosion operation
    kernel = np.ones(kernel_erosion_shape, np.uint8)

    # Apply the erosion operation to the image
    edges_thresh = cv2.erode(edges_thresh, kernel, iterations=1)

    return edges_thresh

def clahe(img, clip=2.0, size=32):
    #contrast enhancement
    if size == None:
        clahe = cv2.createCLAHE(clipLimit=clip)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(size,size)) # depois deixar esse
    
    cl = clahe.apply(np.array(img).astype(np.uint8))
    return cl

def synthesized_images(img):
    """
    Merging of truncation_normalization + clahe1 + clahe2
    """
    breast = crop(img)
    normalized = (breast - np.min(breast))/ (np.max(breast) - np.min(breast))

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)

    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    
    return synthetized

def synthesized_images_whithout_crop(img):
    """
    Merging of truncation_normalization + clahe1 + clahe2
    """
    #breast = crop(img)
    zimg = img.astype(np.float32)
    
    
    normalized = (img - np.min(img))/ (np.max(img) - np.min(img))
    
    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)
    

    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    
    return synthetized

def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
  """
  
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else: 
            deltaSf=deltaS;
            deltaEf=deltaE;
            
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout
