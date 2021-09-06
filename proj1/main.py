# list files
import glob, os

# read, write, process images
import numpy as np
import skimage as sk
import skimage.transform
import skimage.io as skio

def read_image(im_name):
    # read in the image
    im = skio.imread(im_name)
    
    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    return im

def write_image(im, im_name):
    skio.imsave(im_name, im)

def metric_score_ncc(image1, image2):
    im1 = (image1 - np.mean(image1)) / np.std(image1, ddof=1)
    im2 = (image2 - np.mean(image2)) / np.std(image2, ddof=1)
    return np.sum(im1*im2) / (im1.shape[0] * im1.shape[1] - 1)

def align_low_resolution(to_be_aligned, reference, metric):    
    best_score = float("-inf")
    best_aligned = None 
    best_x, best_y = -1, -1
                                    
    for x in range(-16, 16):
        for y in range(-16, 16):
            shifted = np.roll(to_be_aligned, x, axis=1)
            shifted = np.roll(shifted, y, axis=0)

            score = metric(shifted, reference)
            
            if score > best_score:
                best_score = score
                best_aligned = shifted
                best_x, best_y = x, y
        
    return best_aligned, best_x, best_y

def align_high_resolution(to_be_aligned, reference, metric):
    best_aligned = None
    previous_best_x, previous_best_y = -1, -1
    for scale in reversed(range(6)):
        to_be_aligned_scaled = sk.transform.rescale(to_be_aligned, (1/2)**scale)
        reference_scaled = sk.transform.rescale(reference, (1/2)**scale)
        
        best_score = float("-inf")
        best_x, best_y = -1, -1
        
        for x in range(-5, 5):
            for y in range(-5, 5):
                xx, yy = x, y
                if scale != 5:
                    xx = previous_best_x * 2 + xx
                    yy = previous_best_y * 2 + yy
                    
                shifted = np.roll(to_be_aligned_scaled, xx, axis=1)
                shifted = np.roll(shifted, yy, axis=0)

                score = metric_score_ncc(shifted, reference_scaled)

                if score > best_score:
                    best_score = score
                    best_x, best_y = xx, yy
                    best_aligned = shifted
        previous_best_x, previous_best_y = best_x, best_y
        
    return best_aligned, previous_best_x, previous_best_y

def produce(image_file_path, align_func, metric):
    im = read_image(image_file_path)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # David crops images
    height, width = b.shape
    b = b[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    g = g[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    r = r[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    
    # align the images
    ag, gx, gy = align_func(g, b, metric)
    ar, rx, ry = align_func(r, b, metric)
    
    # create a color image
    im_out = np.dstack([ar, ag, b])
    
    return im_out, (gx, gy), (rx, ry) 

def produce_emir(image_file_path, align_func, metric):
    im = read_image(image_file_path)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # David crops images
    height, width = b.shape
    b = b[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    g = g[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    r = r[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
    
    # align the images
    ab, bx, by = align_func(b, g, metric)
    ar, rx, ry = align_func(r, g, metric)
    
    # create a color image
    im_out = np.dstack([ar, ab, g])
    
    return im_out, (bx, by), (rx, ry) 

def main():
    for file in os.listdir("data"):
        # read images with low resolution
        if file.endswith(".jpg"):
            im_in_path = os.path.join("data", file)
            im_out_path = "out/" + file
            print(im_out_path)
            im_out, (gx, gy), (rx, ry) = produce(im_in_path, align_low_resolution, metric_score_ncc)
            f = open(im_out_path + ".displacement.txt", "w")
            f.write("green:" + str((gx, gy)) + ",")
            f.write("red:" + str((rx, ry)))
            f.close()
            write_image(im_out, im_out_path)
            
        # read images with high resolution
        elif file.endswith(".tif"):
            im_in_path = os.path.join("data", file)
            im_out_path = "out/" + file[:-len(".tif")] + ".jpg"
            print(im_out_path)
            if "emir" in im_out_path:
                im_out, (bx, by), (rx, ry) = produce_emir(im_in_path, align_high_resolution, metric_score_ncc)
                f = open(im_out_path + ".displacement.txt", "w")
                f.write("blue:" + str((bx, by)) + ",")
                f.write("red:" + str((rx, ry)))
                f.close()
                write_image(im_out, im_out_path)
            else:
                im_out, (gx, gy), (rx, ry) = produce(im_in_path, align_high_resolution, metric_score_ncc)
                f = open(im_out_path + ".displacement.txt", "w")
                f.write("green:" + str((gx, gy)) + ",")
                f.write("red:" + str((rx, ry)))
                f.close()
                write_image(im_out, im_out_path)
                   
main()
